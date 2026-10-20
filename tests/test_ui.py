"""Tests for the local web UI.

The server is started on port 0 (a free port) in a thread, so the tests exercise
the real HTTP surface rather than the handler functions.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

import pytest
from PIL import Image

from fractalgen import ui
from fractalgen.config import Config


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    httpd = ui.create_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield base
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _get(base: str, path: str) -> tuple[int, str, bytes]:
    with urllib.request.urlopen(base + path, timeout=60) as response:
        return response.status, response.headers.get("Content-Type", ""), response.read()


def _post(base: str, path: str, payload: Any) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _leaf_paths(value: Any, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        found: set[str] = set()
        for key, item in value.items():
            found |= _leaf_paths(item, f"{prefix}.{key}" if prefix else key)
        return found
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return {prefix}
    return {prefix}


def test_index_serves_the_page(server: str):
    status, content_type, body = _get(server, "/")
    assert status == 200
    assert "text/html" in content_type
    page = body.decode()
    assert "fractalgen" in page
    assert "/api/schema" in page and "/api/render" in page
    assert "fetch('/api/render'" in page or 'fetch("/api/render"' in page


def test_schema_covers_every_config_leaf(server: str):
    status, content_type, body = _get(server, "/api/schema")
    assert status == 200 and "json" in content_type

    schema = json.loads(body)
    described = {field["path"] for field in schema["fields"]}

    expected = _leaf_paths(Config().to_dict())
    expected.discard("output")  # the server owns the output path
    missing = expected - described
    assert not missing, f"unexposed config fields: {sorted(missing)}"

    for field in schema["fields"]:
        assert {"path", "label", "kind", "section"} <= set(field)
        if field["kind"] == "enum":
            assert field["choices"], field["path"]

    assert {"fractal", "colour", "image", "video", "runtime"} <= {section["id"] for section in schema["sections"]}
    assert schema["defaults"]["runtime"]["workers"] == 1


def test_render_returns_a_served_png(server: str):
    status, payload = _post(
        server,
        "/api/render",
        {"form": "image", "config": {"fractal": {"size": 48, "iterations": 12}}},
    )
    assert status == 200, payload
    assert payload["ok"] is True
    assert payload["url"].startswith("/outputs/") and payload["url"].endswith(".png")
    assert payload["elapsed"] >= 0

    status, content_type, body = _get(server, payload["url"])
    assert status == 200
    assert content_type == "image/png"
    assert body[:8] == b"\x89PNG\r\n\x1a\n"
    assert Image.open(io.BytesIO(body)).size == (48, 48)


def test_render_accepts_a_full_config(server: str):
    config = Config.from_dict(
        {
            "fractal": {
                "size": 48,
                "iterations": 15,
                "polynomial": {"mode": "roots", "roots": [[1, 0], [-1, 0], [0, 1], [0, -1]]},
            },
            "colour": {"mode": "matplotlib", "name": "magma"},
            "image": {"return_type": "roots", "postprocess": []},
        }
    ).to_dict()
    config.pop("output")
    status, payload = _post(server, "/api/render", {"form": "image", "config": config})
    assert status == 200, payload
    assert payload["ok"] is True


def test_bad_config_reports_the_field(server: str):
    status, payload = _post(server, "/api/render", {"form": "image", "config": {"fractal": {"size": 4}}})
    assert status == 400
    assert payload["ok"] is False
    assert "fractal.size" in payload["error"]

    status, payload = _post(server, "/api/render", {"form": "image", "config": {"fractal": {"nope": 1}}})
    assert status == 400
    assert "unknown key" in payload["error"]

    status, payload = _post(server, "/api/render", {"form": "nope", "config": {}})
    assert status == 400
    assert "form" in payload["error"]


def test_output_names_reject_traversal(server: str):
    for attempt in ("/outputs/..%2f..%2fetc%2fpasswd", "/outputs/%2e%2e%2fsecret.png", "/outputs/nope.png"):
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(server + attempt, timeout=30)
        assert error.value.code in (400, 404)


def test_safe_name_helper():
    assert ui._safe_name("fractal-123.png") == "fractal-123.png"
    for bad in ("../x.png", "a/b.png", "..", "x y.png", ""):
        with pytest.raises(ValueError):
            ui._safe_name(bad)


def _png_bytes(size: int = 32) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), (40, 90, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_from_image_returns_a_palette(server: str):
    encoded = base64.b64encode(_png_bytes()).decode()
    status, payload = _post(server, "/api/from-image", {"filename": "flat.png", "data": encoded, "colours": 3})
    assert status == 200, payload
    assert payload["ok"] is True
    assert payload["colours"] and all(colour.startswith("#") for colour in payload["colours"])
    assert payload["size"] == [32, 32]
    assert payload["suggested"]["colour"]["mode"] == "colors"
    assert payload["suggested"]["fractal"]["map"]["strength"] > 0

    status, _, raster = _get(server, payload["raster_url"])
    assert status == 200 and raster[:8] == b"\x89PNG\r\n\x1a\n"


def test_from_image_rejects_garbage(server: str):
    status, payload = _post(server, "/api/from-image", {"filename": "x.png", "data": "not base64!!"})
    assert status == 400
    assert "base64" in payload["error"]

    status, payload = _post(server, "/api/from-image", {"filename": "../escape.png", "data": "AAAA"})
    assert status == 400
    assert "invalid file name" in payload["error"]


def test_health(server: str):
    status, _, body = _get(server, "/api/health")
    assert status == 200
    assert json.loads(body)["ok"] is True


def test_cli_ui_subcommand_is_wired():
    from fractalgen.cli import build_parser

    arguments = build_parser().parse_args(["ui", "--port", "0", "--no-open"])
    assert arguments.command == "ui"
    assert arguments.port == 0
    assert arguments.host == "127.0.0.1"
