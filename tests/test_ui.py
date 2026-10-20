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
from fractalgen.config import Config, ConfigError
from fractalgen.ui import PAGE, _OUTPUT_DIR, _render, build_schema, import_image


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

    # A traversal attempt is folded into a plain name, and the payload is then
    # rejected for not being an image - never written outside the output dir.
    status, payload = _post(server, "/api/from-image", {"filename": "../escape.png", "data": "AAAA"})
    assert status == 400
    assert "escape.png" in payload["error"] or "unsupported" in payload["error"]


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


def test_schema_labels_are_unique_and_choices_are_labelled():
    schema = build_schema()
    labels = [field["label"] for field in schema["fields"]]
    assert len(labels) == len(set(labels)), "two controls share a label"
    # the two "smooth" knobs must be distinguishable
    assert any("Shading" in label and "smooth" in label for label in labels)
    mode = next(field for field in schema["fields"] if field["path"] == "colour.mode")
    assert all(isinstance(choice, list) and len(choice) == 2 for choice in mode["choices"])
    assert ["accents", "Extract from an image + accents"] in mode["choices"]


def test_schema_colour_name_lists_real_colormaps():
    schema = build_schema()
    name = next(field for field in schema["fields"] if field["path"] == "colour.name")
    values = [choice[0] for choice in name["choices"]]
    assert "viridis" in values and "rocket" in values and len(values) > 50


def test_video_sections_are_form_scoped():
    sections = {section["id"]: section for section in build_schema()["sections"]}
    assert sections["video"]["forms"] == ["video"]
    assert sections["image"]["forms"] == ["image", "video"]
    assert sections["colour"]["forms"] == ["image", "video"]


def test_mp4_falls_back_to_gif_without_ffmpeg():
    from fractalgen import optional

    config = {
        "fractal": {"size": 32, "iterations": 6, "polynomial": {"mode": "roots", "roots": [[1, 0], [-1, 0], [0, 1]]}},
        "video": {"fps": 4, "frames": 2, "codec": "mp4", "animations": ["alpha"]},
    }
    result = _render({"form": "video", "config": config})
    if optional.available("mp4"):
        assert result["codec"] == "mp4"
        assert result["url"].endswith(".mp4")
    else:
        assert result["codec"] == "gif"
        assert result["url"].endswith(".gif")
        assert (Path(_OUTPUT_DIR) / result["filename"]).exists()


def test_unsupported_upload_format_has_a_helpful_message(tmp_path):
    import base64

    payload = {"filename": "photo.heic", "data": base64.b64encode(b"definitely not an image").decode()}
    with pytest.raises(ConfigError) as error:
        import_image(payload)
    assert "unsupported image format" in str(error.value)
    assert "JPEG or PNG" in str(error.value)


def test_preview_offers_a_download():
    assert "download:" in PAGE
    assert "Download ${name}" in PAGE


def test_outputs_endpoint_lists_renders(server: str):
    status, payload = _post(server, "/api/render", {"form": "image", "config": _tiny_config()})
    assert status == 200
    listing = json.loads(_get(server, "/api/outputs")[2])["outputs"]
    assert any(entry["name"] == payload["filename"] for entry in listing)
    entry = next(item for item in listing if item["name"] == payload["filename"])
    assert entry["url"] == payload["url"]
    assert entry["kind"] == "image"
    assert entry["bytes"] > 0


def test_outputs_endpoint_is_traversal_safe(server: str):
    listing = json.loads(_get(server, "/api/outputs")[2])["outputs"]
    assert all("/" not in entry["name"] and ".." not in entry["name"] for entry in listing)


def _tiny_config() -> dict:
    return {
        "fractal": {
            "size": 32,
            "iterations": 8,
            "polynomial": {"mode": "roots", "roots": [[1, 0], [-1, 0], [0, 1]]},
        },
        "colour": {"mode": "matplotlib", "name": "viridis"},
    }


def test_page_has_an_outputs_picker():
    assert 'id="outputs"' in PAGE
    assert "refreshOutputs" in PAGE
    assert "/api/outputs" in PAGE


def test_import_accepts_awkward_file_names():
    from fractalgen.ui import _slug_name

    assert _slug_name("pink house.jpg") == "pink_house.jpg"
    assert _slug_name("C:\\Photos\\pink house.JPG") == "pink_house.jpg"
    assert _slug_name("../../etc/passwd") == "passwd"
    assert _slug_name("") == "upload"
    assert _slug_name("naïve café.png").endswith(".png")


def test_import_image_with_spaces_in_the_name(server: str):
    import base64
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (48, 48), (200, 60, 120)).save(buffer, format="PNG")
    payload = {
        "filename": "pink house.png",
        "data": base64.b64encode(buffer.getvalue()).decode(),
        "colours": 3,
        "raster_size": 32,
    }
    status, body = _post(server, "/api/from-image", payload)
    assert status == 200, body
    assert body["ok"] is True
    assert body["colours"]
    assert "@" not in body["raster_url"]
    assert " " not in body["raster_url"]


def test_import_returns_a_server_path_and_the_render_accepts_a_url(server: str):
    import base64
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (40, 40), (90, 140, 200)).save(buffer, format="PNG")
    status, body = _post(
        server,
        "/api/from-image",
        {"filename": "blue sky.png", "data": base64.b64encode(buffer.getvalue()).decode(), "colours": 3},
    )
    assert status == 200, body
    assert body["raster_path"].startswith("/")
    assert Path(body["raster_path"]).is_file()

    # The page may hand back a URL for the same asset; the server must resolve it.
    url = f"{server}{body['raster_url']}"
    config = _tiny_config()
    config["fractal"]["map"] = {"image": url, "strength": 1.2, "size": 32, "blend": 0.8}
    config["fractal"]["orbit_trap"] = {"image": url, "size": 32, "distance_mode": 1}
    status, payload = _post(server, "/api/render", {"form": "image", "config": config})
    assert status == 200, payload
    assert payload["ok"] is True
