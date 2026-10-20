# Fractal-generation

> **Status: closed.** This project is archived and no longer maintained. The code
> is left as it was on the final commit: it builds, its test suite passes, and the
> documented commands work, but expect no further fixes or features. Issues and
> pull requests are not monitored.
>
> Before closing, a final AI-assisted pass (an agent plus delegated workers) was
> run over the whole repository to clear the outstanding reports: the broken entry
> points, the untyped parameter dict, the shading rings, the deep-zoom collapse,
> the flicker/grain exposure drift, the orbit-trap orientation and the missing UI
> controls were all fixed and are covered by tests. Known limitations that were
> deliberately left open are listed under *Known limitations* below.

Procedural root-finding fractals (Newton / Halley, *RFA* and *nova* variants) with
relief shading, palettes, animations and an optional NFT collection front-end.

```text
src/fractalgen/     fractal maths, rendering, animation   (no NFT code)
src/nftgen/         collection generator built on fractalgen
examples/configs/   ready-to-run JSON configs
tests/              pytest suite
```

## Install

```bash
uv venv && uv pip install -e ".[dev]"     # or: python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Optional extras: `[palette]` (image palettes / seaborn / perceptual palettes),
`[edges]` (Canny edge detector), `[video]` (MP4 output).

## Use the fractal library directly

The numeric core has no rendering or NFT dependency:

```python
import numpy as np
from fractalgen import FractalConfig, Polynomial, solve, build_colormap, render_image

poly = Polynomial.from_roots([1, -1, 1j, -1j])
image = solve(FractalConfig(size=1024), poly, workers=8)

image.iterations   # (N, N) iteration count per pixel
image.roots        # (N, N) root each pixel converged to
image.distance     # (N, N) orbit-trap distance, or None
image.boundary     # (N, N) bool edge mask
```

Rendering is a separate, equally reusable step:

```python
from fractalgen import render

rgb = render.colorize(image.scalar_field("iteration"), build_colormap())
render.save(rgb, "fractal.png")
```

Or go through the pipeline:

```python
from fractalgen import Config, render_image, render_video

render_image(Config.from_dict({"fractal": {"size": 1200}, "output": "out.png"}))
render_video(Config.from_dict({"form": "video", "output": "out.gif",
                               "video": {"animations": ["zoom"]}}))
```

## Command line

```bash
fractalgen render examples/configs/image.json
fractalgen render examples/configs/video.json
fractalgen image --set fractal.size=1200 --set colour.name=magma -o out.png
fractalgen schema            # machine-readable description of every field
fractalgen example --form video
nftgen generate examples/configs/collection.json
```

`--set section.key=value` overrides any field (values are parsed as JSON when
possible), so the CLI never needs a second parameter syntax.

## Web UI

```bash
fractalgen ui                 # http://127.0.0.1:8123 (opens a browser)
fractalgen ui --port 9000 --no-open
python -m fractalgen.ui       # same thing
```

A single page, served locally, with every configuration field (grouped in
collapsible sections), a **Generate** button, a preview pane (PNG or GIF), palette
swatches, a "New polynomial" seed randomiser, an image-import control and a
"Download config" button. The form is generated from `GET /api/schema`, which is
derived from the config dataclasses, so new fields appear automatically.

Endpoints: `GET /api/schema`, `POST /api/render`, `POST /api/from-image`,
`GET /outputs/<file>`, `GET /api/health`. Standard library only (`http.server`),
no build step, no external assets.

Safety notes: binds to `127.0.0.1` by default, renders are serialised behind one
lock and run in a worker thread with a timeout, request bodies are capped at
8 MB, output names are whitelisted (no separators, resolved inside the output
directory), and the only thing the request body can do is populate the
configuration model. UI renders default to `runtime.workers = 1` - the solver's
process pool spawns children, which is not what you want inside a request thread.

## Configuration

One JSON document per render. Unknown keys are rejected with a suggestion, every
field has a default, and `fractalgen schema` prints the whole model. Input paths
(background, orbit trap, palette image) resolve relative to the config file;
`output` resolves relative to the working directory.

```jsonc
{
  "form": "image",                     // or "video"
  "output": "out/fractal.png",
  "fractal": {
    "size": 1200,                      // square resolution in pixels
    "domain": {"x": [-1, 1], "y": [-1, 1]},
    "method": "newton",                // newton | halley
    "variant": "standard",             // standard | nova
    "iterations": 40,                  // base iteration budget
    "auto_iterations": true,           // grow the budget when pixels saturate
    "max_iterations": null,            // ceiling for the automatic budget
    "tolerance": 1e-5,                 // convergence threshold on |delta|
    "damping": [1.0, 0.0],             // complex damping as [real, imag]
    "round_digits": 4,                 // root quantisation before colouring (0 = off)
    "polynomial": {"mode": "random", "max_degree": 6, "seed": 7},
    "orbit_trap": {"image": null, "distance_mode": 1}
  },
  "colour": {"mode": "matplotlib", "name": "viridis"},
  "image": {
    "return_type": "smooth",           // smooth | iteration | iteration-steps | root | roots | distance | boundary
    "smooth": 2.0,                     // colour-field blur (px), kills contour steps
    "shading": {"type": "matplotlib", "height": "auto", "smooth": 1.0, "strength": 0.8,
                "azimuth": 315, "elevation": 20, "vertical_exaggeration": 2.0}
  },
  "runtime": {"workers": 0, "verbose": false}   // workers 0 = all cores
}
```

* **`polynomial.mode`** — `random` (resampled until the frame is neither empty nor
  saturated: `min_area`/`max_area`/`attempts`), `roots`, `coefs`, or `taylor`
  (`function`, `degree`, `interval`).
* **`variant: "nova"`** — each pixel gets its own perturbed polynomial and the
  iteration starts at its critical point.
* **`orbit_trap`** — a bitmap whose distance field is injected during iteration;
  enables the `distance` render type and true orbit normals.
* **`return_type`** chooses the *colour* field: `smooth` (basin hue plus a small
  inner gradient), `iteration` (continuous count), `iteration-steps` (raw
  counter), `root` (flat basin identity), `roots` (log distance to the
  attractor), `distance` (orbit trap), `boundary`, `image` (the source picture's
  colours with the fractal as its texture - see *Image import*).
* **`shading.type`** — `matplotlib`, `fossil`, `blinn-phong` or `none`. Lighting
  is a separate layer: the colour comes from `return_type`, the relief only
  multiplies it (Ultra Fractal's "slope + lighting" split).
* **`shading.height`** — `auto`, `roots` (log distance to the attractor),
  `iteration` (`log1p` of the continuous count), `iteration-steps`, `distance`.

### Why shading used to show rings

Two independent causes, both fixed:

1. **Quantised height.** Integer iteration counts are a staircase; shading them
   embosses every level boundary. Convergent iterations also need a
   *convergence-order* smooth count, not Mandelbrot's escape-time formula:
   `nu = n + log_order(ln(u) / ln(tolerance))` with `order = 2` for Newton and
   `3` for Halley (`u` is the residual at the converging step). Escape-time
   smoothing assumes `|z| -> infinity` and does not transfer.
2. **Colourmapping a monotone field is an iso-contour plot.** Any colormap of
   `iteration` rings by construction, however smooth the count is. The default
   `smooth` colouring therefore reserves one colormap step per basin hue and
   only tints 35 % of a step inside it, so hues can never wrap. `image.smooth`
   additionally blurs the colour field by 2 px to soften the last kinks.

Heights stay in float; lighting is multiplied into the albedo
(`1 - strength + strength * intensity`) rather than added to a wrapping
gradient, per the Ultra Fractal / mathr guidance.

### Deep zoom

Zooming in is limited by two things, both handled explicitly:

1. **Iteration budget.** With a fixed budget, a deep window eventually has every
   pixel hitting the cap, so the whole frame gets the "did not converge" value and
   renders as one flat colour. That is what used to happen around 10^6 zoom. With
   `auto_iterations` (default on) the solver probes the frame and doubles the
   budget (up to `max_iterations`, default `max(8 * iterations, 1024)`, hard cap
   8192) until at least 95 % of pixels converge. A 10^10 zoom needs roughly 600
   iterations where the default 40 gives a blank frame.
2. **float64 resolution.** Coordinates are stored as doubles: past ~10^15 total
   zoom the window is thinner than one ulp and pixels repeat. That limit is
   detected before rendering (`Domain.can_resolve`) and reported as
   `fractal.domain: window ... is below float64 resolution ... maximum zoom ...`,
   instead of silently producing a blocky image. Zoom animations stop zooming at
   that point and hold the last resolvable frame.

Frame brightness during a zoom is normalised against interpolated per-keyframe
ranges, so exposure drifts smoothly instead of pumping (per-frame min/max
stretching) or washing out (one fixed range for the whole animation).

### Colour

| `colour.mode` | fields | notes |
| --- | --- | --- |
| `matplotlib` | `name` | any built-in colormap |
| `seaborn` | `name` | needs `[palette]` |
| `colors` | `colors: ["#112233", ...]` | interpolated, optional `add_black` |
| `accents` | `image`, `accent` | palette extracted from an image, then accented |

`accent` is one of `complementary`, `triadic`, `analogous`,
`split-complementary`, `tetradic`, `shades`.

### Video and animations

```jsonc
"video": {
  "fps": 20, "duration": 3, "codec": "gif",   // or "frames": 60, "codec": "mp4"
  "animations": ["alpha", {"name": "pulse"}, {"name": "flicker"}]
}
```

Effects run in the order given. `zoom`, `translate` and `shading` re-solve the
fractal; `alpha`, `pulse`, `flicker`, `grain` and `explosion` transform frames.
Unknown effect parameters are rejected.

| effect | parameters (defaults) |
| --- | --- |
| `alpha` | `fade_in` 0 (auto), `fade_out` 0 (none), `min_alpha` 60, `max_alpha` 230 |
| `pulse` | `beta` auto, `decal` 0, `oscillation_frequency` π, `oscillation_amplitude` 1.0, `speed` auto |
| `flicker` | `amplitude` 1.0, `percentage` 0.02, `clusters` 12, `speed` 2.5, `blob_size` 0.0022, `sharpen` 4.0, `edge_width` 2, `on_fractal` true, `luminance_weight` 0.5, `mode` additive, `seed` |
| `grain` | `amount` 0.18, `grain_size` 0.003, `octaves` 2, `keyframes` 8, `speed` 0.9, `vignette` 0.0, `seed` |
| `explosion` | `speed` 45, `hold_ratio` 0.2, `seed` |
| `zoom` | `zoom_factor` 1.15, `oversample` 2 (see *Deep zoom*) |
| `translate` | `init_damping` [0.4, -0.5], `end_damping` [1.35, 0.85] |
| `shading` | `rotations` 1.0 |

`flicker` draws *star points on the fractal*: the sparkle mask is clipped to the
fractal body (dilated by `edge_width`), the blobs are 1-2 px wide, each cluster
pulses with its own period and `sharpen` controls how brief the flash is. Light is
added scaled by the remaining head-room and never re-normalised, so the exposure
stays flat. `grain` is multi-octave smooth noise that boils through correlated key
fields, and its delta is scaled by `min(pixel, 255-pixel)`, so it is zero-mean and
cannot clip (a multiplicative grain pulls bright frames down, which is what made
animations look like they were darkening). `alpha` no longer fades out unless you
ask, and a fully revealed frame is fully opaque.

## Image import (palette + map modifier)

```bash
mkdir -p photos && cp shoot/*.jpg photos/
fractalgen from-image                                   # every image in ./photos -> out/<name>.png
fractalgen from-image photos                            # same, explicit folder
fractalgen from-image photos/portrait.jpg -o out/p.png   # one image
fractalgen from-image --raster-only                     # assets only (palette + raster)
```

`photos/` is the drop folder: it is git-ignored (only `photos/.gitkeep` is tracked),
and running `from-image` with no argument processes every `.png/.jpg/.jpeg/.webp/
.bmp/.tif/.tiff/.gif` inside it. Outputs default to `out/<name>.png` with
`<name>_palette.json` and `<name>_raster.png` next to them.

Three things come out of one picture:

| asset | what it is | how it is used |
| --- | --- | --- |
| palette (`*_palette.json`) | dominant colours, dark to light | becomes the colormap (`colour.mode: "colors"`) |
| raster (`*_raster.png`) | luminance, `--raster-size` square | sampled per pixel as the **map modifier** |
| trap bitmap | the raster thresholded (Otsu) | orbit trap, so the silhouette also appears in the relief |

**Orbit-trap colouring.** A distance-only trap does *not* reproduce a picture: it
records how close an orbit came to the shape, and a converging Newton orbit visits
only a small part of the plane, so a trapped tree renders as a few bands. Digital
orbit trapping instead samples the **trap image's colour at every orbit point**
(iq's *bitmap orbit traps*, Carlson's artistic orbit traps). Set
`image.return_type: "trap"` and the pixel takes that colour:

```jsonc
"fractal": {"orbit_trap": {"image": "tree.png", "size": 128, "colour_mode": "closest"}},
"image":   {"return_type": "trap"}
```

`orbit_trap.colour_mode` picks the sample: `closest` (default - colour at the
orbit's closest approach), `average` (mean along the orbit, pale) or `last`
(colour at the final orbit point).

Bitmap traps leave **streaks along the orbit paths** (a Newton orbit travels in a
near-straight line to its root, and whatever colour it meets smears along it - iq's
own bitmap traps show the same). `image.trap_mix` (0..1, default `0.7`) mixes the
fractal's own colouring back in to fade them; `0.3-0.5` keeps the picture readable
without the rays dominating. If you want the photo with no streaks at all, use
`return_type: image` instead - that path never samples the orbit.

`orbit_trap.scale` decides how big the picture is: `1` makes it span the frame
(one huge copy per basin), `2-3` tiles it smaller, which is what makes trap
colouring usable as a texture. `orbit_trap.wrap` tiles instead of smearing at the
edges (`--trap-scale` sets both from the CLI, default `2`). The fractal's own
structure is kept inside the trap colours - without it, the picture's flat regions
turn the frame into wallpaper.

If the picture must sit in the frame *as drawn*, use `fractal.map.blend` or
`image.return_type: "image"` (picture colours, fractal as texture) instead.

**Subject masks.** A bitmap trap needs a silhouette, and a photograph is not one:
thresholding `atrest.jpg` at its median marks half the frame, filling its edge map
marks 86 % of it. `fractalgen.subject` builds a matte instead:

| `image.subject.method` | what it gives |
| --- | --- |
| `auto` (default) | silhouette if the edges close into a region, else the busy-centred region |
| `edges` | the Canny map itself (thin outlines - good structure for a trap) |
| `silhouette` | edges -> close -> fill -> largest blob |
| `focus` | centre-weighted edge density ("the busy area in the middle") |
| `grabcut` | OpenCV GrabCut from a centred rectangle |
| `rect` | an explicit rectangle: `image.subject.rect: [x0, y0, x1, y1]`, fractions |
| `threshold` | Otsu on the luminance (flat graphics) |
| `none` | the whole frame |

`segment` (part of `auto`) is the workhorse for photographs: k-means over the
colours, then keep the cluster that sits near the middle and covers a plausible
share of the frame. On a painterly forest scene it isolates the subject's ochre
mass (28 % of the frame) where every threshold and edge method failed.

With `return_type: "image"` and `image.subject_only: true` (default) the matte is
used as an alpha: the subject is rendered as fractal texture in its own colours
and the rest of the frame stays the untouched picture - the picture's shape, made
of fractal.

The matte is written next to the render as `<name>_mask.png`, so a bad matte is
visible and can be hand-edited; `orbit_trap.mask` then keeps that silhouette while
`orbit_trap.image` still supplies the colours (`--subject`, `--subject-rect` from
the CLI). For a compositional picture the honest answer is `rect` or a hand-drawn
mask - no low-level detector knows which part of a forest is "the subject".

Orbit traps sample the *render domain* (not a hard-coded `[-1, 1]²`, which used to
clamp every zoomed orbit to the border), index `[row=imag, col=real]` (they used
to be transposed), interpolate bilinearly, and are seeded with the pixel's own
position before the iteration starts - otherwise a converging orbit barely
samples the image at all.

Two couplings, because one is not enough:

* **`fractal.map.mode` / `strength`** - the dynamical map. `damping` (default)
  scales the Newton damping per pixel, so the *basin pattern* follows the image;
  `warp` displaces the starting point along the image gradient. Linear, gentle,
  stays subtle: a Newton orbit collapses onto a root within a few steps, so a
  purely dynamical coupling only reshapes the pattern - it cannot paint the photo.
* **`fractal.map.blend`** (0..1, default `0.8`) - the image-space blend. The
  finished frame's brightness is multiplied by the raster, so the photo reads
  directly while every fractal filament survives (`blend: 0` disables it).

The raster is read with bilinear interpolation and pre-blurred: a
nearest-neighbour lookup stamps the raster grid into the fractal as square blocks.

Tuning:

* `blend 0.8 + strength 1.4` (defaults) - the picture is legible and the fractal
  texture is visible inside it;
* `blend 1.0` - the photo dominates; use `strength 0` for a pure "photo made of
  fractal filaments" overlay;
* `blend 0.3-0.5` - subtle watermark; `blend 0` - dynamical map only.

At config level it is just two paths plus the palette:

```jsonc
"fractal": {
  "map": {"image": "out/portrait_raster.png", "strength": 1.4, "size": 96},
  "orbit_trap": {"image": "out/portrait_raster.png", "size": 96, "distance_mode": 1}
},
"colour": {"mode": "colors", "colors": ["#0b1024", "#33507a", "#e8d8a0"]}
```

See `examples/configs/from_image.json`. Library use:

```python
from fractalgen import Config
from fractalgen.image_import import analyse_image, config_from_image, write_assets

analysis = analyse_image("photo.jpg", colours=6)
analysis.colours               # ['#0b1024', '#33507a', ...]
write_assets(analysis, "out", stem="photo")     # raster + palette json
config = config_from_image(Config(), "photo.jpg", strength=1.4)
```

## Post-process filters

`image.postprocess` runs after the frame is rendered, in order. It is what keeps
the banded look usable:

| filter | parameters (defaults) | purpose |
| --- | --- | --- |
| `deband` | `radius` 3, `sigma_color` 14, `sigma_space` 2.5, `passes` 1 | bilateral-style pass that removes the small colour steps of iteration rings while keeping filaments (large steps) |
| `unsharp` | `radius` 2.0, `amount` 0.35, `threshold` 2 | restores local contrast lost to the de-banding blur |

Defaults are `[deband, unsharp]`. Set `"postprocess": []` for the raw banded look,
or use `"return_type": "smooth"` for a colouring that has no rings to remove.

### UI notes for the image map and trap

The import button fills in the whole chain: palette colours, `fractal.map.image`
(the luminance raster, blended at `map.blend`), and `fractal.orbit_trap.image`
(the **original upload**, since trap colouring needs the picture's colours) with
`scale 2`, `wrap on`, `colour_mode closest`, `size 128` - the same defaults as
`from-image --return-type trap`. `image.trap_mix` (slider) is the knob that fades
the orbit streaks.

## NFT collections

`nftgen` is a thin layer on top of `fractalgen`; `fractalgen` never imports it.

```jsonc
{
  "collection": {"name": "Roots", "description": "...", "size": 10, "start_index": 1, "seed": 2024},
  "output": {"directory": "out/collection", "metadata": true},
  "composition": {"background": null, "scale": [0.45, 0.8], "rotation": [-10, 10], "flip": true},
  "fractal": {"size": 700, "polynomial": {"mode": "random", "max_degree": 6}},
  "colour": {"mode": "matplotlib", "name": "inferno"}
}
```

Each token gets `<n>.png` plus `<n>.json` metadata (traits: method, variant,
iterations, polynomial mode/degree, palette, seed, composition), and the run
writes `collection.json`. Seeds are `collection.seed + index`, so a collection is
reproducible.

## What changed from the original

* **It runs.** The old tree imported `Fractal_calculator.Fractal_calculator.*`
  (doubled package path), `daskowner` (a module that never existed), and read
  parameter keys (`raster_image_dir`, `distance_option`, `color_list`, `test`)
  that no code path ever wrote — so the shipped entry point could not start.
* **Modular.** One package per concern: `config`, `polynomial`, `orbit`,
  `solver`, `edges`, `shading`, `colour`, `render`, `frames`, `animations`,
  `pipeline`, `cli`. The numeric core (`solver` + `polynomial`) imports only
  numpy/scipy, and the NFT generator is a separate package.
* **Parameters.** Validated frozen dataclasses instead of an untyped dict; one
  JSON schema, no `param["Image"]["return type"]`-style string keys, no `**kwargs`
  grabbing unknown options silently.
* **Faster** (see below): Horner evaluation, active-set iteration, no memmap
  round-trip, no matplotlib figure per intermediate image.
* **Fixes.** Halley produced a constant (blank) shaded image because it built its
  distance map from an empty mask; `blinn-phong` was blank without an orbit trap;
  a zero derivative produced `inf`/`nan` pixels that poisoned whole frames; the
  "zoom" animation could reference an unbound variable and mutated the caller's
  config; frames on disk were read with a leaking memory map; deep zooms went
  flat because the iteration budget was fixed.
* **Effects that read as intended.** `flicker` used to re-normalise every frame
  (the whole image flashed) and lit isolated pixels (static); `grain` re-rolled
  binary dots every frame (TV snow) and also pumped the exposure. Both now keep
  the mean luminance fixed and animate smoothly - see the parameter table.

## Performance

Measured on an M-series laptop, 4 roots, 40 iterations, tolerance 1e-6, serial:

| size | original | this version |
| --- | --- | --- |
| 400² | ~7.0 s | 0.06 s |
| 800² | — | 0.22 s |
| 1600² | — | 0.95 s |
| 2400² | — | 2.1 s |

The original spent most of its time on per-tile memmap writes/reads (three files
per tile plus reassembly) and on a matplotlib figure per debug image. Worker
processes only pay off above roughly 2000² (pool start-up dominates below that);
`workers: 0` still uses every core, which is why `solve` refuses to spawn a pool
when `__main__` is not importable (interactive/stdin) instead of dying like the
original did.

Numerical parity was checked against the original on a fixed polynomial: 100 % of
pixels land within one iteration of the old result and converge to the same roots.

## Known limitations

Left open on purpose at close, in the order they are likely to bite:

* **Bitmap orbit traps smear along the orbit.** A Newton orbit runs in a
  near-straight line to its root, so a trapped picture arrives as streaks
  (`image.trap_mix` fades them). No setting reproduces a photograph as drawn.
* **Subject detection on painterly images is heuristic.** `image.subject: auto`
  clusters colours and keeps the centre-most, subject-sized one; on a scene with
  no colour separation it returns a plausible blob rather than the subject. Use
  `rect` or hand-edit `<name>_mask.png`.
* **MP4 needs `imageio-ffmpeg`**, which is not installed by default; without it the
  renderer falls back to GIF (the UI reports the codec it used).
* **The web UI is local-only** (127.0.0.1, no auth, single render at a time) and is
  meant for iterating on parameters, not for serving.
* **Animations are CPU-bound and single-machine.** `workers` only parallelises
  within one solve; a long zoom re-solves per keyframe.
* **`iterations` is a budget, not a constant**: `auto_iterations` grows it for deep
  zooms, so two very different windows can report different budgets.

## Tests

```bash
.venv/bin/python -m pytest -q      # or: uv run pytest
```
