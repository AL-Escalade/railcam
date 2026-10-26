## Context

The GUI already shells out to the CLI (`render_panel.py` + `render.py`), finds
FFmpeg on `PATH` (`output.py`) and lets Ultralytics fetch weights on first use
(`pose.py`). All three assumptions break inside a packaged app.

## Goals / Non-Goals

- Goals: installers for macOS, Windows, Linux (Flatpak) built and
  released automatically from Conventional Commits; offline first render;
  update notification.
- Non-Goals: Flathub publication (follow-up), code signing, silent auto-update, app stores, smaller model runtime
  (ONNX is a follow-up, to measure separately).

## Decisions

- **Briefcase** over PyInstaller: one config in `pyproject.toml`, native
  installer per OS, PySide6 supported. PyInstaller would need a separate
  installer tool per OS.
- **CLI subprocess kept**: the render keeps running out of process (isolation,
  cancel by kill). When packaged, `resolve_railcam_command` returns the app's
  own executable with `RAILCAM_RUN_CLI=1`; `app.main` dispatches to
  `railcam.cli.main` when that variable is set. An env variable avoids
  reserving an argument the CLI could later need.
- **FFmpeg via `imageio-ffmpeg`** as a core dependency: it ships a
  static binary per platform wheel. `output.py` uses `PATH` first, then
  `imageio_ffmpeg.get_ffmpeg_exe()`. Its GPL build is compatible with an AGPL
  distribution.
- **Default weights bundled** (`yolov8s-pose.pt`, ~23 MB) as app resources.
  Other sizes still download on demand into the Ultralytics weights directory,
  which is user-writable.
- **`PySide6-Essentials`** instead of `PySide6`: the Addons (WebEngine, 3D…)
  weighed 400 MB of a 1.5 GB macOS app and are unused.
- **Packaged detection**: the app's `railcam` dist-info has `INSTALLER` set to
  `briefcase`; `sys.executable` is then the app binary, which always runs
  `railcam/__main__.py`.
- **Separate Intel `.dmg`** rather than a universal build: PyTorch publishes no
  Intel macOS wheel after 2.2.2, so a universal app would pin Apple Silicon to
  torch 2.2 too. Intel builds get torch 2.2.2 and NumPy 1 (dependency marker);
  detection results are identical, only slower. Drop the Intel build once
  Ultralytics stops supporting torch 2.2.
- **CPU-only torch** in installers: CUDA wheels would add ~2 GB on Linux and
  Windows.
- **Update check**: `QNetworkAccessManager` GET on
  `api.github.com/repos/AL-Escalade/railcam/releases/latest`, compared with
  `importlib.metadata.version("railcam")`. Async, once per launch, packaged
  builds only (pip users update with pip). Failures are logged to stderr and
  never block the GUI.
- **release-please** over python-semantic-release: releases stay a reviewed
  step (merge the release PR) instead of one per push, and it bumps
  `[project].version` and `CHANGELOG.md` in that PR. The single version source
  stays `[project].version`.
- **One workflow** for release and build: tags created with `GITHUB_TOKEN` do
  not trigger other workflows, so the build matrix runs in `release.yml` when
  release-please reports `release_created`, and uploads to that release.
- **PR title lint** (`amannn/action-semantic-pull-request`): the repo only
  allows squash merges, so the PR title is the commit release-please reads.
  The squash title setting is switched from "commit or PR title" to "PR title"
  so a single-commit PR cannot bypass the lint.
- **No AppImage**: Briefcase supports it best-effort only, cannot use binary
  wheels (torch) and has known PySide issues; its CI build failed.
- **Flatpak as a bundle** (`.flatpak` attached to the release, via Briefcase):
  it gets the same update notice as other formats. Flathub would give native
  updates but needs a source-built manifest and a review; out of scope here.
- **First release is `1.0.0`**: forced with a `Release-As: 1.0.0` footer on the
  squashed commit, which applies once; a `release-as` config key would pin every
  later release too. Commits
  before the bootstrap SHA are ignored.

## Risks / Trade-offs

- Installer size (macOS `.dmg` measured at 326 MB) → accepted for now; ONNX
  follow-up if it hurts.
- Unsigned macOS app blocked by Gatekeeper → documented manual "Open anyway".
- Unsigned Windows installer triggers SmartScreen → documented.
- How Briefcase exposes `sys.executable` and detects a packaged run must be
  checked on each OS → first task is a spike that validates the dispatch.

## Open Questions

- None.
