# Change: Ship railcam-gui as desktop installers

## Why

Using railcam-gui today requires Python, pip and a separately installed FFmpeg,
which rules out most climbers and coaches. They need an installer they can
double-click, and a way to learn that a newer version exists.

## What Changes

- Build installers with Briefcase from the existing package: `.dmg` (macOS),
  `.msi` (Windows), Flatpak bundle (Linux).
- Bundle everything a render needs: Python, PySide6, CPU-only torch, FFmpeg, and
  the default pose model weights, so the first render works offline.
- Continuous delivery driven by Conventional Commits: release-please keeps a
  release PR open on `main` with the next version (from `feat`/`fix`/breaking
  changes) and the changelog; merging it tags, creates the GitHub Release and
  builds and attaches the installers.
- Enforce Conventional Commits on PR titles, which become the squashed commit
  messages on `main`.
- In packaged builds, check the latest GitHub Release at startup and offer a
  download link when it is newer. No silent self-update.
- Run the render subprocess through the packaged executable, since a packaged
  app has no `railcam` script or plain Python interpreter.
- Distribute installers under AGPL-3.0, required by the bundled Ultralytics
  models. Source stays MIT; add the license files and third-party notices.
- macOS builds are not signed nor notarized: the README documents how to open
  them past Gatekeeper.

## Impact

- Affected specs: `gui-interface` (render execution), new `desktop-distribution`
- Affected code: `pyproject.toml`, `src/railcam/gui/render.py`,
  `src/railcam/gui/app.py`, `src/railcam/output.py`, `src/railcam/pose.py`,
  new update check module, new `.github/workflows/release.yml` and `pr-title.yml`,
  `release-please-config.json`, `CHANGELOG.md`, `README.md`,
  `LICENSE`
- pip installs no longer need a system FFmpeg (`imageio-ffmpeg` dependency, `PATH`
  still preferred), and the `gui` extra installs `PySide6-Essentials`, the only
  Qt modules used (QtCore, QtGui, QtWidgets, QtNetwork).
- Workflow change: PR titles must follow Conventional Commits; the repository
  squash commit message setting moves to "PR title".
