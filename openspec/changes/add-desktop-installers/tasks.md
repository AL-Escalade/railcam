## 1. Spike
- [x] 1.1 Minimal Briefcase build on macOS: GUI opens, CLI dispatch via `RAILCAM_RUN_CLI` runs a render
- [ ] 1.2 Same on Windows and Linux (AppImage and Flatpak) runners in CI

## 2. Packaged runtime
- [x] 2.1 `app.main` dispatches to the CLI when `RAILCAM_RUN_CLI` is set
- [x] 2.2 `resolve_railcam_command` returns the packaged executable when packaged
- [x] 2.3 FFmpeg fallback to `imageio-ffmpeg` in `output.py`
- [x] 2.4 Bundled default weights resolved in `pose.py`
- [x] 2.5 Tests for command resolution and FFmpeg resolution

## 3. Update check
- [x] 3.1 Version comparison (pure, tested)
- [x] 3.2 Async GitHub check with download link, packaged builds only

## 4. Release
- [x] 4.1 Briefcase config in `pyproject.toml` (CPU-only torch, resources)
- [ ] 4.9 App icon (Briefcase default for now)
- [x] 4.2 release-please config (Python, `pyproject.toml`, `CHANGELOG.md`, bootstrap SHA, `release-as: 1.0.0`)
- [ ] 4.8 After the 1.0.0 release, remove `release-as`
- [x] 4.3 `release.yml`: release-please, then build matrix (dmg, msi, AppImage, flatpak) uploaded to the release
- [x] 4.4 `pr-title.yml`: Conventional Commits lint on PR titles
- [ ] 4.5 Switch the repo squash commit title to "PR title" (with the user's go-ahead)
- [x] 4.6 `LICENSE` (MIT), AGPL-3.0 text and third-party notices in installers
- [x] 4.7 README: contributing with Conventional Commits; install per OS, Gatekeeper and SmartScreen steps
