## ADDED Requirements

### Requirement: Desktop Installers

The project SHALL publish railcam-gui installers for macOS (`.dmg`), Windows
(`.msi`) and Linux (AppImage and Flatpak bundle) for each release, and each
installer SHALL contain everything needed to render without Python, FFmpeg or
network access.

#### Scenario: Installers attached to a release

- **WHEN** a release `vX.Y.Z` is created
- **THEN** CI SHALL attach the macOS, Windows, AppImage and Flatpak installers
  of version `X.Y.Z` to it

#### Scenario: Offline first render

- **WHEN** a user installs railcam-gui on a machine without Python or FFmpeg and
  renders with the default model while offline
- **THEN** the render SHALL succeed

### Requirement: Continuous Delivery from Conventional Commits

Releases SHALL be derived from Conventional Commits on `main`: a release pull
request SHALL carry the next semantic version and the changelog, and merging it
SHALL create the tag and the GitHub Release.

#### Scenario: Release PR updated

- **WHEN** a `feat` or `fix` commit lands on `main`
- **THEN** the release PR SHALL propose a minor or patch bump respectively and
  list the commit in the changelog

#### Scenario: Breaking change

- **WHEN** a commit marked as breaking (`!` or `BREAKING CHANGE:`) lands on `main`
- **THEN** the release PR SHALL propose a major bump

#### Scenario: Release PR merged

- **WHEN** the release PR is merged
- **THEN** the tag `vX.Y.Z` and its GitHub Release SHALL be created with the
  changelog section as notes, and the installers SHALL be built and attached

#### Scenario: Non-conventional PR title

- **WHEN** a pull request title does not follow Conventional Commits
- **THEN** the PR title check SHALL fail

### Requirement: Installer License

Installers SHALL be distributed under AGPL-3.0 and SHALL include the license
texts of railcam and its bundled third-party components.

#### Scenario: License shipped

- **WHEN** a user inspects an installed application
- **THEN** the AGPL-3.0 text and third-party notices SHALL be present

### Requirement: Update Notification

A packaged railcam-gui SHALL check the latest GitHub Release at startup and,
when it is newer than the running version, SHALL offer a link to download it.
The check SHALL NOT block or break the GUI.

#### Scenario: Newer version available

- **WHEN** the latest release is newer than the running version
- **THEN** the GUI SHALL show a notice with a link to the release page

#### Scenario: Up to date

- **WHEN** the running version is the latest
- **THEN** the GUI SHALL show no notice

#### Scenario: Check fails

- **WHEN** the network or the GitHub API is unavailable
- **THEN** the GUI SHALL work normally and show no notice

#### Scenario: pip install

- **WHEN** railcam-gui runs from a pip installation
- **THEN** no update check SHALL be made
