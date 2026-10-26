"""Notify packaged installations that a newer release is available."""

from __future__ import annotations

import json
import re
import sys
from importlib.metadata import version

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest

LATEST_RELEASE_URL = "https://api.github.com/repos/AL-Escalade/railcam/releases/latest"

_VERSION = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def parse_version(text: str) -> tuple[int, int, int] | None:
    """Parse `X.Y.Z` or `vX.Y.Z`; None for anything else, such as a pre-release."""
    match = _VERSION.match(text.strip())
    if match is None:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def is_newer(latest: str, current: str) -> bool:
    """Whether release `latest` is newer than the running `current` version."""
    latest_version = parse_version(latest)
    current_version = parse_version(current)
    if latest_version is None or current_version is None:
        return False
    return latest_version > current_version


class UpdateChecker(QObject):
    """Fetches the latest GitHub release once and reports it when newer."""

    updateAvailable = Signal(str, str)  # version, release page url

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._network = QNetworkAccessManager(self)
        self._network.finished.connect(self._on_finished)

    def check(self) -> None:
        request = QNetworkRequest(QUrl(LATEST_RELEASE_URL))
        request.setRawHeader(b"Accept", b"application/vnd.github+json")
        self._network.get(request)

    def _on_finished(self, reply: QNetworkReply) -> None:
        reply.deleteLater()
        if reply.error() != QNetworkReply.NetworkError.NoError:
            print(f"Update check failed: {reply.errorString()}", file=sys.stderr)
            return
        try:
            release = json.loads(bytes(reply.readAll().data()))
            tag, page = str(release["tag_name"]), str(release["html_url"])
        except (ValueError, KeyError, TypeError) as error:
            print(f"Update check failed: unexpected response ({error})", file=sys.stderr)
            return
        if is_newer(tag, version("railcam")):
            self.updateAvailable.emit(tag.lstrip("v"), page)
