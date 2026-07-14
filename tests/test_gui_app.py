"""Tests for the railcam-gui entry point."""

from __future__ import annotations

import sys

import pytest


def test_main_without_pyside6_prints_install_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Poison the import machinery so any PySide6 import raises ImportError,
    # and drop cached GUI modules that may already hold a reference.
    monkeypatch.setitem(sys.modules, "PySide6", None)
    for name in list(sys.modules):
        if name.startswith("railcam.gui.main_window") or name == "railcam.gui.app":
            monkeypatch.delitem(sys.modules, name)

    from railcam.gui.app import main

    exit_code = main()

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "railcam[gui]" in stderr
