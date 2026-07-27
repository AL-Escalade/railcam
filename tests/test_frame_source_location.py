"""The frame source lives in the core; the GUI re-exports it.

Gap repair needs the same indexed access the GUI already had, so there must be
exactly one implementation, and the core must not depend on the GUI package.
"""

from __future__ import annotations


def test_frame_source_importable_from_core() -> None:
    from railcam.frame_source import FrameSource

    assert FrameSource is not None


def test_gui_reexports_the_same_class() -> None:
    from railcam.frame_source import FrameSource as CoreFrameSource
    from railcam.gui.frame_source import FrameSource as GuiFrameSource

    assert GuiFrameSource is CoreFrameSource


def test_core_module_does_not_import_the_gui() -> None:
    from pathlib import Path

    source = Path("src/railcam/frame_source.py").read_text(encoding="utf-8")

    assert "railcam.gui" not in source
