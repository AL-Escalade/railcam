"""Frame-exact video access, shared with the core pipeline.

The implementation moved to `railcam.frame_source` when gap repair started
needing the same indexed access. Re-exported here so GUI code keeps importing
it from the package it belongs to.
"""

from __future__ import annotations

from railcam.frame_source import DEFAULT_CACHE_SIZE, SEQUENTIAL_WINDOW, FrameSource

__all__ = ["DEFAULT_CACHE_SIZE", "SEQUENTIAL_WINDOW", "FrameSource"]
