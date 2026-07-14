"""Per-video player: frame display, timeline, range selection, climber choice."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from railcam.gui.frame_source import FrameSource
from railcam.gui.imaging import frame_to_qimage
from railcam.gui.project import VideoEntry
from railcam.gui.timeline import TimelineWidget

_STEP_SMALL = 1
_STEP_LARGE = 10

CLIMBER_LABELS = [("auto", "Auto"), ("left", "Gauche"), ("right", "Droite")]


class FrameDisplay(QLabel):
    """Displays the current frame scaled to fit while keeping aspect ratio."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setObjectName("frameDisplay")

    def set_frame_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._rescale()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class PlayerWidget(QFrame):
    """One video of the session: navigation, range selection and climber choice."""

    stateChanged = Signal()
    removeRequested = Signal()

    def __init__(self, path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source = FrameSource(path)
        self.current_frame = 0
        self.start_frame = 0
        self.end_frame = self.source.frame_count - 1

        self.setObjectName("playerCard")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._build_ui(path)
        self.display_frame(0)

    # --- UI construction -------------------------------------------------

    def _build_ui(self, path: Path) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel(path.name)
        title.setObjectName("playerTitle")
        meta = self.source.metadata
        info = QLabel(f"{meta.width}×{meta.height} · {meta.fps:.4g} fps · {meta.total_frames} img")
        info.setObjectName("playerMeta")
        remove = QToolButton()
        remove.setText("✕")
        remove.setToolTip("Retirer cette vidéo")
        remove.clicked.connect(self.removeRequested.emit)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(info)
        header.addWidget(remove)
        layout.addLayout(header)

        self.display = FrameDisplay()
        layout.addWidget(self.display, stretch=1)

        self.timeline = TimelineWidget(self.source.frame_count)
        self.timeline.frameScrubbed.connect(self.display_frame)
        layout.addWidget(self.timeline)

        nav = QHBoxLayout()
        self.counter = QLabel()
        self.counter.setObjectName("frameCounter")
        for text, tip, delta in (
            ("⏮", "Aller au début", None),
            ("◀◀", "-10 images (Maj+←)", -_STEP_LARGE),
            ("◀", "-1 image (←)", -_STEP_SMALL),
            ("▶", "+1 image (→)", _STEP_SMALL),
            ("▶▶", "+10 images (Maj+→)", _STEP_LARGE),
        ):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            if delta is None:
                button.clicked.connect(lambda: self.display_frame(0))
            else:
                button.clicked.connect(lambda _=False, d=delta: self.step(d))
            nav.addWidget(button)
        nav.addStretch()
        nav.addWidget(self.counter)
        layout.addLayout(nav)

        range_row = QHBoxLayout()
        self.start_button = QPushButton("Début = ici")
        self.start_button.setToolTip("Utiliser la frame affichée comme frame de départ (sync)")
        self.start_button.clicked.connect(self._set_start_here)
        self.end_button = QPushButton("Fin = ici")
        self.end_button.setToolTip("Utiliser la frame affichée comme frame de fin")
        self.end_button.clicked.connect(self._set_end_here)
        self.range_label = QLabel()
        self.range_label.setObjectName("rangeLabel")
        range_row.addWidget(self.start_button)
        range_row.addWidget(self.end_button)
        range_row.addWidget(self.range_label)
        range_row.addStretch()
        range_row.addWidget(QLabel("Grimpeur :"))
        self.climber_combo = QComboBox()
        for value, label in CLIMBER_LABELS:
            self.climber_combo.addItem(label, userData=value)
        self.climber_combo.currentIndexChanged.connect(lambda _: self.stateChanged.emit())
        range_row.addWidget(self.climber_combo)
        layout.addLayout(range_row)

        self._refresh_labels()

    # --- State -----------------------------------------------------------

    @property
    def climber(self) -> str:
        return str(self.climber_combo.currentData())

    def to_video_entry(self) -> VideoEntry:
        return VideoEntry(
            path=self.source.path,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            climber=self.climber,
        )

    def apply_entry(self, entry: VideoEntry) -> None:
        """Restore range and climber from a project entry."""
        self.start_frame = entry.start_frame
        self.end_frame = entry.end_frame
        index = self.climber_combo.findData(entry.climber)
        if index >= 0:
            self.climber_combo.setCurrentIndex(index)
        self.timeline.set_range(self.start_frame, self.end_frame)
        self.display_frame(self.start_frame)
        self._refresh_labels()
        self.stateChanged.emit()

    def is_range_valid(self) -> bool:
        return self.end_frame > self.start_frame

    # --- Navigation ------------------------------------------------------

    def display_frame(self, index: int) -> None:
        index = max(0, min(self.source.frame_count - 1, index))
        frame = self.source.get_frame(index)
        self.current_frame = index
        self.display.set_frame_pixmap(QPixmap.fromImage(frame_to_qimage(frame)))
        self.timeline.set_current_frame(index)
        self._refresh_labels()

    def step(self, delta: int) -> None:
        self.display_frame(self.current_frame + delta)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        step = _STEP_LARGE if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else _STEP_SMALL
        if event.key() == Qt.Key.Key_Left:
            self.step(-step)
        elif event.key() == Qt.Key.Key_Right:
            self.step(step)
        else:
            super().keyPressEvent(event)

    # --- Range selection ---------------------------------------------------

    def _set_start_here(self) -> None:
        self.start_frame = self.current_frame
        if self.end_frame <= self.start_frame:
            self.end_frame = self.source.frame_count - 1
        self._range_updated()

    def _set_end_here(self) -> None:
        self.end_frame = self.current_frame
        self._range_updated()

    def _range_updated(self) -> None:
        self.timeline.set_range(self.start_frame, self.end_frame)
        self._refresh_labels()
        self.stateChanged.emit()

    def _refresh_labels(self) -> None:
        self.counter.setText(f"{self.current_frame} / {self.source.frame_count - 1}")
        valid = self.is_range_valid()
        marker = "" if valid else "  ⚠ fin ≤ début"
        self.range_label.setText(f"[{self.start_frame} → {self.end_frame}]{marker}")
        self.range_label.setProperty("invalid", not valid)
        style = self.range_label.style()
        style.unpolish(self.range_label)
        style.polish(self.range_label)

    def close_source(self) -> None:
        self.source.close()
