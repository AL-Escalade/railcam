"""Global transport bar and synchronized playback controller."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QElapsedTimer, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QToolButton, QWidget

from railcam.gui.playback import PlaybackClock, frame_at, session_duration
from railcam.gui.player_widget import PlayerWidget

_TICK_MS = 33  # ~30 UI updates per second
SPEED_CHOICES = [0.1, 0.25, 0.5, 0.75, 1.0]
DEFAULT_SPEED = 0.25


class SyncPlayback(QObject):
    """Drives all players from a common clock at the preview speed."""

    stateChanged = Signal(bool)  # True while playing

    def __init__(self, get_players: Callable[[], list[PlayerWidget]]) -> None:
        super().__init__()
        self._get_players = get_players
        self._clock = PlaybackClock(speed=DEFAULT_SPEED)
        self._elapsed = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._on_tick)
        self._shown: dict[PlayerWidget, int] = {}

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    def set_speed(self, speed: float) -> None:
        self._clock.speed = speed

    def toggle(self) -> None:
        if self.playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        if not self._get_players():
            return
        self._elapsed.restart()
        self._timer.start()
        self.stateChanged.emit(True)

    def pause(self) -> None:
        self._timer.stop()
        self.stateChanged.emit(False)

    def forget(self, player: PlayerWidget) -> None:
        """Drop internal references to a removed player."""
        self._shown.pop(player, None)

    def stop(self) -> None:
        """Pause and rewind every video to its start frame."""
        self._timer.stop()
        self._clock.reset()
        self._shown.clear()
        for player in self._get_players():
            player.display_frame(player.start_frame)
        self.stateChanged.emit(False)

    def _on_tick(self) -> None:
        self._clock.advance(self._elapsed.restart() / 1000.0)
        players = self._get_players()
        ranges = []
        for player in players:
            index = frame_at(self._clock.t, player.start_frame, player.end_frame, player.source.fps)
            if self._shown.get(player) != index:
                player.display_frame(index)
                self._shown[player] = index
            ranges.append((player.start_frame, player.end_frame, player.source.fps))

        # Auto-pause once every video is frozen on its end frame
        if self._clock.t > session_duration(ranges):
            self.pause()


class TransportBar(QWidget):
    """Play/pause, stop and preview-speed controls for synchronized playback."""

    def __init__(self, playback: SyncPlayback, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._playback = playback
        self.setObjectName("transportBar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        self._play_button = QToolButton()
        self._play_button.setText("▶")
        self._play_button.setToolTip("Lecture/pause synchronisée (Espace)")
        self._play_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._play_button.clicked.connect(playback.toggle)
        layout.addWidget(self._play_button)

        stop_button = QToolButton()
        stop_button.setText("⏹")
        stop_button.setToolTip("Stop : retour aux frames de départ")
        stop_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        stop_button.clicked.connect(playback.stop)
        layout.addWidget(stop_button)

        layout.addSpacing(12)
        layout.addWidget(QLabel("Vitesse :"))
        self._speed_combo = QComboBox()
        self._speed_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for speed in SPEED_CHOICES:
            self._speed_combo.addItem(f"{speed:g}×", userData=speed)
        self._speed_combo.setCurrentIndex(SPEED_CHOICES.index(DEFAULT_SPEED))
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        layout.addWidget(self._speed_combo)

        hint = QLabel("Lecture synchronisée : chaque vidéo démarre à sa frame de départ")
        hint.setObjectName("transportHint")
        layout.addSpacing(12)
        layout.addWidget(hint)
        layout.addStretch()

        playback.stateChanged.connect(self._on_playback_state)

    def _on_speed_changed(self) -> None:
        self._playback.set_speed(float(self._speed_combo.currentData()))

    def _on_playback_state(self, playing: bool) -> None:
        self._play_button.setText("⏸" if playing else "▶")
