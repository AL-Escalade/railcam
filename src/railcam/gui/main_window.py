"""Main window: toolbar, side-by-side players, project open/save."""

from __future__ import annotations

import sys
from importlib import resources
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from railcam.gui.player_widget import PlayerWidget
from railcam.gui.project import Project, ProjectError, RenderOptions
from railcam.video import VideoError

PROJECT_FILTER = "Projet railcam (*.railcam.json)"
VIDEO_FILTER = "Vidéos (*.mp4 *.avi *.mov *.mkv *.webm *.m4v)"


class MainWindow(QMainWindow):
    """railcam-gui main window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("railcam")
        self.resize(1200, 800)
        self._project_path: Path | None = None
        self._render_options = RenderOptions()
        self._output_path: Path | None = None

        self._build_toolbar()

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 8, 12, 12)

        self._players_row = QHBoxLayout()
        self._players_row.setSpacing(10)
        root.addLayout(self._players_row, stretch=1)

        self._empty_label = QLabel("Ajoutez une vidéo pour commencer  (➕ Ajouter une vidéo)")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setObjectName("emptyState")
        self._players_row.addWidget(self._empty_label)

        self.setCentralWidget(central)

    # --- Toolbar ---------------------------------------------------------

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Principal")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = toolbar.addAction("📁 Ouvrir un projet")
        open_action.triggered.connect(self._open_project_dialog)
        save_action = toolbar.addAction("💾 Enregistrer le projet")
        save_action.triggered.connect(self._save_project_dialog)
        toolbar.addSeparator()
        add_action = toolbar.addAction("➕ Ajouter une vidéo")
        add_action.triggered.connect(self._add_video_dialog)

    # --- Players ---------------------------------------------------------

    def players(self) -> list[PlayerWidget]:
        result: list[PlayerWidget] = []
        for i in range(self._players_row.count()):
            item = self._players_row.itemAt(i)
            if item is not None and isinstance(widget := item.widget(), PlayerWidget):
                result.append(widget)
        return result

    def add_video(self, path: Path) -> PlayerWidget | None:
        try:
            player = PlayerWidget(path)
        except VideoError as error:
            QMessageBox.warning(self, "Vidéo illisible", str(error))
            return None
        player.removeRequested.connect(lambda p=player: self._remove_player(p))
        player.stateChanged.connect(self._on_state_changed)
        self._players_row.addWidget(player, stretch=1)
        self._empty_label.hide()
        self._on_state_changed()
        return player

    def _remove_player(self, player: PlayerWidget) -> None:
        self._players_row.removeWidget(player)
        player.close_source()
        player.deleteLater()
        if not self.players():
            self._empty_label.show()
        self._on_state_changed()

    def _on_state_changed(self) -> None:
        """Session state changed; render panel refresh hooks in here (phase 4)."""

    # --- Project ---------------------------------------------------------

    def current_project(self) -> Project:
        return Project(
            videos=[player.to_video_entry() for player in self.players()],
            render=self._render_options,
            output_path=self._output_path,
        )

    def _add_video_dialog(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self, "Ajouter une vidéo", "", VIDEO_FILTER)
        if file_name:
            self.add_video(Path(file_name))

    def _open_project_dialog(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self, "Ouvrir un projet", "", PROJECT_FILTER)
        if not file_name:
            return
        try:
            project = Project.load(Path(file_name))
        except ProjectError as error:
            QMessageBox.critical(self, "Projet invalide", str(error))
            return

        for existing in self.players():
            self._remove_player(existing)
        self._project_path = Path(file_name)
        self._render_options = project.render
        self._output_path = project.output_path

        missing = []
        for entry in project.videos:
            if not entry.path.exists():
                missing.append(entry.path)
                continue
            player = self.add_video(entry.path)
            if player is not None:
                player.apply_entry(entry)
        if missing:
            names = "\n".join(str(path) for path in missing)
            QMessageBox.warning(self, "Vidéos introuvables", f"Fichiers introuvables :\n{names}")

    def _save_project_dialog(self) -> None:
        target = self._project_path
        if target is None:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Enregistrer le projet", "session.railcam.json", PROJECT_FILTER
            )
            if not file_name:
                return
            target = Path(file_name)
        self.current_project().save(target)
        self._project_path = target
        self.statusBar().showMessage(f"Projet enregistré : {target}", 5000)


def run_app() -> int:
    """Create the application, apply the theme, and run the main window."""
    app = QApplication(sys.argv)
    stylesheet = resources.files("railcam.gui").joinpath("style.qss").read_text(encoding="utf-8")
    app.setStyleSheet(stylesheet)
    window = MainWindow()
    window.show()
    return app.exec()
