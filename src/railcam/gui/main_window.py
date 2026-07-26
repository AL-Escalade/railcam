"""Main window: toolbar, side-by-side players, project open/save."""

from __future__ import annotations

import sys
from importlib import resources
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
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
from railcam.gui.render_panel import RenderPanel
from railcam.gui.transport import SyncPlayback, TransportBar
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

        self.playback = SyncPlayback(self.players)
        root.addWidget(TransportBar(self.playback))

        self.render_panel = RenderPanel(self.current_project)
        self.render_panel.optionsChanged.connect(self._on_state_changed)
        root.addWidget(self.render_panel)

        space = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        space.activated.connect(self.playback.toggle)

        self.setCentralWidget(central)
        self._on_state_changed()

    # --- Toolbar ---------------------------------------------------------

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Principal")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        new_action = toolbar.addAction("🆕 Nouveau projet")
        new_action.triggered.connect(self._new_project)
        open_action = toolbar.addAction("📁 Ouvrir un projet")
        open_action.triggered.connect(self._open_project_dialog)
        save_action = toolbar.addAction("💾 Enregistrer")
        save_action.triggered.connect(self._save_project)
        save_as_action = toolbar.addAction("💾 Enregistrer sous…")
        save_as_action.triggered.connect(self._save_project_as)
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
        self.playback.forget(player)
        self._players_row.removeWidget(player)
        player.close_source()
        player.deleteLater()
        if not self.players():
            self._empty_label.show()
        self._on_state_changed()

    def _on_state_changed(self) -> None:
        """Refresh the CLI command display and render availability."""
        problems = []
        players = self.players()
        if not players:
            problems.append("Ajoutez au moins une vidéo")
        problems.extend(
            f"Plage invalide sur {player.source.path.name}"
            for player in players
            if not player.is_range_valid()
        )
        self.render_panel.refresh(self.current_project(), problems)

    # --- Project ---------------------------------------------------------

    def current_project(self) -> Project:
        project = Project(
            videos=[player.to_video_entry() for player in self.players()],
            render=self.render_panel.render_options(),
        )
        project.output_path = self.render_panel.output_path(project)
        return project

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
        self.setWindowTitle(f"railcam — {self._project_path.name}")
        self.render_panel.apply_options(project.render, project.output_path)

        for entry in project.videos:
            path = entry.path
            if not path.exists():
                relocated = self._relocate_video(path)
                if relocated is None:
                    continue
                path = relocated
            player = self.add_video(path)
            if player is not None:
                player.apply_entry(entry)

    def _relocate_video(self, path: Path) -> Path | None:
        """Offer to locate a project video whose file no longer exists."""
        answer = QMessageBox.question(
            self,
            "Vidéo introuvable",
            f"Fichier introuvable :\n{path}\n\nVoulez-vous le relocaliser ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return None
        file_name, _ = QFileDialog.getOpenFileName(
            self, f"Relocaliser {path.name}", "", VIDEO_FILTER
        )
        return Path(file_name) if file_name else None

    def _new_project(self) -> None:
        """Reset the session: remove all videos and restore default options."""
        if self.players():
            answer = QMessageBox.question(
                self,
                "Nouveau projet",
                "Fermer la session en cours et repartir de zéro ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.playback.stop()
        for player in self.players():
            self._remove_player(player)
        self.render_panel.apply_options(RenderOptions(), None)
        self._project_path = None
        self.setWindowTitle("railcam")
        self.statusBar().showMessage("Nouveau projet", 3000)

    def _save_project(self) -> None:
        if self._project_path is None:
            self._save_project_as()
        else:
            self._write_project(self._project_path)

    def _save_project_as(self) -> None:
        suggestion = str(self._project_path) if self._project_path else "session.railcam.json"
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer le projet sous", suggestion, PROJECT_FILTER
        )
        if file_name:
            self._write_project(Path(file_name))

    def _write_project(self, target: Path) -> None:
        self.current_project().save(target)
        self._project_path = target
        self.setWindowTitle(f"railcam — {target.name}")
        self.statusBar().showMessage(f"Projet enregistré : {target}", 5000)


def run_app() -> int:
    """Create the application, apply the theme, and run the main window."""
    app = QApplication(sys.argv)
    stylesheet = resources.files("railcam.gui").joinpath("style.qss").read_text(encoding="utf-8")
    app.setStyleSheet(stylesheet)
    window = MainWindow()
    window.show()
    return app.exec()
