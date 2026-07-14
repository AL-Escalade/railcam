"""Render panel: options, equivalent CLI command, and render execution."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QProcess, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from railcam.gui.progress import parse_progress, split_output_chunks
from railcam.gui.project import Project, RenderOptions
from railcam.gui.render import build_cli_args, format_cli_command, resolve_railcam_command


class RenderProcess(QWidget):
    """Runs the railcam CLI in a QProcess and reports progress."""

    progressed = Signal(str, int, int)  # stage, current, total
    logLine = Signal(str)
    finished = Signal(bool)  # success

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._process: QProcess | None = None

    @property
    def running(self) -> bool:
        return (
            self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning
        )

    def start(self, args: list[str], working_dir: Path) -> None:
        program, prefix = resolve_railcam_command()
        process = QProcess(self)
        process.setWorkingDirectory(str(working_dir))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._on_output)
        process.finished.connect(self._on_finished)
        self._process = process
        process.start(program, [*prefix, *args])

    def cancel(self) -> None:
        if self._process is not None and self.running:
            self._process.kill()

    def _on_output(self) -> None:
        if self._process is None:
            return
        chunk = bytes(self._process.readAllStandardOutput().data()).decode(
            "utf-8", errors="replace"
        )
        for line in split_output_chunks(chunk):
            progress = parse_progress(line)
            if progress is not None:
                self.progressed.emit(*progress)
            else:
                self.logLine.emit(line)

    def _on_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self.finished.emit(exit_code == 0)


class RenderPanel(QWidget):
    """Render options, live CLI command, progress and result actions."""

    optionsChanged = Signal()

    def __init__(
        self, project_provider: Callable[[], Project], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._project_provider = project_provider
        self._process = RenderProcess(self)
        self._process.progressed.connect(self._on_progress)
        self._process.logLine.connect(self._append_log)
        self._process.finished.connect(self._on_finished)
        self._last_output: Path | None = None
        self._build_ui()

    # --- UI ----------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        options_row = QHBoxLayout()
        options_row.addWidget(QLabel("Format :"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["mp4", "gif"])
        options_row.addWidget(self.format_combo)

        options_row.addSpacing(10)
        options_row.addWidget(QLabel("Hauteur :"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(0, 4320)
        self.height_spin.setSpecialValueText("auto")
        self.height_spin.setSingleStep(120)
        options_row.addWidget(self.height_spin)

        options_row.addSpacing(10)
        options_row.addWidget(QLabel("Vitesse de sortie :"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.05, 4.0)
        self.speed_spin.setSingleStep(0.05)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSuffix("×")
        options_row.addWidget(self.speed_spin)

        options_row.addSpacing(10)
        self.debug_check = QCheckBox("Debug (squelette)")
        options_row.addWidget(self.debug_check)

        options_row.addSpacing(10)
        options_row.addWidget(QLabel("Sortie :"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("auto")
        options_row.addWidget(self.output_edit, stretch=1)
        browse = QToolButton()
        browse.setText("…")
        browse.clicked.connect(self._browse_output)
        options_row.addWidget(browse)
        layout.addLayout(options_row)

        for signal in (
            self.format_combo.currentIndexChanged,
            self.height_spin.valueChanged,
            self.speed_spin.valueChanged,
            self.debug_check.toggled,
            self.output_edit.textChanged,
        ):
            signal.connect(lambda *_: self.optionsChanged.emit())

        command_row = QHBoxLayout()
        self.command_edit = QLineEdit()
        self.command_edit.setReadOnly(True)
        self.command_edit.setObjectName("commandEdit")
        command_row.addWidget(self.command_edit, stretch=1)
        copy_button = QToolButton()
        copy_button.setText("📋")
        copy_button.setToolTip("Copier la commande")
        copy_button.clicked.connect(self._copy_command)
        command_row.addWidget(copy_button)
        layout.addLayout(command_row)

        action_row = QHBoxLayout()
        self.render_button = QPushButton("▶ Générer la vidéo")
        self.render_button.setObjectName("renderButton")
        self.render_button.clicked.connect(self._start_render)
        action_row.addWidget(self.render_button)

        self.cancel_button = QPushButton("Annuler")
        self.cancel_button.clicked.connect(self._process.cancel)
        self.cancel_button.hide()
        action_row.addWidget(self.cancel_button)

        self.stage_label = QLabel("")
        self.stage_label.setObjectName("stageLabel")
        action_row.addWidget(self.stage_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()
        action_row.addWidget(self.progress_bar, stretch=1)

        self.open_button = QPushButton("Ouvrir la vidéo")
        self.open_button.clicked.connect(self._open_result)
        self.open_button.hide()
        action_row.addWidget(self.open_button)

        self.validity_label = QLabel("")
        self.validity_label.setObjectName("validityLabel")
        action_row.addWidget(self.validity_label)
        action_row.addStretch()
        layout.addLayout(action_row)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(140)
        self.log_view.hide()
        layout.addWidget(self.log_view)

    # --- Options state -------------------------------------------------------

    def render_options(self) -> RenderOptions:
        height = self.height_spin.value()
        return RenderOptions(
            format=self.format_combo.currentText(),
            height=height if height > 0 else None,
            speed=round(self.speed_spin.value(), 4),
            debug=self.debug_check.isChecked(),
        )

    def output_path(self, project: Project) -> Path | None:
        text = self.output_edit.text().strip()
        if text:
            return Path(text)
        if project.videos:
            first = project.videos[0].path
            return first.with_name(f"{first.stem}_railcam.{self.render_options().format}")
        return None

    def apply_options(self, options: RenderOptions, output_path: Path | None) -> None:
        """Restore options from a loaded project."""
        self.format_combo.setCurrentText(options.format)
        self.height_spin.setValue(options.height or 0)
        self.speed_spin.setValue(options.speed)
        self.debug_check.setChecked(options.debug)
        self.output_edit.setText(str(output_path) if output_path else "")

    def refresh(self, project: Project, problems: list[str]) -> None:
        """Update the displayed command and render availability."""
        self.command_edit.setText(format_cli_command(project))
        ready = not problems and not self._process.running
        self.render_button.setEnabled(ready)
        self.validity_label.setText(" · ".join(problems))

    # --- Actions ---------------------------------------------------------------

    def _browse_output(self) -> None:
        extension = self.render_options().format
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Fichier de sortie", f"comparison.{extension}", f"*.{extension}"
        )
        if file_name:
            self.output_edit.setText(file_name)

    def _copy_command(self) -> None:
        QGuiApplication.clipboard().setText(self.command_edit.text())

    def _start_render(self) -> None:
        project = self._project_provider()
        if not project.videos:
            return
        self._last_output = project.output_path
        self.log_view.clear()
        self.log_view.hide()
        self.open_button.hide()
        self.cancel_button.show()
        self.render_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.stage_label.setText("Démarrage…")
        working_dir = project.videos[0].path.parent
        self._process.start(build_cli_args(project), working_dir)

    def _on_progress(self, stage: str, current: int, total: int) -> None:
        percent = int(current * 100 / total) if total else 0
        self.progress_bar.setValue(percent)
        self.stage_label.setText(f"{stage} {percent}%")

    def _append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    def _on_finished(self, success: bool) -> None:
        self.cancel_button.hide()
        self.progress_bar.hide()
        self.render_button.setEnabled(True)
        if success:
            self.stage_label.setText("✔ Rendu terminé")
            if self._last_output is not None:
                self.open_button.show()
        else:
            self.stage_label.setText("✖ Échec du rendu — voir le journal")
            self.log_view.show()
        self.optionsChanged.emit()

    def _open_result(self) -> None:
        if self._last_output is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output)))
