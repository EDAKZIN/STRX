from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

from PyQt6.QtCore import QEvent, Qt, QUrl
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QStackedWidget,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QCheckBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from config import configure_model_environment
from core.exporter import StrExporter
from core.ocr_worker import OcrWorker
from core.timecode import format_timecode, parse_timecode
from models.subtitle_segment import SubtitleSegment
from ui.timeline_widget import TimelineWidget
from ui.toggle_switch import ToggleSwitch


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("STRX MVP - OCR STR")
        self.resize(1700, 980)

        self.video_path: Path | None = None
        self.video_duration_ms = 0
        self.segments: dict[str, SubtitleSegment] = {}
        self.row_segment_ids: list[str] = []
        self.segment_row_index: dict[str, int] = {}
        self.process_row_segment_ids: list[str] = []
        self.worker: OcrWorker | None = None
        self.ignore_table_events = False
        self.ignore_selection_sync = False
        self.ocr_completed = False
        self.gpu_available = False
        self.gpu_name = ""
        self.gpu_error = ""
        self.cpu_name = self._detect_cpu_name()

        self._build_ui()
        self._connect_events()
        self.refresh_gpu_status()
        self.update_action_buttons()

    def _build_ui(self) -> None:
        self.setObjectName("mainWindow")
        root = QWidget(self)
        root.setObjectName("root")
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        controls = QHBoxLayout()
        self.btn_open = QPushButton("Abrir video")
        self.btn_open.setObjectName("btnGhost")
        self.btn_play = QPushButton("Play/Pausa")
        self.btn_play.setObjectName("btnGhost")
        self.btn_start_ocr = QPushButton("Iniciar OCR")
        self.btn_start_ocr.setObjectName("btnPrimary")
        self.btn_cancel_ocr = QPushButton("Cancelar OCR")
        self.btn_cancel_ocr.setObjectName("btnDanger")
        self.btn_export = QPushButton("Exportar .str")
        self.btn_export.setObjectName("btnSecondary")
        self.btn_view_process = QPushButton("Ver proceso OCR")
        self.btn_view_process.setObjectName("btnSecondary")
        self.btn_back_editor = QPushButton("Volver al editor")
        self.btn_back_editor.setObjectName("btnSecondary")
        self.btn_back_editor.setVisible(False)
        self.chk_use_gpu = ToggleSwitch()
        self.chk_use_gpu.setChecked(True)
        self.chk_use_gpu.setObjectName("chkGpu")
        self.lbl_gpu_toggle = QLabel("Usar GPU (CUDA)")
        self.lbl_gpu_toggle.setObjectName("gpuToggleLabel")
        self.lbl_gpu_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lbl_gpu_toggle.mousePressEvent = self.on_gpu_label_clicked  # type: ignore[assignment]
        self.gpu_toggle_wrap = QWidget()
        gpu_toggle_layout = QHBoxLayout(self.gpu_toggle_wrap)
        gpu_toggle_layout.setContentsMargins(0, 0, 0, 0)
        gpu_toggle_layout.setSpacing(8)
        gpu_toggle_layout.addWidget(self.chk_use_gpu)
        gpu_toggle_layout.addWidget(self.lbl_gpu_toggle)
        self.lbl_sample_fps = QLabel("Muestras FPS:")
        self.lbl_sample_fps.setObjectName("toolLabel")
        self.sample_fps = QSpinBox()
        self.sample_fps.setMinimum(1)
        self.sample_fps.setMaximum(10)
        self.sample_fps.setValue(1)
        self.sample_fps.setObjectName("fpsSpin")
        self.lbl_ocr_engine = QLabel("OCR: PaddleOCR (EN)")
        self.lbl_ocr_engine.setObjectName("toolLabel")
        self.chk_language_correction = QCheckBox("Corregir texto OCR")
        self.chk_language_correction.setObjectName("chkCorrection")
        self.chk_language_correction.setChecked(True)
        self.gpu_status_label = QLabel("GPU: validando...")
        self.gpu_status_label.setObjectName("gpuStatus")

        controls.addWidget(self.btn_open)
        controls.addWidget(self.btn_play)
        controls.addWidget(self.lbl_sample_fps)
        controls.addWidget(self.sample_fps)
        controls.addWidget(self.lbl_ocr_engine)
        controls.addWidget(self.chk_language_correction)
        controls.addWidget(self.gpu_toggle_wrap)
        controls.addWidget(self.btn_start_ocr)
        controls.addWidget(self.btn_cancel_ocr)
        controls.addWidget(self.btn_export)
        controls.addWidget(self.btn_view_process)
        controls.addWidget(self.btn_back_editor)
        controls.addStretch(1)
        controls.addWidget(self.gpu_status_label)

        self.progress = QProgressBar()
        self.progress.setObjectName("ocrProgress")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status_label = QLabel("Listo.")
        self.status_label.setObjectName("statusLabel")

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(520)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.installEventFilter(self)

        self.video_overlay_label = QLabel("", self.video_widget)
        self.video_overlay_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)
        self.video_overlay_label.setObjectName("subtitleOverlay")
        self.video_overlay_label.setWordWrap(True)
        self.video_overlay_label.hide()
        self.video_overlay_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self.video_container = QWidget()
        self.video_container.setObjectName("videoContainer")
        video_container_layout = QVBoxLayout(self.video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)
        video_container_layout.addWidget(self.video_widget)
        self._layout_video_overlay()

        self.timeline = TimelineWidget()
        self.timeline.setObjectName("timeline")
        self.timeline.setMinimumHeight(240)

        self.table = QTableWidget(0, 3)
        self.table.setObjectName("editorTable")
        self.table.setHorizontalHeaderLabels(["Inicio", "Fin", "Texto"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)

        self.process_table = QTableWidget(0, 4)
        self.process_table.setObjectName("processTable")
        self.process_table.setHorizontalHeaderLabels(["#", "Inicio", "Fin", "Texto OCR"])
        self.process_table.horizontalHeader().setStretchLastSection(True)
        self.process_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.process_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.process_table.setAlternatingRowColors(True)
        self.process_log = QPlainTextEdit()
        self.process_log.setObjectName("processLog")
        self.process_log.setReadOnly(True)
        self.process_log.setPlaceholderText("El estado de OCR se verá aquí.")

        process_panel = QWidget()
        process_layout = QVBoxLayout(process_panel)
        process_layout.setContentsMargins(0, 0, 0, 0)
        process_layout.addWidget(self.process_table, 4)
        process_layout.addWidget(self.process_log, 2)

        editor_splitter = QSplitter()
        video_panel = QWidget()
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(8)
        video_layout.addWidget(self.video_container, 6)
        video_layout.addWidget(self.timeline, 3)
        editor_splitter.addWidget(video_panel)
        editor_splitter.addWidget(self.table)
        editor_splitter.setStretchFactor(0, 7)
        editor_splitter.setStretchFactor(1, 4)
        editor_splitter.setSizes([1200, 560])
        editor_splitter.handle(1).setCursor(Qt.CursorShape.SplitHCursor)

        editor_page = QWidget()
        editor_layout = QVBoxLayout(editor_page)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.addWidget(editor_splitter)

        self.content_stack = QStackedWidget()
        self.content_stack.addWidget(editor_page)
        self.content_stack.addWidget(process_panel)

        root_layout.addLayout(controls)
        root_layout.addWidget(self.progress)
        root_layout.addWidget(self.status_label)
        root_layout.addWidget(self.content_stack)

        self.audio_output = QAudioOutput(self)
        self.media_player = QMediaPlayer(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        self.video_sink = self.video_widget.videoSink()

        self._apply_styles()

    def _connect_events(self) -> None:
        self.btn_open.clicked.connect(self.open_video)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_start_ocr.clicked.connect(self.start_ocr)
        self.btn_cancel_ocr.clicked.connect(self.cancel_ocr)
        self.btn_export.clicked.connect(self.export_str)
        self.btn_view_process.clicked.connect(self.show_process_view)
        self.btn_back_editor.clicked.connect(self.show_editor_view)
        self.chk_use_gpu.toggled.connect(self.on_compute_toggle_changed)

        self.table.itemChanged.connect(self.on_table_item_changed)
        self.table.itemSelectionChanged.connect(self.on_table_selection_changed)
        self.process_table.itemSelectionChanged.connect(self.on_process_selection_changed)

        self.timeline.segment_selected.connect(self.on_timeline_segment_selected)
        self.timeline.segment_changing.connect(self.on_timeline_segment_changing)
        self.timeline.segment_changed.connect(self.on_timeline_segment_changed)

        self.media_player.positionChanged.connect(self.on_player_position_changed)
        self.media_player.durationChanged.connect(self.on_player_duration_changed)

    def open_video(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar video",
            "",
            "Videos (*.mp4 *.mkv *.avi *.mov *.wmv);;Todos los archivos (*)",
        )
        if not path_str:
            return

        self.video_path = Path(path_str)
        self.media_player.setSource(QUrl.fromLocalFile(str(self.video_path)))
        self._set_video_subtitle_text("")
        self.video_overlay_label.hide()
        self.status_label.setText(f"Video cargado: {self.video_path.name}")
        self.update_action_buttons()

    def toggle_play(self) -> None:
        if self.media_player.isPlaying():
            self.media_player.pause()
            return
        self.media_player.play()

    def start_ocr(self) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "OCR", "Ya hay un proceso OCR en ejecución.")
            return
        if not self.video_path:
            QMessageBox.warning(self, "OCR", "Primero debes cargar un video.")
            return
        self.refresh_gpu_status()
        use_gpu = self.chk_use_gpu.isChecked() and self.gpu_available
        enable_correction = self.chk_language_correction.isChecked()

        self.progress.setValue(0)
        self.ocr_completed = False
        self._set_video_subtitle_text("")
        self.video_overlay_label.hide()
        self.segments.clear()
        self.clear_process_table()
        self.process_log.clear()
        self.refresh_table()
        self.refresh_timeline()
        self.append_process_log("Backend OCR: paddleocr")
        self.append_process_log("Idioma OCR: en")
        self.append_process_log(
            "Corrección de idioma: activada" if enable_correction else "Corrección de idioma: desactivada"
        )
        if use_gpu:
            self.append_process_log(f"GPU activa para OCR: {self.gpu_name}")
        else:
            warning_text = "OCR en CPU: el proceso será más lento que con GPU."
            QMessageBox.warning(self, "OCR en CPU", warning_text)
            self.append_process_log(warning_text)

        model_paths = configure_model_environment()
        self.worker = OcrWorker(
            video_path=self.video_path,
            sample_fps=float(self.sample_fps.value()),
            model_paths=model_paths,
            enable_language_correction=enable_correction,
            use_gpu=use_gpu,
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.status.connect(self.on_worker_status)
        self.worker.error.connect(self.on_worker_error)
        self.worker.segment_found.connect(self.on_segment_found)
        self.worker.completed.connect(self.on_worker_completed)
        self.worker.cancelled.connect(self.on_worker_cancelled)
        self.worker.finished.connect(self.on_worker_finished)

        self.worker.start()
        self.show_process_view()
        self.update_action_buttons()
        self.status_label.setText("Iniciando OCR...")

    def cancel_ocr(self) -> None:
        if not self.worker or not self.worker.isRunning():
            return
        self.worker.cancel()
        self.status_label.setText("Cancelando OCR...")
        self.append_process_log("Solicitud de cancelación enviada.")

    def on_worker_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.append_process_log(message)

    def on_worker_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error OCR", message)
        self.status_label.setText("OCR finalizó con error.")
        self.append_process_log(f"ERROR: {message}")
        self.update_action_buttons()

    def on_segment_found(self, segment: SubtitleSegment) -> None:
        self.segments[segment.id] = segment
        self.append_process_row(segment)
        self.refresh_table()
        self.refresh_timeline()
        self.update_subtitle_overlay(self.media_player.position())

    def on_worker_completed(self) -> None:
        self.ocr_completed = True
        self.update_subtitle_overlay(self.media_player.position())
        self.status_label.setText(f"OCR finalizado. Segmentos detectados: {len(self.segments)}")
        self.append_process_log("OCR finalizado correctamente.")

    def on_worker_cancelled(self) -> None:
        self.status_label.setText("OCR cancelado por usuario.")
        self.append_process_log("OCR cancelado por usuario.")

    def on_worker_finished(self) -> None:
        self.worker = None
        self.update_action_buttons()

    def sorted_segments(self) -> list[SubtitleSegment]:
        return sorted(self.segments.values(), key=lambda item: (item.start_ms, item.end_ms, item.id))

    def refresh_table(self) -> None:
        self.ignore_table_events = True
        try:
            segments = self.sorted_segments()
            self.table.setRowCount(len(segments))
            self.row_segment_ids = []
            self.segment_row_index = {}

            for row, segment in enumerate(segments):
                self.row_segment_ids.append(segment.id)
                self.segment_row_index[segment.id] = row
                self.table.setItem(row, 0, QTableWidgetItem(format_timecode(segment.start_ms)))
                self.table.setItem(row, 1, QTableWidgetItem(format_timecode(segment.end_ms)))
                self.table.setItem(row, 2, QTableWidgetItem(segment.text))
        finally:
            self.ignore_table_events = False

    def refresh_timeline(self) -> None:
        self.timeline.set_segments(self.sorted_segments(), self.video_duration_ms)
        if self.timeline.selected_segment_id:
            self.timeline.select_segment(self.timeline.selected_segment_id, emit_signal=False)

    def on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self.ignore_table_events:
            return
        row = item.row()
        column = item.column()
        if row < 0 or row >= len(self.row_segment_ids):
            return

        segment_id = self.row_segment_ids[row]
        segment = self.segments.get(segment_id)
        if not segment:
            return

        try:
            if column == 0:
                new_start_ms = parse_timecode(item.text())
                if new_start_ms >= segment.end_ms:
                    raise ValueError("El inicio debe ser menor al fin.")
                segment.start_ms = new_start_ms
            elif column == 1:
                new_end_ms = parse_timecode(item.text())
                if new_end_ms <= segment.start_ms:
                    raise ValueError("El fin debe ser mayor al inicio.")
                segment.end_ms = new_end_ms
            elif column == 2:
                segment.text = item.text()
            segment.source = "manual"
            self.refresh_timeline()
            self.update_subtitle_overlay(self.media_player.position())
        except Exception as exc:
            self.status_label.setText(f"Edición inválida: {exc}")
            self.refresh_table()
            self.refresh_timeline()

    def on_table_selection_changed(self) -> None:
        if self.ignore_selection_sync:
            return

        selected_items = self.table.selectedItems()
        if not selected_items:
            return
        row = selected_items[0].row()
        if row < 0 or row >= len(self.row_segment_ids):
            return

        segment_id = self.row_segment_ids[row]
        segment = self.segments.get(segment_id)
        if not segment:
            return

        self.ignore_selection_sync = True
        try:
            self.timeline.select_segment(segment_id, emit_signal=False)
            self.media_player.setPosition(segment.start_ms)
        finally:
            self.ignore_selection_sync = False

    def on_timeline_segment_selected(self, segment_id: str) -> None:
        if self.ignore_selection_sync:
            return
        row = self.segment_row_index.get(segment_id)
        if row is None:
            return

        self.ignore_selection_sync = True
        try:
            self.table.selectRow(row)
            segment = self.segments.get(segment_id)
            if segment:
                self.media_player.setPosition(segment.start_ms)
        finally:
            self.ignore_selection_sync = False

    def on_timeline_segment_changing(self, segment_id: str, start_ms: int, end_ms: int) -> None:
        segment = self.segments.get(segment_id)
        if not segment:
            return
        segment.start_ms = start_ms
        segment.end_ms = end_ms
        segment.source = "manual"
        self.update_single_row(segment_id)

    def on_timeline_segment_changed(self, segment_id: str, start_ms: int, end_ms: int) -> None:
        segment = self.segments.get(segment_id)
        if not segment:
            return
        segment.start_ms = start_ms
        segment.end_ms = end_ms
        segment.source = "manual"
        self.refresh_table()
        self.refresh_timeline()
        self.timeline.select_segment(segment_id, emit_signal=False)
        self.update_subtitle_overlay(self.media_player.position())

    def update_single_row(self, segment_id: str) -> None:
        row = self.segment_row_index.get(segment_id)
        segment = self.segments.get(segment_id)
        if row is None or segment is None:
            return

        self.ignore_table_events = True
        try:
            self.table.item(row, 0).setText(format_timecode(segment.start_ms))
            self.table.item(row, 1).setText(format_timecode(segment.end_ms))
        finally:
            self.ignore_table_events = False

    def on_player_position_changed(self, position: int) -> None:
        self.timeline.set_playhead(position)
        self.update_subtitle_overlay(position)

    def on_player_duration_changed(self, duration: int) -> None:
        self.video_duration_ms = max(0, duration)
        self.refresh_timeline()

    def export_str(self) -> None:
        segments = self.sorted_segments()
        if not segments:
            QMessageBox.warning(self, "Exportación", "No hay segmentos para exportar.")
            return

        errors: list[str] = []
        for index, segment in enumerate(segments, start=1):
            if not segment.text.strip():
                errors.append(f"Fila {index}: texto vacío.")
            if segment.end_ms <= segment.start_ms:
                errors.append(f"Fila {index}: rango de tiempo inválido.")

        if errors:
            details = "\n".join(errors[:10])
            QMessageBox.warning(self, "Validación", f"No se puede exportar:\n{details}")
            return

        default_name = "subtitulos.str"
        if self.video_path:
            default_name = f"{self.video_path.stem}.str"

        output_path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar STR",
            str((self.video_path.parent if self.video_path else Path.cwd()) / default_name),
            "STR (*.str)",
        )
        if not output_path_str:
            return

        try:
            output_path = StrExporter.export(segments, output_path_str)
            self.status_label.setText(f"Archivo exportado: {output_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Exportación", f"No se pudo exportar: {exc}")

    def show_process_view(self) -> None:
        self.content_stack.setCurrentIndex(1)
        self.btn_view_process.setVisible(False)
        self.btn_back_editor.setVisible(True)

    def show_editor_view(self) -> None:
        self.content_stack.setCurrentIndex(0)
        self.btn_view_process.setVisible(True)
        self.btn_back_editor.setVisible(False)

    def clear_process_table(self) -> None:
        self.process_table.setRowCount(0)
        self.process_row_segment_ids = []

    def append_process_row(self, segment: SubtitleSegment) -> None:
        row = self.process_table.rowCount()
        self.process_table.insertRow(row)
        self.process_row_segment_ids.append(segment.id)
        self.process_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        self.process_table.setItem(row, 1, QTableWidgetItem(format_timecode(segment.start_ms)))
        self.process_table.setItem(row, 2, QTableWidgetItem(format_timecode(segment.end_ms)))
        self.process_table.setItem(row, 3, QTableWidgetItem(segment.text))
        self.process_table.scrollToBottom()

    def on_process_selection_changed(self) -> None:
        selected_items = self.process_table.selectedItems()
        if not selected_items:
            return
        row = selected_items[0].row()
        if row < 0 or row >= len(self.process_row_segment_ids):
            return
        segment_id = self.process_row_segment_ids[row]
        segment = self.segments.get(segment_id)
        if not segment:
            return
        self.media_player.setPosition(segment.start_ms)
        self.timeline.select_segment(segment_id, emit_signal=True)

    def append_process_log(self, message: str) -> None:
        self.process_log.appendPlainText(message)

    def on_compute_toggle_changed(self, checked: bool) -> None:
        del checked
        self._apply_compute_status_badge()

    def refresh_gpu_status(self) -> None:
        previous_gpu_choice = self.chk_use_gpu.isChecked()
        self.gpu_available = False
        self.gpu_name = ""
        self.gpu_error = ""
        self.chk_use_gpu.setEnabled(False)
        self.chk_use_gpu.blockSignals(True)
        self.chk_use_gpu.setChecked(False)
        self.chk_use_gpu.blockSignals(False)
        try:
            import paddle
        except Exception as exc:
            self.gpu_error = f"No se pudo importar paddle: {exc}. Se usará CPU."
            self._apply_compute_status_badge()
            return

        try:
            has_cuda = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        except Exception:
            has_cuda = False

        if not has_cuda:
            self.gpu_error = "No se detectó CUDA en el entorno. Se usará CPU (más lento)."
            self._apply_compute_status_badge()
            return

        self.gpu_name = self._detect_gpu_name()
        self.gpu_available = True
        self.chk_use_gpu.setEnabled(True)
        self.chk_use_gpu.blockSignals(True)
        self.chk_use_gpu.setChecked(previous_gpu_choice)
        self.chk_use_gpu.blockSignals(False)
        self._apply_compute_status_badge()

    def _apply_compute_status_badge(self) -> None:
        use_gpu = self.chk_use_gpu.isChecked() and self.gpu_available
        if use_gpu:
            self.gpu_status_label.setText(f"GPU: {self.gpu_name}")
            self.gpu_status_label.setProperty("state", "ok")
        elif self.gpu_available:
            self.gpu_status_label.setText(f"CPU: {self.cpu_name}")
            self.gpu_status_label.setProperty("state", "cpu")
        else:
            self.gpu_status_label.setText(f"CPU: {self.cpu_name}")
            self.gpu_status_label.setProperty("state", "cpu")
        self.gpu_status_label.style().unpolish(self.gpu_status_label)
        self.gpu_status_label.style().polish(self.gpu_status_label)

    def _detect_cpu_name(self) -> str:
        if os.name == "nt":
            name = self._run_cpu_query(["wmic", "cpu", "get", "name"])
            if name:
                return name
            name = self._run_cpu_query(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name | Select-Object -First 1)",
                ]
            )
            if name:
                return name

        name = platform.processor().strip()
        if name:
            return name

        env_name = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
        if env_name:
            return env_name

        return "Procesador no identificado"

    @staticmethod
    def _run_cpu_query(command: list[str]) -> str:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=2, check=False)
        except Exception:
            return ""

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return ""

        if lines[0].lower() == "name" and len(lines) > 1:
            return lines[1]
        return lines[0]

    def _detect_gpu_name(self) -> str:
        if os.name == "nt":
            name = self._run_cpu_query(
                [
                    "nvidia-smi",
                    "--query-gpu=name",
                    "--format=csv,noheader",
                ]
            )
            if name:
                return name
        return "GPU CUDA"

    def update_action_buttons(self) -> None:
        worker_running = self.worker is not None and self.worker.isRunning()
        has_video = self.video_path is not None
        self.btn_start_ocr.setEnabled(has_video and not worker_running)
        self.btn_cancel_ocr.setEnabled(worker_running)
        self.btn_play.setEnabled(has_video)
        self.btn_export.setEnabled(bool(self.segments))

    def update_subtitle_overlay(self, position_ms: int) -> None:
        if not self.segments:
            self._set_video_subtitle_text("")
            self.video_overlay_label.hide()
            return

        active_texts = [
            segment.text.strip()
            for segment in self.sorted_segments()
            if segment.text.strip() and segment.start_ms <= position_ms <= segment.end_ms
        ]
        if not active_texts:
            self._set_video_subtitle_text("")
            self.video_overlay_label.hide()
            return

        visible_text = "\n".join(active_texts[:3])
        self._set_video_subtitle_text(visible_text)
        self.video_overlay_label.setText(visible_text)
        self._layout_video_overlay()
        self.video_overlay_label.show()

    def _set_video_subtitle_text(self, text: str) -> None:
        if self.video_sink is None:
            return
        self.video_sink.setSubtitleText(text)

    def on_gpu_label_clicked(self, event) -> None:
        if self.chk_use_gpu.isEnabled():
            self.chk_use_gpu.toggle()
        event.accept()

    def _layout_video_overlay(self) -> None:
        width = max(260, self.video_widget.width() - 80)
        height = max(64, min(160, int(self.video_widget.height() * 0.2)))
        x = max(16, (self.video_widget.width() - width) // 2)
        y = max(12, self.video_widget.height() - height - 18)
        self.video_overlay_label.setGeometry(x, y, width, height)
        self.video_overlay_label.raise_()

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if watched is self.video_widget and event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self._layout_video_overlay()
        return super().eventFilter(watched, event)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow#mainWindow {
                background-color: #070c15;
            }
            QWidget#root {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0a101b,
                    stop:0.45 #0b1220,
                    stop:1 #090f1a
                );
                color: #d7e3f5;
                font-size: 13px;
            }
            QLabel {
                color: #d7e3f5;
            }
            QLabel#toolLabel {
                color: #94a3b8;
                font-weight: 700;
            }
            QLabel#gpuToggleLabel {
                color: #cfd8e5;
                font-weight: 700;
            }
            QLabel#statusLabel {
                color: #dbeafe;
                font-weight: 600;
                background: rgba(15, 23, 42, 0.86);
                border: 1px solid #25364f;
                border-radius: 12px;
                padding: 9px 13px;
            }
            QLabel#gpuStatus {
                padding: 7px 13px;
                border-radius: 999px;
                font-weight: 700;
                border: 1px solid #1d4ed8;
                background: #0f2038;
                color: #bfdbfe;
            }
            QLabel#gpuStatus[state="error"] {
                border-color: #7f1d1d;
                background: #2a1018;
                color: #ffe4e6;
            }
            QLabel#gpuStatus[state="ok"] {
                border-color: #0891b2;
                background: #092433;
                color: #a5f3fc;
            }
            QLabel#gpuStatus[state="cpu"] {
                border-color: #475569;
                background: #1a2330;
                color: #e2e8f0;
            }
            QWidget#videoContainer,
            QTableWidget#editorTable,
            QTableWidget#processTable,
            QPlainTextEdit#processLog,
            QGraphicsView#timeline {
                background: #0f141d;
                border: 1px solid #223247;
                border-radius: 14px;
            }
            QTableWidget#editorTable,
            QTableWidget#processTable,
            QPlainTextEdit#processLog {
                alternate-background-color: #121926;
                color: #e2ecfa;
                gridline-color: #202734;
                selection-background-color: #3a4e69;
                selection-color: #eff6ff;
            }
            QTableCornerButton::section,
            QHeaderView::section {
                background: #141b27;
                color: #aeb8c8;
                border: none;
                border-bottom: 1px solid #273040;
                padding: 8px;
                font-weight: 700;
            }
            QPlainTextEdit#processLog {
                font-family: Consolas, "Courier New", monospace;
                font-size: 12px;
            }
            QProgressBar#ocrProgress {
                border: 1px solid #263b57;
                border-radius: 999px;
                text-align: center;
                color: #dbeafe;
                font-weight: 700;
                background: #0f1728;
                height: 18px;
            }
            QProgressBar#ocrProgress::chunk {
                border-radius: 999px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #36506e,
                    stop:1 #4f728f
                );
            }
            QPushButton {
                min-height: 36px;
                padding: 8px 15px;
                border-radius: 11px;
                border: 1px solid #2b3441;
                background: #161d28;
                color: #e1e7f0;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #1d2532;
                border-color: #3a4658;
            }
            QPushButton:pressed {
                background: #0f141d;
            }
            QPushButton:disabled {
                background: #0e1626;
                color: #5e6f88;
                border-color: #202c3f;
            }
            QPushButton#btnPrimary {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2f4661,
                    stop:1 #466480
                );
                border-color: #607a96;
                color: #eff6ff;
            }
            QPushButton#btnPrimary:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3a5674,
                    stop:1 #557491
                );
            }
            QPushButton#btnDanger {
                background: #32131d;
                border-color: #7f1d1d;
                color: #ffe4e6;
            }
            QPushButton#btnDanger:hover {
                background: #451b27;
            }
            QPushButton#btnSecondary {
                background: #162233;
                border-color: #324a66;
                color: #d6e0ec;
            }
            QPushButton#btnSecondary:hover {
                background: #1d2d42;
            }
            QPushButton#btnGhost {
                background: #121923;
                border-color: #2f3946;
                color: #cad4e2;
            }
            QPushButton#btnGhost:hover {
                background: #1a2330;
            }
            QCheckBox#chkGpu {
                background: transparent;
            }
            QSpinBox#fpsSpin {
                min-width: 82px;
                min-height: 32px;
                padding: 3px 8px;
                border-radius: 9px;
                border: 1px solid #354253;
                background: #0f141d;
                color: #dbeafe;
            }
            QCheckBox#chkCorrection {
                color: #d1d9e6;
                font-weight: 600;
                spacing: 6px;
            }
            QCheckBox#chkCorrection::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid #465365;
                background: #131b28;
            }
            QCheckBox#chkCorrection::indicator:checked {
                background: #36506e;
                border-color: #5b7898;
            }
            QSpinBox#fpsSpin::up-button,
            QSpinBox#fpsSpin::down-button {
                width: 22px;
                border-left: 1px solid #3a4454;
                background: #1c2431;
            }
            QSpinBox#fpsSpin::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                border-top-right-radius: 8px;
            }
            QSpinBox#fpsSpin::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                border-bottom-right-radius: 8px;
                border-top: 1px solid #3a4454;
            }
            QSpinBox#fpsSpin::up-button:hover,
            QSpinBox#fpsSpin::down-button:hover {
                background: #2a3442;
            }
            QSpinBox#fpsSpin::up-arrow {
                image: none;
                width: 0px;
                height: 0px;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-bottom: 7px solid #d5e0ef;
            }
            QSpinBox#fpsSpin::down-arrow {
                image: none;
                width: 0px;
                height: 0px;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 7px solid #d5e0ef;
            }
            QSplitter::handle {
                background: #242d3c;
                width: 7px;
                margin: 2px;
                border-radius: 3px;
            }
            QScrollBar:vertical {
                background: #0d1626;
                width: 11px;
                margin: 10px 0 10px 0;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #3a4c64;
                min-height: 28px;
                border-radius: 5px;
            }
            QScrollBar:horizontal {
                background: #0d1626;
                height: 11px;
                margin: 0 10px 0 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal {
                background: #3a4c64;
                min-width: 28px;
                border-radius: 5px;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                width: 0px;
                height: 0px;
            }
            QLabel#subtitleOverlay {
                background: transparent;
                color: #f8fbff;
                border: none;
                border-radius: 0px;
                padding: 4px 10px;
                font-size: 21px;
                font-weight: 700;
            }
            """
        )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        super().closeEvent(event)
