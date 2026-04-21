from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from difflib import SequenceMatcher

from PyQt6.QtCore import QEvent, QPoint, QRect, QSize, Qt, QUrl
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
    QRubberBand,
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
        self.setWindowTitle("STRX")
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
        self.video_frame_size = QSize()
        self.region_selection_rect: QRect | None = None
        self.region_drag_origin: QPoint | None = None

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
        self.btn_toggle_region = QPushButton("OCR por área")
        self.btn_toggle_region.setObjectName("btnToggleRegion")
        self.btn_toggle_region.setCheckable(True)
        self.btn_toggle_region.setChecked(False)
        self.gpu_status_label = QLabel("GPU: validando...")
        self.gpu_status_label.setObjectName("gpuStatus")

        controls.addWidget(self.btn_open)
        controls.addWidget(self.btn_play)
        controls.addWidget(self.lbl_sample_fps)
        controls.addWidget(self.sample_fps)
        controls.addWidget(self.lbl_ocr_engine)
        controls.addWidget(self.chk_language_correction)
        controls.addWidget(self.gpu_toggle_wrap)
        controls.addWidget(self.btn_toggle_region)
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

        # Tool window flotante sobre el rendering DirectX del QVideoWidget
        self.region_selection_band = QLabel()
        self.region_selection_band.setObjectName("selectionFrame")
        self.region_selection_band.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        )
        self.region_selection_band.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.region_selection_band.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.region_selection_band.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.region_selection_band.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.region_selection_band.setAutoFillBackground(False)
        self.region_selection_band.setStyleSheet(
            "border: 2px solid #0ea5e9; border-radius: 3px; background: transparent;"
        )
        self.region_selection_band.hide()
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
        self.btn_toggle_region.toggled.connect(self.on_region_toggle_changed)

        self.table.itemChanged.connect(self.on_table_item_changed)
        self.table.itemSelectionChanged.connect(self.on_table_selection_changed)
        self.process_table.itemSelectionChanged.connect(self.on_process_selection_changed)

        self.timeline.segment_selected.connect(self.on_timeline_segment_selected)
        self.timeline.segment_changing.connect(self.on_timeline_segment_changing)
        self.timeline.segment_changed.connect(self.on_timeline_segment_changed)
        self.timeline.seek_requested.connect(self.media_player.setPosition)
        self.timeline.scrub_started.connect(self.on_timeline_scrub_started)
        self.timeline.scrub_ended.connect(self.on_timeline_scrub_ended)

        self.media_player.positionChanged.connect(self.on_player_position_changed)
        self.media_player.durationChanged.connect(self.on_player_duration_changed)
        if self.video_sink is not None:
            self.video_sink.videoFrameChanged.connect(self.on_video_frame_changed)
        self.content_stack.currentChanged.connect(self._on_page_changed)

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
        self.video_frame_size = QSize()
        self.region_drag_origin = None
        self.region_selection_rect = None
        self._sync_region_selection_band()
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
        crop_region = self._selected_crop_region()
        if crop_region is not None:
            x_norm, y_norm, width_norm, height_norm = crop_region
            self.append_process_log(
                f"OCR por área activo: x={x_norm:.3f}, y={y_norm:.3f}, ancho={width_norm:.3f}, alto={height_norm:.3f}"
            )
        elif self.btn_toggle_region.isChecked():
            self.append_process_log("OCR por área activo sin selección válida. Se usará pantalla completa.")

        model_paths = configure_model_environment()
        self.worker = OcrWorker(
            video_path=self.video_path,
            sample_fps=float(self.sample_fps.value()),
            model_paths=model_paths,
            enable_language_correction=enable_correction,
            use_gpu=use_gpu,
            crop_region_norm=crop_region,
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
        merged = False
        for existing_segment in list(self.segments.values()):
            similarity = SequenceMatcher(None, existing_segment.text.lower(), segment.text.lower()).ratio()
            if similarity >= 0.75:
                gap1 = segment.start_ms - existing_segment.end_ms
                gap2 = existing_segment.start_ms - segment.end_ms
                if (0 <= gap1 <= 400) or (0 <= gap2 <= 400) or (gap1 < 0 and gap2 < 0):
                    existing_segment.start_ms = min(existing_segment.start_ms, segment.start_ms)
                    existing_segment.end_ms = max(existing_segment.end_ms, segment.end_ms)
                    if segment.confidence is not None:
                        if existing_segment.confidence is None:
                            existing_segment.confidence = segment.confidence
                        else:
                            existing_segment.confidence = max(existing_segment.confidence, segment.confidence)
                    merged = True
                    try:
                        row = self.process_row_segment_ids.index(existing_segment.id)
                        self.process_table.item(row, 1).setText(format_timecode(existing_segment.start_ms))
                        self.process_table.item(row, 2).setText(format_timecode(existing_segment.end_ms))
                    except ValueError:
                        pass
                    break
                    
        if not merged:
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

    def on_timeline_scrub_started(self) -> None:
        self.was_playing_before_scrub = self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        self.media_player.pause()

    def on_timeline_scrub_ended(self) -> None:
        if getattr(self, "was_playing_before_scrub", False):
            self.media_player.play()

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
        # Ocultar el marco de selección al salir del editor
        self.region_selection_band.hide()

    def show_editor_view(self) -> None:
        self.content_stack.setCurrentIndex(0)
        self.btn_view_process.setVisible(True)
        self.btn_back_editor.setVisible(False)
        self._sync_region_selection_band()

    def _on_page_changed(self, index: int) -> None:
        if index == 0:
            self._sync_region_selection_band()
        else:
            self.region_selection_band.hide()

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            if self.windowState() & Qt.WindowState.WindowMinimized:
                self.region_selection_band.hide()

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

    def on_region_toggle_changed(self, checked: bool) -> None:
        self.region_drag_origin = None
        if checked:
            self.video_widget.setCursor(Qt.CursorShape.CrossCursor)
            self.status_label.setText("OCR por área activado. Arrastra sobre el video para seleccionar un área.")
        else:
            self.video_widget.unsetCursor()
            self.status_label.setText("OCR por área desactivado. Se usará pantalla completa.")
        self._sync_region_selection_band()

    def on_video_frame_changed(self, frame) -> None:
        size_getter = getattr(frame, "size", None)
        frame_size = size_getter() if callable(size_getter) else QSize()
        if not isinstance(frame_size, QSize):
            return
        if frame_size.width() <= 0 or frame_size.height() <= 0:
            return
        if frame_size == self.video_frame_size:
            return
        self.video_frame_size = frame_size
        self._sync_region_selection_band()

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
        self.btn_toggle_region.setEnabled(has_video and not worker_running)

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
        self._sync_region_selection_band()

    def _video_display_rect(self) -> QRect:
        widget_rect = self.video_widget.rect()
        if widget_rect.width() <= 0 or widget_rect.height() <= 0:
            return QRect()
        if self.video_frame_size.width() <= 0 or self.video_frame_size.height() <= 0:
            return widget_rect

        video_aspect = self.video_frame_size.width() / self.video_frame_size.height()
        widget_aspect = widget_rect.width() / widget_rect.height()
        if widget_aspect > video_aspect:
            height = widget_rect.height()
            width = max(1, int(round(height * video_aspect)))
            x = (widget_rect.width() - width) // 2
            return QRect(x, 0, width, height)

        width = widget_rect.width()
        height = max(1, int(round(width / video_aspect)))
        y = (widget_rect.height() - height) // 2
        return QRect(0, y, width, height)

    def _normalize_selection_rect(self, start: QPoint, end: QPoint) -> QRect:
        video_rect = self._video_display_rect()
        if video_rect.isEmpty():
            return QRect()
        min_x, max_x = video_rect.left(), video_rect.right()
        min_y, max_y = video_rect.top(), video_rect.bottom()
        start_x = min(max(start.x(), min_x), max_x)
        start_y = min(max(start.y(), min_y), max_y)
        end_x = min(max(end.x(), min_x), max_x)
        end_y = min(max(end.y(), min_y), max_y)
        return QRect(QPoint(start_x, start_y), QPoint(end_x, end_y)).normalized().intersected(video_rect)

    def _sync_region_selection_band(self) -> None:
        if hasattr(self, "content_stack") and self.content_stack.currentIndex() != 0:
            self.region_selection_band.hide()
            return
        if not self.btn_toggle_region.isChecked():
            self.region_selection_band.hide()
            return
        
        # Determinar el rectángulo visible
        rect_to_show = None
        
        if self.region_drag_origin is not None and self.region_selection_rect is not None:
            video_rect = self._video_display_rect()
            if not video_rect.isEmpty():
                visible_rect = self.region_selection_rect.intersected(video_rect)
                if visible_rect.width() >= 1 and visible_rect.height() >= 1:
                    rect_to_show = visible_rect
        elif self.region_selection_rect is not None:
            video_rect = self._video_display_rect()
            if not video_rect.isEmpty():
                visible_rect = self.region_selection_rect.intersected(video_rect)
                if visible_rect.width() > 1 and visible_rect.height() > 1:
                    rect_to_show = visible_rect
        
        if rect_to_show is None:
            self.region_selection_band.hide()
            return
            
        self.region_selection_band.setStyleSheet(
            "border: 2px solid #0ea5e9; border-radius: 3px; background: transparent;"
        )
        # Coordenadas globales para la Tool window flotante
        global_pos = self.video_widget.mapToGlobal(rect_to_show.topLeft())
        self.region_selection_band.setGeometry(QRect(global_pos, rect_to_show.size()))
        self.region_selection_band.show()
        self.region_selection_band.raise_()

    def _selected_crop_region(self) -> tuple[float, float, float, float] | None:
        if not self.btn_toggle_region.isChecked() or self.region_selection_rect is None:
            return None
        video_rect = self._video_display_rect()
        if video_rect.width() <= 0 or video_rect.height() <= 0:
            return None
        selection = self.region_selection_rect.intersected(video_rect)
        if selection.width() <= 2 or selection.height() <= 2:
            return None
        x_norm = (selection.left() - video_rect.left()) / video_rect.width()
        y_norm = (selection.top() - video_rect.top()) / video_rect.height()
        width_norm = selection.width() / video_rect.width()
        height_norm = selection.height() / video_rect.height()
        if width_norm <= 0.01 or height_norm <= 0.01:
            return None
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        width_norm = max(0.0, min(1.0 - x_norm, width_norm))
        height_norm = max(0.0, min(1.0 - y_norm, height_norm))
        return x_norm, y_norm, width_norm, height_norm

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if watched is self.video_widget:
            event_type = event.type()
            if event_type in (QEvent.Type.Resize, QEvent.Type.Show):
                self._layout_video_overlay()

            if self.btn_toggle_region.isChecked():
                # Mostrar marco deshabilitado cuando el mouse está fuera del área de video
                if event_type == QEvent.Type.MouseMove and self.region_drag_origin is None:
                    mouse_pos = event.position().toPoint()
                    video_rect = self._video_display_rect()
                    if not video_rect.contains(mouse_pos):
                        # Mouse fuera del área de video - cursor bloqueado
                        self.video_widget.setCursor(Qt.CursorShape.ForbiddenCursor)
                        self._hide_disabled_frame()
                    else:
                        # Mouse dentro del video - cursor de selección
                        self.video_widget.setCursor(Qt.CursorShape.CrossCursor)
                        self._hide_disabled_frame()
                    return True
                    
                if event_type == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                    click_position = event.position().toPoint()
                    if self._video_display_rect().contains(click_position):
                        self.region_drag_origin = click_position
                        self.region_selection_rect = QRect(click_position, click_position)
                        self._sync_region_selection_band()
                        return True
                elif event_type == QEvent.Type.MouseMove and self.region_drag_origin is not None:
                    current_position = event.position().toPoint()
                    self.region_selection_rect = self._normalize_selection_rect(self.region_drag_origin, current_position)
                    self._sync_region_selection_band()
                    return True
                elif (
                    event_type == QEvent.Type.MouseButtonRelease
                    and event.button() == Qt.MouseButton.LeftButton
                    and self.region_drag_origin is not None
                ):
                    release_position = event.position().toPoint()
                    self.region_selection_rect = self._normalize_selection_rect(self.region_drag_origin, release_position)
                    self.region_drag_origin = None
                    if self.region_selection_rect.width() < 12 or self.region_selection_rect.height() < 12:
                        self.region_selection_rect = None
                        self.status_label.setText("Selección descartada. Se usará pantalla completa.")
                    else:
                        self.status_label.setText("Área de OCR seleccionada.")
                    self._sync_region_selection_band()
                    return True
        return super().eventFilter(watched, event)

    def _show_disabled_frame(self) -> None:
        video_rect = self._video_display_rect()
        if video_rect.isEmpty():
            return
        global_pos = self.video_widget.mapToGlobal(video_rect.topLeft())
        self.region_selection_band.setGeometry(QRect(global_pos, video_rect.size()))
        self.region_selection_band.setProperty("disabled", True)
        self.region_selection_band.setStyleSheet(
            "border: 2px dashed #38bdf8; border-radius: 3px; background: transparent;"
        )
        self.region_selection_band.show()
        self.region_selection_band.raise_()

    def _hide_disabled_frame(self) -> None:
        if self.region_selection_band.property("disabled"):
            self.region_selection_band.setProperty("disabled", False)
            self.region_selection_band.setStyleSheet(
                "border: 2px solid #0ea5e9; border-radius: 3px; background: transparent;"
            )
            if self.region_drag_origin is None and self.region_selection_rect is None:
                self.region_selection_band.hide()

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
            QPushButton#btnToggleRegion {
                background: #162233;
                border-color: #324a66;
                color: #d6e0ec;
            }
            QPushButton#btnToggleRegion:hover {
                background: #1d2d42;
            }
            QPushButton#btnToggleRegion:checked {
                background: #224766;
                border-color: #5a82aa;
                color: #eff6ff;
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
            QLabel#selectionFrame {
                background: transparent;
                border: 2px solid #3b82f6;
                border-radius: 0px;
            }
            QLabel#selectionFrame.disabled {
                border: 2px dashed #6b7280;
                background: transparent;
            }
            """
        )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        super().closeEvent(event)
