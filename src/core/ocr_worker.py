from __future__ import annotations

import os
import re
import traceback
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import hypot
from pathlib import Path
from uuid import uuid4

from PyQt6.QtCore import QThread, pyqtSignal

from core.segmenter import SegmentDraft, TextChangeSegmenter
from core.text_correction import LanguageCorrector
from models.subtitle_segment import SubtitleSegment

CJK_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
REPEATED_CHAR_PATTERN = re.compile(r"^(.)\1{2,}$", re.IGNORECASE)
COMMON_SINGLE_WORD_SUBTITLES = {
    "a",
    "i",
    "ah",
    "aw",
    "bye",
    "come",
    "dad",
    "go",
    "hello",
    "hey",
    "hmm",
    "huh",
    "mom",
    "no",
    "nope",
    "ok",
    "okay",
    "please",
    "run",
    "sorry",
    "stop",
    "thanks",
    "wait",
    "what",
    "why",
    "wow",
    "yeah",
    "yep",
    "yes",
}


@dataclass(slots=True)
class OcrDetection:
    text: str
    confidence: float | None
    bbox: list[tuple[float, float]]
    center: tuple[float, float]


@dataclass(slots=True)
class TrackState:
    track_id: str
    segmenter: TextChangeSegmenter
    center: tuple[float, float]
    last_text: str
    last_bbox: list[tuple[float, float]] | None
    last_confidence: float | None
    last_seen_ms: int
    segment_hits: int = 1
    missed_steps: int = 0


class OcrWorker(QThread):
    progress = pyqtSignal(int)
    segment_found = pyqtSignal(object)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    completed = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(
        self,
        video_path: Path,
        sample_fps: float,
        model_paths: dict[str, Path],
        enable_language_correction: bool = False,
        use_gpu: bool = True,
        crop_region_norm: tuple[float, float, float, float] | None = None,
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.sample_fps = max(sample_fps, 0.2)
        self.model_paths = model_paths
        self.language_code = "en"
        self.enable_language_correction = enable_language_correction
        self.use_gpu = use_gpu
        self.crop_region_norm = self._normalize_crop_region(crop_region_norm)
        self._cancel_requested = False
        self._cv2 = None
        self._tracks: dict[str, TrackState] = {}
        self._track_counter = 0
        self._sample_interval_ms = max(1, int(round(1000.0 / self.sample_fps)))
        self._track_ttl_steps = 1
        self._frame_width = 0.0
        self._frame_height = 0.0
        self._recalculate_track_ttl()

    def cancel(self) -> None:
        self._cancel_requested = True

    def _recalculate_track_ttl(self) -> None:
        grace_window_ms = 450
        self._track_ttl_steps = max(1, int(round(grace_window_ms / max(1, self._sample_interval_ms))))

    def _update_frame_context(self, frame: object) -> None:
        shape = getattr(frame, "shape", None)
        if not shape or len(shape) < 2:
            return
        self._frame_height = float(shape[0] or 0.0)
        self._frame_width = float(shape[1] or 0.0)

    def _estimate_close_timestamp(self, current_timestamp_ms: int, last_seen_ms: int) -> int:
        trailing_padding = max(80, int(self._sample_interval_ms * 0.55))
        return min(current_timestamp_ms, last_seen_ms + trailing_padding)

    def run(self) -> None:
        self.status.emit("Inicializando OCR...")
        os.environ["PADDLE_HOME"] = str(self.model_paths["paddle_home"])
        if "paddlex_cache_home" in self.model_paths:
            os.environ["PADDLE_PDX_CACHE_HOME"] = str(self.model_paths["paddlex_cache_home"])
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        gpu_available = self._resolve_gpu_flag()
        effective_gpu = self.use_gpu and gpu_available
        if self.use_gpu and not effective_gpu:
            self.status.emit("CUDA no disponible. OCR continuará en CPU.")

        try:
            import cv2
        except Exception as exc:
            self.error.emit(f"No se pudo cargar OpenCV: {exc}")
            return
        self._cv2 = cv2

        reader = self._create_reader(effective_gpu)
        if reader is None:
            return

        corrector = self._create_corrector()

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            self.error.emit(f"No se pudo abrir el video: {self.video_path}")
            return

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0

        frame_step = max(int(round(fps / self.sample_fps)), 1)
        self._sample_interval_ms = max(1, int((frame_step / fps) * 1000))
        self._recalculate_track_ttl()
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        last_timestamp_ms = 0
        frame_index = 0
        self._tracks.clear()
        self._track_counter = 0

        if effective_gpu:
            self.status.emit("Procesando video con PaddleOCR en GPU...")
        else:
            self.status.emit("Procesando video con PaddleOCR en CPU...")
        if self.crop_region_norm is not None:
            self.status.emit("Recorte OCR activo. Se procesará solo el área seleccionada.")

        try:
            while not self._cancel_requested:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index % frame_step != 0:
                    frame_index += 1
                    continue

                timestamp_ms = int((frame_index / fps) * 1000)
                last_timestamp_ms = timestamp_ms

                frame_diagonal = self._frame_diagonal(frame)
                self._update_frame_context(frame)
                frame_for_ocr, offset_x, offset_y = self._build_crop_frame(frame)
                detections = self._extract_detections(reader, frame_for_ocr)
                if detections and (offset_x != 0 or offset_y != 0):
                    detections = self._translate_detections(detections, offset_x, offset_y)
                if corrector and detections:
                    detections = self._apply_correction(corrector, detections)
                for closed_segment in self._update_tracks(timestamp_ms, detections, frame_diagonal):
                    self.segment_found.emit(closed_segment)

                if total_frames > 0:
                    progress = min(100, int((frame_index / total_frames) * 100))
                    self.progress.emit(progress)

                frame_index += 1

            flush_end = last_timestamp_ms + max(80, int(self._sample_interval_ms * 0.55))
            for closed_segment in self._flush_tracks(flush_end):
                self.segment_found.emit(closed_segment)

            if self._cancel_requested:
                self.status.emit("OCR cancelado.")
                self.cancelled.emit()
                return

            self.progress.emit(100)
            self.status.emit("OCR finalizado.")
            self.completed.emit()
        except Exception as exc:
            error_text = f"Ocurrió un error durante OCR: {exc}\n{traceback.format_exc()}"
            self.error.emit(error_text)
        finally:
            capture.release()

    def _build_segment(
        self,
        draft: SegmentDraft,
        bbox: list[tuple[float, float]] | None = None,
        confidence: float | None = None,
    ) -> SubtitleSegment:
        clean_bbox = None
        if bbox:
            clean_bbox = [(float(x), float(y)) for x, y in bbox]
        return SubtitleSegment(
            id=uuid4().hex,
            start_ms=draft.start_ms,
            end_ms=draft.end_ms,
            text=draft.text,
            source="ocr",
            confidence=confidence,
            bbox=clean_bbox,
        )

    def _update_tracks(
        self,
        timestamp_ms: int,
        detections: list[OcrDetection],
        frame_diagonal: float,
    ) -> list[SubtitleSegment]:
        emitted: list[SubtitleSegment] = []
        assignments = self._assign_detections_to_tracks(detections, frame_diagonal)
        matched_track_ids: set[str] = set()
        matched_detection_indexes: set[int] = set()

        for detection_index, track_id in assignments.items():
            track = self._tracks.get(track_id)
            if track is None:
                continue
            detection = detections[detection_index]
            matched_track_ids.add(track_id)
            matched_detection_indexes.add(detection_index)
            track.center = detection.center
            track.last_text = detection.text
            track.last_bbox = detection.bbox
            track.last_confidence = detection.confidence
            track.last_seen_ms = timestamp_ms
            track.missed_steps = 0
            closed_drafts = track.segmenter.push(timestamp_ms, detection.text)
            if closed_drafts:
                previous_hits = track.segment_hits
                for closed in closed_drafts:
                    segment = self._build_segment(closed, track.last_bbox, track.last_confidence)
                    if self._should_emit_segment(segment, previous_hits):
                        emitted.append(segment)
                track.segment_hits = 1
            else:
                track.segment_hits += 1

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indexes:
                continue
            track = self._start_track(detection, timestamp_ms)
            matched_track_ids.add(track.track_id)

        stale_track_ids: list[str] = []
        for track_id, track in list(self._tracks.items()):
            if track_id in matched_track_ids:
                continue
            track.missed_steps += 1
            if track.missed_steps > self._track_ttl_steps:
                close_timestamp = self._estimate_close_timestamp(timestamp_ms, track.last_seen_ms)
                for closed in track.segmenter.flush(close_timestamp):
                    segment = self._build_segment(closed, track.last_bbox, track.last_confidence)
                    if self._should_emit_segment(segment, track.segment_hits):
                        emitted.append(segment)
                stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            self._tracks.pop(track_id, None)
        return emitted

    def _flush_tracks(self, end_timestamp_ms: int) -> list[SubtitleSegment]:
        emitted: list[SubtitleSegment] = []
        for track in list(self._tracks.values()):
            for closed in track.segmenter.flush(end_timestamp_ms):
                segment = self._build_segment(closed, track.last_bbox, track.last_confidence)
                if self._should_emit_segment(segment, track.segment_hits):
                    emitted.append(segment)
        self._tracks.clear()
        return emitted

    def _start_track(self, detection: OcrDetection, timestamp_ms: int) -> TrackState:
        track_id = f"track-{self._track_counter}"
        self._track_counter += 1
        track = TrackState(
            track_id=track_id,
            segmenter=TextChangeSegmenter(min_duration_ms=200, similarity_threshold=0.84),
            center=detection.center,
            last_text=detection.text,
            last_bbox=detection.bbox,
            last_confidence=detection.confidence,
            last_seen_ms=timestamp_ms,
            segment_hits=1,
            missed_steps=0,
        )
        track.segmenter.push(timestamp_ms, detection.text)
        self._tracks[track_id] = track
        return track

    def _assign_detections_to_tracks(
        self,
        detections: list[OcrDetection],
        frame_diagonal: float,
    ) -> dict[int, str]:
        if not self._tracks or not detections:
            return {}

        candidate_pairs: list[tuple[float, int, str]] = []
        for detection_index, detection in enumerate(detections):
            for track_id, track in self._tracks.items():
                match_score = self._track_match_score(detection, track, frame_diagonal)
                if match_score <= 0:
                    continue
                candidate_pairs.append((match_score, detection_index, track_id))

        candidate_pairs.sort(reverse=True, key=lambda item: item[0])

        assignments: dict[int, str] = {}
        used_detections: set[int] = set()
        used_tracks: set[str] = set()

        for _, detection_index, track_id in candidate_pairs:
            if detection_index in used_detections or track_id in used_tracks:
                continue
            assignments[detection_index] = track_id
            used_detections.add(detection_index)
            used_tracks.add(track_id)
        return assignments

    @staticmethod
    def _track_match_score(detection: OcrDetection, track: TrackState, frame_diagonal: float) -> float:
        max_distance = max(60.0, frame_diagonal * 0.18)
        distance = hypot(detection.center[0] - track.center[0], detection.center[1] - track.center[1])
        if distance > max_distance:
            return -1.0

        spatial_score = 1.0 - (distance / max_distance)
        text_similarity = OcrWorker._text_similarity(detection.text, track.last_text)

        if text_similarity < 0.2 and spatial_score < 0.72:
            return -1.0

        confidence_bonus = (detection.confidence or 0.0) * 0.06
        return (spatial_score * 0.72) + (text_similarity * 0.28) + confidence_bonus

    @staticmethod
    def _extract_paddleocr_detections(reader: object, frame: object) -> list[OcrDetection]:
        if hasattr(reader, "predict"):
            results = reader.predict(frame)
        else:
            results = reader.ocr(frame, cls=True)
        if not results:
            return []

        first_result = results
        if isinstance(results, list) and results:
            first_result = results[0]

        detections: list[OcrDetection] = []
        if isinstance(first_result, dict):
            rec_texts = first_result.get("rec_texts", []) or []
            rec_scores = first_result.get("rec_scores", []) or []
            rec_polys = first_result.get("rec_polys") or first_result.get("dt_polys") or []
            for index, raw_text in enumerate(rec_texts):
                text = str(raw_text).strip()
                confidence = OcrWorker._safe_float(rec_scores[index] if index < len(rec_scores) else None)
                bbox = OcrWorker._normalize_bbox(rec_polys[index] if index < len(rec_polys) else None)
                center = OcrWorker._bbox_center(bbox, index)
                detections.append(
                    OcrDetection(
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                        center=center,
                    )
                )
            return detections

        # Compatibilidad con formatos antiguos de PaddleOCR.
        rows = first_result if isinstance(first_result, list) else []
        detections = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            bbox = OcrWorker._normalize_bbox(row[0])
            text_info = row[1]
            if not isinstance(text_info, (list, tuple)) or len(text_info) < 1:
                continue
            text = str(text_info[0]).strip()
            confidence = OcrWorker._safe_float(text_info[1] if len(text_info) >= 2 else None)
            center = OcrWorker._bbox_center(bbox, len(detections))
            detections.append(
                OcrDetection(
                    text=text,
                    confidence=confidence,
                    bbox=bbox,
                    center=center,
                )
            )
        return detections

    def _extract_detections(self, reader: object, frame: object) -> list[OcrDetection]:
        frame_diagonal = self._frame_diagonal(frame)
        base_detections = self._extract_paddleocr_detections(reader, frame)
        filtered_base = self._filter_detections(base_detections)
        if filtered_base:
            deduped = self._deduplicate_detections(filtered_base, frame_diagonal)
            merged_lines = self._merge_line_detections(deduped)
            merged_boxes = self._merge_box_detections(merged_lines)
            return self._merge_bubble_detections(merged_boxes)

        for variant in self._build_full_frame_variants(frame):
            variant_detections = self._extract_paddleocr_detections(reader, variant)
            filtered_variant = self._filter_detections(variant_detections)
            if filtered_variant:
                deduped = self._deduplicate_detections(filtered_variant, frame_diagonal)
                merged_lines = self._merge_line_detections(deduped)
                merged_boxes = self._merge_box_detections(merged_lines)
                return self._merge_bubble_detections(merged_boxes)

        return []

    def _build_crop_frame(self, frame: object) -> tuple[object, int, int]:
        if self.crop_region_norm is None:
            return frame, 0, 0
        shape = getattr(frame, "shape", None)
        if not shape or len(shape) < 2:
            return frame, 0, 0

        frame_height = int(shape[0] or 0)
        frame_width = int(shape[1] or 0)
        if frame_width <= 2 or frame_height <= 2:
            return frame, 0, 0

        x_norm, y_norm, width_norm, height_norm = self.crop_region_norm
        x_start = int(round(x_norm * frame_width))
        y_start = int(round(y_norm * frame_height))
        x_end = int(round((x_norm + width_norm) * frame_width))
        y_end = int(round((y_norm + height_norm) * frame_height))

        x_start = max(0, min(frame_width - 1, x_start))
        y_start = max(0, min(frame_height - 1, y_start))
        x_end = max(x_start + 1, min(frame_width, x_end))
        y_end = max(y_start + 1, min(frame_height, y_end))

        if x_end - x_start <= 1 or y_end - y_start <= 1:
            return frame, 0, 0

        try:
            cropped_frame = frame[y_start:y_end, x_start:x_end]
        except Exception:
            return frame, 0, 0
        cropped_size = getattr(cropped_frame, "size", 0)
        if not cropped_size:
            return frame, 0, 0
        return cropped_frame, x_start, y_start

    @staticmethod
    def _translate_detections(
        detections: list[OcrDetection],
        offset_x: int,
        offset_y: int,
    ) -> list[OcrDetection]:
        translated: list[OcrDetection] = []
        for detection in detections:
            translated_bbox = [(x + offset_x, y + offset_y) for x, y in detection.bbox] if detection.bbox else []
            translated.append(
                OcrDetection(
                    text=detection.text,
                    confidence=detection.confidence,
                    bbox=translated_bbox,
                    center=(detection.center[0] + offset_x, detection.center[1] + offset_y),
                )
            )
        return translated

    def _filter_detections(self, detections: list[OcrDetection]) -> list[OcrDetection]:
        filtered: list[OcrDetection] = []
        for detection in detections:
            normalized = self._normalize_english_candidate(detection.text)
            if not normalized:
                continue
            if self._is_noise_text(normalized, detection.confidence):
                continue
            filtered.append(
                OcrDetection(
                    text=normalized,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    center=detection.center,
                )
            )
        return filtered

    def _apply_correction(
        self,
        corrector: LanguageCorrector,
        detections: list[OcrDetection],
    ) -> list[OcrDetection]:
        corrected: list[OcrDetection] = []
        for detection in detections:
            corrected_text = corrector.correct_sentence(detection.text, average_confidence=detection.confidence)
            normalized = self._normalize_english_candidate(corrected_text)
            if not normalized:
                continue
            if self._is_noise_text(normalized, detection.confidence):
                continue
            corrected.append(
                OcrDetection(
                    text=normalized,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    center=detection.center,
                )
            )
        if corrected:
            return corrected
        return detections

    @staticmethod
    def _deduplicate_detections(
        detections: list[OcrDetection],
        frame_diagonal: float,
    ) -> list[OcrDetection]:
        if len(detections) <= 1:
            return detections

        distance_limit = max(22.0, frame_diagonal * 0.02)
        ordered = sorted(
            detections,
            key=lambda item: OcrWorker._text_quality_score(item.text, item.confidence),
            reverse=True,
        )
        compact: list[OcrDetection] = []

        for candidate in ordered:
            duplicate = False
            for accepted in compact:
                distance = hypot(
                    candidate.center[0] - accepted.center[0],
                    candidate.center[1] - accepted.center[1],
                )
                similarity = OcrWorker._text_similarity(candidate.text, accepted.text)
                if similarity >= 0.94 and distance <= distance_limit:
                    duplicate = True
                    break
            if not duplicate:
                compact.append(candidate)

        compact.sort(key=lambda item: (item.center[1], item.center[0]))
        return compact

    @staticmethod
    def _merge_line_detections(detections: list[OcrDetection]) -> list[OcrDetection]:
        if len(detections) <= 1:
            return detections

        ordered = sorted(detections, key=lambda item: (item.center[1], item.center[0]))
        lines: list[list[OcrDetection]] = []

        for detection in ordered:
            y_min, y_max = OcrWorker._y_bounds(detection)
            y_center = (y_min + y_max) / 2.0
            line_match: list[OcrDetection] | None = None
            for line in lines:
                line_y_min = min(OcrWorker._y_bounds(item)[0] for item in line)
                line_y_max = max(OcrWorker._y_bounds(item)[1] for item in line)
                line_height = max(1.0, line_y_max - line_y_min)
                line_center = (line_y_min + line_y_max) / 2.0
                if abs(y_center - line_center) <= max(18.0, line_height * 0.65):
                    line_match = line
                    break
            if line_match is None:
                lines.append([detection])
            else:
                line_match.append(detection)

        merged: list[OcrDetection] = []
        for line in lines:
            line_sorted = sorted(line, key=lambda item: OcrWorker._x_bounds(item)[0])
            active_group: list[OcrDetection] = []
            for detection in line_sorted:
                if not active_group:
                    active_group = [detection]
                    continue

                prev = active_group[-1]
                prev_x_min, prev_x_max = OcrWorker._x_bounds(prev)
                curr_x_min, _ = OcrWorker._x_bounds(detection)
                gap = curr_x_min - prev_x_max

                prev_y_min, prev_y_max = OcrWorker._y_bounds(prev)
                curr_y_min, curr_y_max = OcrWorker._y_bounds(detection)
                overlap = max(0.0, min(prev_y_max, curr_y_max) - max(prev_y_min, curr_y_min))
                min_height = max(1.0, min(prev_y_max - prev_y_min, curr_y_max - curr_y_min))
                overlap_ratio = overlap / min_height
                avg_height = max(1.0, ((prev_y_max - prev_y_min) + (curr_y_max - curr_y_min)) / 2.0)

                if gap <= max(46.0, avg_height * 2.1) and overlap_ratio >= 0.33:
                    active_group.append(detection)
                else:
                    merged.append(OcrWorker._build_group_detection(active_group))
                    active_group = [detection]

            if active_group:
                merged.append(OcrWorker._build_group_detection(active_group))

        merged.sort(key=lambda item: (item.center[1], item.center[0]))
        return merged

    @staticmethod
    def _merge_box_detections(detections: list[OcrDetection]) -> list[OcrDetection]:
        if len(detections) <= 1:
            return detections

        groups: list[list[OcrDetection]] = []
        pending = set(range(len(detections)))
        while pending:
            current_index = pending.pop()
            queue = [current_index]
            group_indexes = [current_index]
            while queue:
                source_index = queue.pop()
                source = detections[source_index]
                linked_indexes = []
                for candidate_index in list(pending):
                    candidate = detections[candidate_index]
                    if OcrWorker._should_merge_as_box(source, candidate):
                        linked_indexes.append(candidate_index)
                for linked in linked_indexes:
                    pending.remove(linked)
                    queue.append(linked)
                    group_indexes.append(linked)
            groups.append([detections[index] for index in group_indexes])

        merged: list[OcrDetection] = []
        for group in groups:
            if len(group) == 1:
                merged.append(group[0])
                continue
            ordered_group = sorted(group, key=lambda item: (item.center[1], item.center[0]))
            merged.append(OcrWorker._build_group_detection(ordered_group))

        merged.sort(key=lambda item: (item.center[1], item.center[0]))
        return merged

    @staticmethod
    def _should_merge_as_box(left: OcrDetection, right: OcrDetection) -> bool:
        left_x_min, left_x_max = OcrWorker._x_bounds(left)
        right_x_min, right_x_max = OcrWorker._x_bounds(right)
        left_y_min, left_y_max = OcrWorker._y_bounds(left)
        right_y_min, right_y_max = OcrWorker._y_bounds(right)

        left_width = max(1.0, left_x_max - left_x_min)
        right_width = max(1.0, right_x_max - right_x_min)
        left_height = max(1.0, left_y_max - left_y_min)
        right_height = max(1.0, right_y_max - right_y_min)

        x_overlap = max(0.0, min(left_x_max, right_x_max) - max(left_x_min, right_x_min))
        x_overlap_ratio = x_overlap / min(left_width, right_width)
        if x_overlap_ratio < 0.52:
            return False

        width_ratio = max(left_width, right_width) / min(left_width, right_width)
        if width_ratio > 2.4:
            return False

        left_center_x = (left_x_min + left_x_max) / 2.0
        right_center_x = (right_x_min + right_x_max) / 2.0
        if abs(left_center_x - right_center_x) > max(34.0, min(left_width, right_width) * 0.35):
            return False

        vertical_gap = max(0.0, max(left_y_min, right_y_min) - min(left_y_max, right_y_max))
        avg_height = (left_height + right_height) / 2.0
        if vertical_gap > max(30.0, avg_height * 1.8):
            return False

        return True

    @staticmethod
    def _merge_bubble_detections(detections: list[OcrDetection]) -> list[OcrDetection]:
        if len(detections) <= 1:
            return detections

        groups: list[list[OcrDetection]] = []
        pending = set(range(len(detections)))
        while pending:
            current_index = pending.pop()
            queue = [current_index]
            group_indexes = [current_index]
            while queue:
                source_index = queue.pop()
                source = detections[source_index]
                linked_indexes = []
                for candidate_index in list(pending):
                    candidate = detections[candidate_index]
                    if OcrWorker._should_merge_as_bubble(source, candidate):
                        linked_indexes.append(candidate_index)
                for linked in linked_indexes:
                    pending.remove(linked)
                    queue.append(linked)
                    group_indexes.append(linked)
            groups.append([detections[index] for index in group_indexes])

        merged: list[OcrDetection] = []
        for group in groups:
            if len(group) == 1:
                merged.append(group[0])
                continue
            ordered_group = sorted(group, key=lambda item: (item.center[1], item.center[0]))
            merged.append(OcrWorker._build_group_detection(ordered_group))

        merged.sort(key=lambda item: (item.center[1], item.center[0]))
        return merged

    @staticmethod
    def _should_merge_as_bubble(left: OcrDetection, right: OcrDetection) -> bool:
        left_x_min, left_x_max = OcrWorker._x_bounds(left)
        right_x_min, right_x_max = OcrWorker._x_bounds(right)
        left_y_min, left_y_max = OcrWorker._y_bounds(left)
        right_y_min, right_y_max = OcrWorker._y_bounds(right)

        left_width = max(1.0, left_x_max - left_x_min)
        right_width = max(1.0, right_x_max - right_x_min)
        left_height = max(1.0, left_y_max - left_y_min)
        right_height = max(1.0, right_y_max - right_y_min)

        x_overlap = max(0.0, min(left_x_max, right_x_max) - max(left_x_min, right_x_min))
        y_overlap = max(0.0, min(left_y_max, right_y_max) - max(left_y_min, right_y_min))
        x_overlap_ratio = x_overlap / min(left_width, right_width)
        y_overlap_ratio = y_overlap / min(left_height, right_height)

        vertical_gap = max(0.0, max(left_y_min, right_y_min) - min(left_y_max, right_y_max))
        horizontal_gap = max(0.0, max(left_x_min, right_x_min) - min(left_x_max, right_x_max))
        avg_height = (left_height + right_height) / 2.0

        if x_overlap_ratio >= 0.4 and vertical_gap <= max(34.0, avg_height * 2.2):
            return True
        if x_overlap_ratio >= 0.62 and vertical_gap <= max(24.0, avg_height * 1.8):
            return True

        # Caso límite para globos inclinados o curvas de texto.
        if y_overlap_ratio >= 0.45 and horizontal_gap <= max(60.0, min(left_width, right_width) * 0.42):
            center_distance = hypot(left.center[0] - right.center[0], left.center[1] - right.center[1])
            if center_distance <= max(left_width, right_width) * 1.2:
                return True

        return False

    @staticmethod
    def _build_group_detection(group: list[OcrDetection]) -> OcrDetection:
        if len(group) == 1:
            return group[0]

        x_min = min(OcrWorker._x_bounds(item)[0] for item in group)
        x_max = max(OcrWorker._x_bounds(item)[1] for item in group)
        y_min = min(OcrWorker._y_bounds(item)[0] for item in group)
        y_max = max(OcrWorker._y_bounds(item)[1] for item in group)

        fragments = [item.text for item in group if item.text.strip()]
        merged_text = OcrWorker._join_text_fragments(fragments)
        confidences = [item.confidence for item in group if item.confidence is not None]
        merged_confidence = (sum(confidences) / len(confidences)) if confidences else None
        merged_bbox = [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        ]
        center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
        return OcrDetection(
            text=merged_text,
            confidence=merged_confidence,
            bbox=merged_bbox,
            center=center,
        )

    @staticmethod
    def _join_text_fragments(fragments: list[str]) -> str:
        if not fragments:
            return ""
        text = fragments[0].strip()
        if not text:
            text = ""
        for fragment in fragments[1:]:
            chunk = fragment.strip()
            if not chunk:
                continue
            if chunk[0] in ".,!?;:%)]}":
                text = f"{text}{chunk}"
            elif text and text[-1] in "([{":
                text = f"{text}{chunk}"
            else:
                text = f"{text} {chunk}" if text else chunk
        return " ".join(text.split())

    @staticmethod
    def _x_bounds(detection: OcrDetection) -> tuple[float, float]:
        if detection.bbox:
            xs = [point[0] for point in detection.bbox]
            return min(xs), max(xs)
        x_center = detection.center[0]
        return x_center - 25.0, x_center + 25.0

    @staticmethod
    def _y_bounds(detection: OcrDetection) -> tuple[float, float]:
        if detection.bbox:
            ys = [point[1] for point in detection.bbox]
            return min(ys), max(ys)
        y_center = detection.center[1]
        return y_center - 10.0, y_center + 10.0

    def _build_full_frame_variants(self, frame: object) -> list[object]:
        cv2 = self._cv2
        if cv2 is None:
            return []

        try:
            sharpened = cv2.addWeighted(
                frame,
                1.35,
                cv2.GaussianBlur(frame, (0, 0), 1.0),
                -0.35,
                0,
            )
            return [sharpened]
        except Exception:
            return []

    @staticmethod
    def _normalize_english_candidate(text: str) -> str:
        normalized = " ".join(text.split())
        if not normalized:
            return ""

        has_cjk = bool(CJK_PATTERN.search(normalized))
        has_latin = bool(LATIN_PATTERN.search(normalized))
        if has_cjk and not has_latin:
            return ""
        normalized = normalized.strip("-_/:|")
        if not normalized:
            return ""
        return normalized

    @staticmethod
    def _is_noise_text(text: str, confidence: float | None) -> bool:
        normalized = " ".join(text.split())
        if len(normalized) < 2:
            return True

        words = WORD_PATTERN.findall(normalized)
        if not words:
            return True

        letters = sum(1 for char in normalized if char.isalpha())
        digits = sum(1 for char in normalized if char.isdigit())
        symbols = sum(
            1
            for char in normalized
            if not (char.isalnum() or char.isspace() or char in ".,!?':;-\"()[]")
        )

        if letters == 0:
            return True
        if confidence is not None and confidence < 0.45 and len(words) <= 1:
            return True
        if digits > letters and len(words) <= 2:
            return True
        visible_len = max(1, len(normalized.replace(" ", "")))
        if (symbols / visible_len) > 0.28:
            return True
        if OcrWorker._text_quality_score(normalized, confidence) < 3.0:
            return True
        return False

    def _should_emit_segment(self, segment: SubtitleSegment, segment_hits: int) -> bool:
        text = " ".join(segment.text.split())
        words = WORD_PATTERN.findall(text)
        word_count = len(words)
        duration_ms = max(0, segment.end_ms - segment.start_ms)

        if self._is_likely_scene_text(text, words, duration_ms, segment_hits, segment.bbox):
            return False

        if word_count >= 2:
            return True
        if word_count == 0:
            return False

        token = words[0]
        token_lower = token.lower()
        if token_lower in COMMON_SINGLE_WORD_SUBTITLES:
            return True
        if duration_ms >= (self._sample_interval_ms * 2) and segment_hits >= 2:
            return not self._is_suspicious_single_word(token)
        if segment_hits >= 3:
            return not self._is_suspicious_single_word(token)
        return False

    def _is_likely_scene_text(
        self,
        text: str,
        words: list[str],
        duration_ms: int,
        segment_hits: int,
        bbox: list[tuple[float, float]] | None,
    ) -> bool:
        if not words:
            return True

        has_terminal_punctuation = bool(re.search(r"[.!?]['\"]?$", text))
        if len(words) <= 2 and segment_hits >= 3:
            if duration_ms >= max(2800, self._sample_interval_ms * 4) and not has_terminal_punctuation:
                return True

        if len(words) <= 3 and segment_hits >= 3 and not has_terminal_punctuation:
            title_or_upper_words = 0
            for word in words:
                if word.isupper() or word.istitle():
                    title_or_upper_words += 1
            if title_or_upper_words == len(words) and duration_ms >= max(1600, self._sample_interval_ms * 2):
                return True

        if not bbox:
            return False
        if self._frame_width <= 0 or self._frame_height <= 0:
            return False

        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        if not xs or not ys:
            return False
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        box_width = max(1.0, x_max - x_min)
        box_height = max(1.0, y_max - y_min)
        area_ratio = (box_width * box_height) / max(1.0, self._frame_width * self._frame_height)
        if area_ratio < 0.00045 and len(words) <= 2 and not has_terminal_punctuation:
            return True

        return False

    @staticmethod
    def _is_suspicious_single_word(token: str) -> bool:
        clean = token.strip()
        if not clean:
            return True
        if len(clean) <= 1:
            return True
        if REPEATED_CHAR_PATTERN.match(clean):
            return True
        letters_only = re.sub(r"[^A-Za-z]", "", clean)
        if not letters_only:
            return True
        if len(letters_only) >= 3 and letters_only.isupper():
            return True
        vowels = sum(1 for char in letters_only.lower() if char in "aeiouy")
        if len(letters_only) >= 4 and vowels == 0:
            return True
        return False

    @staticmethod
    def _text_quality_score(text: str, confidence: float | None) -> float:
        normalized = " ".join(text.split())
        if not normalized:
            return float("-inf")

        has_cjk = bool(CJK_PATTERN.search(normalized))
        has_latin = bool(LATIN_PATTERN.search(normalized))
        if has_cjk and not has_latin:
            return float("-inf")

        letters = sum(1 for char in normalized if char.isalpha())
        words = WORD_PATTERN.findall(normalized)
        digits = sum(1 for char in normalized if char.isdigit())
        symbols = sum(
            1
            for char in normalized
            if not (char.isalnum() or char.isspace() or char in ".,!?':;-\"()")
        )
        long_words = sum(1 for word in words if len(word) >= 4)

        confidence_score = (confidence or 0.0) * 6.5
        return (
            confidence_score
            + (letters * 0.045)
            + (len(words) * 0.85)
            + (long_words * 0.35)
            - (symbols * 1.75)
            - (max(0, digits - 4) * 0.2)
        )

    @staticmethod
    def _text_similarity(left: str, right: str) -> float:
        return SequenceMatcher(a=left.lower(), b=right.lower()).ratio()

    @staticmethod
    def _frame_diagonal(frame: object) -> float:
        shape = getattr(frame, "shape", None)
        if not shape or len(shape) < 2:
            return 1920.0
        height = float(shape[0] or 0.0)
        width = float(shape[1] or 0.0)
        if height <= 0 or width <= 0:
            return 1920.0
        return (height * height + width * width) ** 0.5

    @staticmethod
    def _normalize_crop_region(
        crop_region_norm: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, float, float] | None:
        if crop_region_norm is None:
            return None
        if len(crop_region_norm) != 4:
            return None
        try:
            x_norm = float(crop_region_norm[0])
            y_norm = float(crop_region_norm[1])
            width_norm = float(crop_region_norm[2])
            height_norm = float(crop_region_norm[3])
        except Exception:
            return None
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        width_norm = max(0.0, min(1.0 - x_norm, width_norm))
        height_norm = max(0.0, min(1.0 - y_norm, height_norm))
        if width_norm <= 0.01 or height_norm <= 0.01:
            return None
        return x_norm, y_norm, width_norm, height_norm

    @staticmethod
    def _safe_float(value: object) -> float | None:
        try:
            number = float(value)  # type: ignore[arg-type]
        except Exception:
            return None
        if number != number:  # NaN
            return None
        return number

    @staticmethod
    def _normalize_bbox(raw_bbox: object) -> list[tuple[float, float]]:
        if raw_bbox is None:
            return []
        try:
            points = list(raw_bbox)  # type: ignore[arg-type]
        except Exception:
            return []

        normalized: list[tuple[float, float]] = []
        for point in points:
            try:
                x_value = float(point[0])  # type: ignore[index]
                y_value = float(point[1])  # type: ignore[index]
            except Exception:
                continue
            normalized.append((x_value, y_value))
        return normalized

    @staticmethod
    def _bbox_center(bbox: list[tuple[float, float]], fallback_index: int) -> tuple[float, float]:
        if not bbox:
            return 0.0, float((fallback_index + 1) * 40)
        x_sum = sum(point[0] for point in bbox)
        y_sum = sum(point[1] for point in bbox)
        size = max(1, len(bbox))
        return x_sum / size, y_sum / size

    def _create_reader(self, effective_gpu: bool) -> object | None:
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:
            self.error.emit(
                "No se encontró PaddleOCR. Instálalo con: pip install -r requirements.txt "
                f"(detalle: {exc})"
            )
            return None

        try:
            device = "gpu:0" if effective_gpu else "cpu"
            return PaddleOCR(
                device=device,
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name="en_PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=True,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                text_rec_score_thresh=0.6,
            )
        except Exception as exc:
            if "Unknown argument: device" not in str(exc):
                self.error.emit(f"No se pudo inicializar PaddleOCR: {exc}")
                return None

        try:
            return PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=effective_gpu,
                show_log=False,
            )
        except Exception as exc:
            self.error.emit(f"No se pudo inicializar PaddleOCR: {exc}")
            return None

    def _create_corrector(self) -> LanguageCorrector | None:
        if not self.enable_language_correction:
            return None
        try:
            corrector = LanguageCorrector(self.language_code)
            self.status.emit("Corrección por diccionario activada.")
            return corrector
        except Exception as exc:
            self.status.emit(f"No se pudo activar corrección de idioma: {exc}")
            return None

    @staticmethod
    def _resolve_gpu_flag() -> bool:
        try:
            import paddle
        except Exception:
            return False
        try:
            if not paddle.is_compiled_with_cuda():
                return False
            return paddle.device.cuda.device_count() > 0
        except Exception:
            return False
