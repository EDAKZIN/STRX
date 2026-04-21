from __future__ import annotations

from dataclasses import replace

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import QGraphicsLineItem, QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView

from models.subtitle_segment import SubtitleSegment


class TimelineSegmentItem(QGraphicsRectItem):
    EDGE_MARGIN_PX = 8
    MIN_DURATION_MS = 120

    def __init__(self, timeline: "TimelineWidget", segment: SubtitleSegment, lane_index: int) -> None:
        super().__init__()
        self.timeline = timeline
        self.segment = segment
        self.lane_index = lane_index
        self.mode: str | None = None
        self.origin_scene_x = 0.0
        self.origin_start_ms = 0
        self.origin_end_ms = 0

        self.label_item = QGraphicsSimpleTextItem(self)
        self.label_item.setBrush(QBrush(QColor("#dce6f2")))
        self.label_item.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations, False)

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
        self._normal_pen = QPen(QColor("#5e7590"), 1.2)
        self._normal_brush = QBrush(QColor("#36495f"))
        self._selected_pen = QPen(QColor("#8ea4bc"), 2.0)
        self._selected_brush = QBrush(QColor("#465f7a"))
        self.setPen(self._normal_pen)
        self.setBrush(self._normal_brush)
        self.setZValue(5)
        self.update_geometry(segment.start_ms, segment.end_ms, segment.text)

    def update_geometry(self, start_ms: int, end_ms: int, text: str | None = None) -> None:
        x = self.timeline.ms_to_x(start_ms)
        width = max(12.0, self.timeline.ms_to_x(end_ms) - x)
        y = self.timeline.lane_to_y(self.lane_index)
        height = self.timeline.lane_height
        self.setRect(x, y, width, height)

        segment_text = text if text is not None else self.segment.text
        short_text = segment_text.strip()
        if len(short_text) > 28:
            short_text = f"{short_text[:28]}..."
        self.label_item.setText(short_text or "(vacío)")
        self.label_item.setPos(x + 6, y + 6)

    def apply_selected_style(self, selected: bool) -> None:
        if selected:
            self.setPen(self._selected_pen)
            self.setBrush(self._selected_brush)
            self.setZValue(9)
        else:
            self.setPen(self._normal_pen)
            self.setBrush(self._normal_brush)
            self.setZValue(5)

    def hoverMoveEvent(self, event) -> None:  # type: ignore[override]
        mode = self._detect_mode(event.pos().x())
        if mode == "trim_left" or mode == "trim_right":
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.timeline.select_segment(self.segment.id, emit_signal=True)
            self.mode = self._detect_mode(event.pos().x())
            self.origin_scene_x = event.scenePos().x()
            self.origin_start_ms = self.segment.start_ms
            self.origin_end_ms = self.segment.end_ms
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if not self.mode:
            super().mouseMoveEvent(event)
            return

        delta_x = event.scenePos().x() - self.origin_scene_x
        delta_ms = int((delta_x / self.timeline.px_per_second) * 1000)
        start_ms = self.origin_start_ms
        end_ms = self.origin_end_ms

        if self.mode == "move":
            duration_ms = self.origin_end_ms - self.origin_start_ms
            start_ms = max(0, self.origin_start_ms + delta_ms)
            end_ms = start_ms + duration_ms
        elif self.mode == "trim_left":
            limit = self.origin_end_ms - self.MIN_DURATION_MS
            start_ms = min(max(0, self.origin_start_ms + delta_ms), limit)
        elif self.mode == "trim_right":
            limit = self.origin_start_ms + self.MIN_DURATION_MS
            end_ms = max(limit, self.origin_end_ms + delta_ms)

        self.timeline.preview_segment_change(self.segment.id, start_ms, end_ms, self)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self.mode:
            self.timeline.commit_segment_change(self.segment.id, self.segment.start_ms, self.segment.end_ms)
            self.mode = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _detect_mode(self, local_x: float) -> str:
        rect = self.rect()
        if abs(local_x - rect.left()) <= self.EDGE_MARGIN_PX:
            return "trim_left"
        if abs(local_x - rect.right()) <= self.EDGE_MARGIN_PX:
            return "trim_right"
        return "move"


class TimelineWidget(QGraphicsView):
    segment_selected = pyqtSignal(str)
    segment_changing = pyqtSignal(str, int, int)
    segment_changed = pyqtSignal(str, int, int)
    seek_requested = pyqtSignal(int)
    scrub_started = pyqtSignal()
    scrub_ended = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.px_per_second = 120.0
        self.lane_height = 34.0
        self.lane_gap = 12.0
        self.top_padding = 12.0
        self.bottom_padding = 20.0
        self.selected_segment_id: str | None = None
        self.duration_ms = 0
        self.playhead_ms = 0

        self._segments: dict[str, SubtitleSegment] = {}
        self._items: dict[str, TimelineSegmentItem] = {}
        self._playhead_item: QGraphicsLineItem | None = None

        self.setRenderHints(self.renderHints())
        self.setBackgroundBrush(QBrush(QColor("#0f141d")))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def set_segments(self, segments: list[SubtitleSegment], duration_ms: int) -> None:
        self.duration_ms = max(duration_ms, max((segment.end_ms for segment in segments), default=0))
        self._segments = {segment.id: replace(segment) for segment in segments}
        self._rebuild()

    def select_segment(self, segment_id: str, emit_signal: bool = False) -> None:
        self.selected_segment_id = segment_id
        for item_id, item in self._items.items():
            item.setSelected(item_id == segment_id)
            item.apply_selected_style(item_id == segment_id)
        if segment_id in self._items:
            self.centerOn(self._items[segment_id])
        if emit_signal:
            self.segment_selected.emit(segment_id)

    def preview_segment_change(
        self,
        segment_id: str,
        start_ms: int,
        end_ms: int,
        item: TimelineSegmentItem,
    ) -> None:
        segment = self._segments.get(segment_id)
        if not segment:
            return
        segment.start_ms = start_ms
        segment.end_ms = end_ms
        item.segment.start_ms = start_ms
        item.segment.end_ms = end_ms
        item.update_geometry(start_ms, end_ms)
        self.segment_changing.emit(segment_id, start_ms, end_ms)
        self._update_playhead()

    def commit_segment_change(self, segment_id: str, start_ms: int, end_ms: int) -> None:
        self.segment_changed.emit(segment_id, start_ms, end_ms)
        self._rebuild()
        if self.selected_segment_id:
            self.select_segment(self.selected_segment_id, emit_signal=False)

    def update_segment_from_table(self, segment_id: str, start_ms: int, end_ms: int, text: str) -> None:
        segment = self._segments.get(segment_id)
        if not segment:
            return
        segment.start_ms = start_ms
        segment.end_ms = end_ms
        segment.text = text
        self._rebuild()
        if self.selected_segment_id:
            self.select_segment(self.selected_segment_id, emit_signal=False)

    def set_playhead(self, position_ms: int) -> None:
        self.playhead_ms = max(0, position_ms)
        self._update_playhead()
        self._follow_playhead()

    def ms_to_x(self, value_ms: int) -> float:
        return (value_ms / 1000.0) * self.px_per_second

    def lane_to_y(self, lane_index: int) -> float:
        return self.top_padding + (lane_index * (self.lane_height + self.lane_gap))

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.pos())
            # Si no hicimos clic en un segmento (o su texto), iniciamos el scrubbing.
            if not isinstance(item, TimelineSegmentItem) and not getattr(item, "parentItem", lambda: None)().__class__ is TimelineSegmentItem:
                self._is_scrubbing = True
                self.scrub_started.emit()
                scene_pos = self.mapToScene(event.pos())
                if scene_pos.x() >= 0:
                    ms = int((scene_pos.x() / self.px_per_second) * 1000)
                    self.seek_requested.emit(ms)
                # No llamamos a super() aquí para evitar que QGraphicsView inicie el ScrollHandDrag y robe el drag.
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if getattr(self, "_is_scrubbing", False):
            scene_pos = self.mapToScene(event.pos())
            if scene_pos.x() >= 0:
                ms = int((scene_pos.x() / self.px_per_second) * 1000)
                self.seek_requested.emit(ms)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if getattr(self, "_is_scrubbing", False):
            if event.button() == Qt.MouseButton.LeftButton:
                self._is_scrubbing = False
                self.scrub_ended.emit()
            return
        super().mouseReleaseEvent(event)

    def _rebuild(self) -> None:
        self.scene.clear()
        self._items.clear()
        self._playhead_item = None

        sorted_segments = sorted(self._segments.values(), key=lambda item: (item.start_ms, item.end_ms, item.id))
        lane_map, lanes_count = self._assign_lanes(sorted_segments)
        width = max(800.0, self.ms_to_x(self.duration_ms + 1000))
        height = self.top_padding + self.bottom_padding + lanes_count * (self.lane_height + self.lane_gap)
        self.scene.setSceneRect(0, 0, width, height)

        self._draw_grid(width, height)

        for segment in sorted_segments:
            lane_index = lane_map[segment.id]
            item = TimelineSegmentItem(self, segment, lane_index)
            self.scene.addItem(item)
            self._items[segment.id] = item

        self._playhead_item = self.scene.addLine(0, 0, 0, height, QPen(QColor("#82a8bf"), 1.6))
        self._playhead_item.setZValue(20)
        self._update_playhead()

    def _draw_grid(self, width: float, height: float) -> None:
        max_seconds = int(width // self.px_per_second) + 1
        for second in range(max_seconds + 1):
            x = self.ms_to_x(second * 1000)
            color = QColor("#2d3d52") if second % 5 == 0 else QColor("#1f2836")
            self.scene.addLine(x, 0, x, height, QPen(color, 1.0))
            if second % 5 == 0:
                label = self.scene.addSimpleText(f"{second}s")
                label.setBrush(QBrush(QColor("#98a9bf")))
                label.setPos(x + 2, 0)

    def _assign_lanes(self, segments: list[SubtitleSegment]) -> tuple[dict[str, int], int]:
        lane_ends: list[int] = []
        lane_map: dict[str, int] = {}

        for segment in segments:
            assigned_lane: int | None = None
            for lane_index, lane_end in enumerate(lane_ends):
                if segment.start_ms >= lane_end:
                    assigned_lane = lane_index
                    lane_ends[lane_index] = segment.end_ms
                    break
            if assigned_lane is None:
                assigned_lane = len(lane_ends)
                lane_ends.append(segment.end_ms)
            lane_map[segment.id] = assigned_lane

        lanes_count = max(1, len(lane_ends))
        return lane_map, lanes_count

    def _update_playhead(self) -> None:
        if not self._playhead_item:
            return
        x = self.ms_to_x(self.playhead_ms)
        rect = self.scene.sceneRect()
        self._playhead_item.setLine(x, 0, x, rect.height())

    def _follow_playhead(self) -> None:
        if not self._playhead_item:
            return
        view_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        if view_rect.width() <= 0:
            return

        x = self.ms_to_x(self.playhead_ms)
        left_limit = view_rect.left() + (view_rect.width() * 0.28)
        right_limit = view_rect.right() - (view_rect.width() * 0.28)

        if x < left_limit or x > right_limit:
            target_left = max(0.0, x - (view_rect.width() * 0.35))
            self.horizontalScrollBar().setValue(int(target_left))
