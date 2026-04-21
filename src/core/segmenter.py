from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import math


@dataclass(slots=True)
class SegmentDraft:
    start_ms: int
    end_ms: int
    text: str


class TextChangeSegmenter:
    def __init__(self, min_duration_ms: int = 200, similarity_threshold: float = 0.82) -> None:
        self.min_duration_ms = min_duration_ms
        self.similarity_threshold = similarity_threshold
        self.current_text: str | None = None
        self.current_start_ms: int | None = None
        self.last_timestamp_ms: int | None = None
        self.variants: dict[str, int] = {}
        self.confidences_total: dict[str, float] = {}
        self.variant_order: dict[str, int] = {}
        self._variant_counter = 0

    def push(self, timestamp_ms: int, text: str, confidence: float | None = 0.0) -> list[SegmentDraft]:
        normalized_text = self._normalize_text(text)
        closed: list[SegmentDraft] = []

        if not normalized_text:
            draft = self._close_current(timestamp_ms)
            if draft:
                closed.append(draft)
            return closed

        if self.current_text is None:
            self._start_segment(timestamp_ms, normalized_text, confidence)
            return closed

        matched_variant, similarity = self._find_closest_variant(normalized_text)
        if matched_variant is not None and similarity >= self.similarity_threshold:
            self._register_variant(normalized_text, confidence)
            self.current_text = self._choose_canonical_variant()
            self.last_timestamp_ms = timestamp_ms
            return closed

        if normalized_text == self.current_text:
            self._register_variant(normalized_text, confidence)
            self.current_text = self._choose_canonical_variant()
            self.last_timestamp_ms = timestamp_ms
            return closed

        draft = self._close_current(timestamp_ms)
        if draft:
            closed.append(draft)

        self._start_segment(timestamp_ms, normalized_text, confidence)
        return closed

    def flush(self, end_timestamp_ms: int) -> list[SegmentDraft]:
        draft = self._close_current(end_timestamp_ms)
        return [draft] if draft else []

    def _close_current(self, fallback_end_ms: int) -> SegmentDraft | None:
        if self.current_text is None or self.current_start_ms is None:
            return None

        end_ms = max(fallback_end_ms, self.last_timestamp_ms or fallback_end_ms)
        end_ms = max(self.current_start_ms + self.min_duration_ms, end_ms)
        draft = SegmentDraft(
            start_ms=self.current_start_ms,
            end_ms=end_ms,
            text=self._choose_canonical_variant(),
        )

        self.current_text = None
        self.current_start_ms = None
        self.last_timestamp_ms = None
        self.variants.clear()
        self.confidences_total.clear()
        self.variant_order.clear()
        self._variant_counter = 0
        return draft

    def _start_segment(self, timestamp_ms: int, text: str, confidence: float | None) -> None:
        self.current_text = text
        self.current_start_ms = timestamp_ms
        self.last_timestamp_ms = timestamp_ms
        self.variants.clear()
        self.confidences_total.clear()
        self.variant_order.clear()
        self._variant_counter = 0
        self._register_variant(text, confidence)

    def _register_variant(self, text: str, confidence: float | None) -> None:
        self.variants[text] = self.variants.get(text, 0) + 1
        self.confidences_total[text] = self.confidences_total.get(text, 0.0) + self._sanitize_confidence(confidence)
        if text not in self.variant_order:
            self.variant_order[text] = self._variant_counter
            self._variant_counter += 1

    def _find_closest_variant(self, text: str) -> tuple[str | None, float]:
        best_variant: str | None = None
        best_score = 0.0
        for variant in self.variants:
            score = self._text_similarity(text, variant)
            if score > best_score:
                best_score = score
                best_variant = variant
        return best_variant, best_score

    def _choose_canonical_variant(self) -> str:
        if not self.variants:
            return self.current_text or ""

        return max(self.variants.keys(), key=self._variant_score)

    def _variant_score(self, text: str) -> tuple[float, float, int, int, int]:
        count = self.variants.get(text, 0)
        total_confidence = self.confidences_total.get(text, 0.0)
        average_confidence = total_confidence / max(1, count)
        certainty_score = average_confidence * math.log1p(max(0, count))
        return (
            certainty_score,
            self._quality_score(text),
            count,
            len(text),
            -self.variant_order.get(text, 0),
        )

    @staticmethod
    def _text_similarity(left: str, right: str) -> float:
        return SequenceMatcher(a=left, b=right).ratio()

    @staticmethod
    def _quality_score(text: str) -> float:
        total_chars = len(text)
        if total_chars > 0:
            alnum_chars = sum(1 for char in text if char.isalnum())
            if (alnum_chars / total_chars) < 0.60:
                return -25.0
        letters = sum(1 for char in text if char.isalpha())
        words = len([token for token in text.split(" ") if token])
        uncommon = sum(
            1
            for char in text
            if not (char.isalnum() or char.isspace() or char in ".,!?;:'\"-()[]")
        )
        return (words * 2.0) + (letters * 0.05) - (uncommon * 1.5)

    @staticmethod
    def _sanitize_confidence(confidence: float | None) -> float:
        if confidence is None:
            return 0.0
        try:
            value = float(confidence)
        except Exception:
            return 0.0
        if value != value:
            return 0.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def _normalize_text(value: str) -> str:
        return " ".join(value.strip().split())
