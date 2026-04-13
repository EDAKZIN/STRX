from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(slots=True)
class SubtitleSegment:
    id: str
    start_ms: int
    end_ms: int
    text: str
    source: str = "ocr"
    confidence: float | None = None
    bbox: list[tuple[float, float]] | None = None

    def copy_with(self, **changes: object) -> "SubtitleSegment":
        return replace(self, **changes)

