from __future__ import annotations

from pathlib import Path
from typing import Iterable

from core.timecode import format_timecode
from models.subtitle_segment import SubtitleSegment


def _format_srt_timecode(ms: int) -> str:
    total_s, millis = divmod(ms, 1000)
    total_m, secs = divmod(total_s, 60)
    hours, mins = divmod(total_m, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


class SrtExporter:
    @staticmethod
    def serialize(segments: Iterable[SubtitleSegment]) -> str:
        ordered_segments = sorted(segments, key=lambda item: (item.start_ms, item.end_ms, item.id))
        lines: list[str] = []

        for index, segment in enumerate(ordered_segments, start=1):
            text = segment.text.strip()
            if not text:
                raise ValueError(f"El segmento {index} no tiene texto.")
            if segment.end_ms <= segment.start_ms:
                raise ValueError(f"El segmento {index} tiene rango de tiempo inválido.")

            lines.append(str(index))
            lines.append(f"{_format_srt_timecode(segment.start_ms)} --> {_format_srt_timecode(segment.end_ms)}")
            lines.append(text)
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    @classmethod
    def export(cls, segments: Iterable[SubtitleSegment], output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(cls.serialize(segments), encoding="utf-8")
        return path


class StrExporter:
    @staticmethod
    def serialize(segments: Iterable[SubtitleSegment]) -> str:
        ordered_segments = sorted(segments, key=lambda item: (item.start_ms, item.end_ms, item.id))
        lines: list[str] = []

        for index, segment in enumerate(ordered_segments, start=1):
            text = segment.text.strip()
            if not text:
                raise ValueError(f"El segmento {index} no tiene texto.")
            if segment.end_ms <= segment.start_ms:
                raise ValueError(f"El segmento {index} tiene rango de tiempo inválido.")

            lines.append(str(index))
            lines.append(f"{format_timecode(segment.start_ms)} --> {format_timecode(segment.end_ms)}")
            lines.append(text)
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    @classmethod
    def export(cls, segments: Iterable[SubtitleSegment], output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(cls.serialize(segments), encoding="utf-8")
        return path

