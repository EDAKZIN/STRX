from __future__ import annotations

import re


TIMECODE_PATTERN = re.compile(r"^(\d{2}):([0-5]\d):([0-5]\d),(\d{3})$")


def format_timecode(milliseconds: int) -> str:
    if milliseconds < 0:
        raise ValueError("El tiempo no puede ser negativo.")

    hours = milliseconds // 3_600_000
    remainder = milliseconds % 3_600_000
    minutes = remainder // 60_000
    remainder %= 60_000
    seconds = remainder // 1_000
    millis = remainder % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def parse_timecode(value: str) -> int:
    match = TIMECODE_PATTERN.match(value.strip())
    if not match:
        raise ValueError(f"Timecode inválido: {value}")

    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    return (hours * 3_600_000) + (minutes * 60_000) + (seconds * 1_000) + millis

