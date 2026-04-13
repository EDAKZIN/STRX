from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.timecode import format_timecode, parse_timecode


class TimecodeTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        original_ms = 3_901_042
        value = format_timecode(original_ms)
        parsed = parse_timecode(value)
        self.assertEqual(parsed, original_ms)

    def test_parse_invalido(self) -> None:
        with self.assertRaises(ValueError):
            parse_timecode("01:99:12,111")

    def test_format_negativo(self) -> None:
        with self.assertRaises(ValueError):
            format_timecode(-1)


if __name__ == "__main__":
    unittest.main()

