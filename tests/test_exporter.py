from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.exporter import StrExporter
from models.subtitle_segment import SubtitleSegment


class ExporterTests(unittest.TestCase):
    def test_exporta_formato_srt_con_extension_str(self) -> None:
        segments = [
            SubtitleSegment(id="2", start_ms=1500, end_ms=2200, text="Second"),
            SubtitleSegment(id="1", start_ms=0, end_ms=1000, text="First"),
        ]
        content = StrExporter.serialize(segments)

        self.assertIn("1\n00:00:00,000 --> 00:00:01,000\nFirst\n", content)
        self.assertIn("2\n00:00:01,500 --> 00:00:02,200\nSecond\n", content)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "demo.str"
            StrExporter.export(segments, out_path)
            self.assertTrue(out_path.exists())
            self.assertEqual(out_path.read_text(encoding="utf-8"), content)

    def test_falla_si_hay_rango_invalido(self) -> None:
        invalid_segments = [SubtitleSegment(id="x", start_ms=1000, end_ms=900, text="Bad")]
        with self.assertRaises(ValueError):
            StrExporter.serialize(invalid_segments)


if __name__ == "__main__":
    unittest.main()

