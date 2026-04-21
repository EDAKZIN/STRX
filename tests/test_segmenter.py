from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.segmenter import TextChangeSegmenter


class SegmenterTests(unittest.TestCase):
    def test_segmenta_por_cambio_texto(self) -> None:
        segmenter = TextChangeSegmenter(min_duration_ms=100)
        result_1 = segmenter.push(0, "Hello")
        result_2 = segmenter.push(1000, "Hello")
        result_3 = segmenter.push(2000, "World")
        result_4 = segmenter.flush(3000)

        self.assertEqual(result_1, [])
        self.assertEqual(result_2, [])
        self.assertEqual(len(result_3), 1)
        self.assertEqual(result_3[0].text, "Hello")
        self.assertEqual(result_3[0].start_ms, 0)
        self.assertEqual(result_3[0].end_ms, 2000)
        self.assertEqual(len(result_4), 1)
        self.assertEqual(result_4[0].text, "World")
        self.assertEqual(result_4[0].start_ms, 2000)
        self.assertEqual(result_4[0].end_ms, 3000)

    def test_cierra_segmento_si_no_hay_texto(self) -> None:
        segmenter = TextChangeSegmenter(min_duration_ms=100)
        segmenter.push(500, "Line")
        closed = segmenter.push(800, "")
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].start_ms, 500)
        self.assertEqual(closed[0].end_ms, 800)

    def test_conserva_variante_mas_consistente(self) -> None:
        segmenter = TextChangeSegmenter(min_duration_ms=100, similarity_threshold=0.8)
        segmenter.push(0, "I really like this")
        segmenter.push(1000, "I really iike this")
        segmenter.push(2000, "I really like this")
        closed = segmenter.push(3000, "Next sentence")

        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].text, "I really like this")
        self.assertEqual(closed[0].start_ms, 0)
        self.assertEqual(closed[0].end_ms, 3000)

    def test_prioriza_variante_con_mejor_confianza(self) -> None:
        segmenter = TextChangeSegmenter(min_duration_ms=100, similarity_threshold=0.72)
        segmenter.push(0, "I really iike this", confidence=0.2)
        segmenter.push(1000, "I really iike this", confidence=0.2)
        segmenter.push(2000, "I really like this", confidence=0.93)
        segmenter.push(3000, "I really iike this", confidence=0.2)
        segmenter.push(4000, "I really like this", confidence=0.94)
        closed = segmenter.push(5000, "Next sentence", confidence=0.9)

        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].text, "I really like this")

    def test_penaliza_texto_con_exceso_de_simbolos(self) -> None:
        noisy_score = TextChangeSegmenter._quality_score("##==??____")
        clean_score = TextChangeSegmenter._quality_score("Please wait")
        self.assertLess(noisy_score, clean_score)
        self.assertLess(noisy_score, -10.0)


if __name__ == "__main__":
    unittest.main()
