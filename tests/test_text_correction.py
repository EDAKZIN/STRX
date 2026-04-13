from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.text_correction import LanguageCorrector


class TextCorrectionTests(unittest.TestCase):
    def test_corrige_palabra_inglesa_simple(self) -> None:
        corrector = LanguageCorrector("en")
        source = "hellp world"
        corrected = corrector.correct_sentence(source, average_confidence=0.4)
        self.assertNotEqual(source, corrected)
        self.assertTrue(corrected.lower().endswith(" world"))

    def test_evita_sobrecorregir_con_confianza_alta(self) -> None:
        corrector = LanguageCorrector("en")
        source = "hellp world"
        corrected = corrector.correct_sentence(source, average_confidence=0.95)
        self.assertEqual(source, corrected)


if __name__ == "__main__":
    unittest.main()
