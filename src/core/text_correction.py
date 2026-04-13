from __future__ import annotations

import re
from typing import Iterable


WORD_PATTERN = re.compile(r"[^\W\d_]+|\d+|[^\w\s]+|\s+", re.UNICODE)


class LanguageCorrector:
    def __init__(self, language_code: str) -> None:
        try:
            from spellchecker import SpellChecker
        except Exception as exc:
            raise RuntimeError(
                "No se encontró pyspellchecker. Instálalo con: pip install pyspellchecker"
            ) from exc

        language_map = {
            "en": "en",
            "es": "es",
        }
        spell_language = language_map.get(language_code, "en")
        self.spell = SpellChecker(language=spell_language, distance=1)

    def correct_sentence(self, text: str, average_confidence: float | None = None) -> str:
        if not text.strip():
            return text

        # Evita sobrecorrección cuando la lectura OCR ya parece estable.
        if average_confidence is not None and average_confidence >= 0.88:
            return text

        tokens = WORD_PATTERN.findall(text)
        corrected: list[str] = []

        for token in tokens:
            if not token:
                continue
            if not self._is_word(token):
                corrected.append(token)
                continue

            if self._skip_token(token):
                corrected.append(token)
                continue

            lower = token.lower()
            if lower in self.spell:
                corrected.append(token)
                continue

            suggestion = self.spell.correction(lower)
            if not suggestion:
                corrected.append(token)
                continue

            if not self._is_reasonable_change(lower, suggestion):
                corrected.append(token)
                continue

            corrected.append(self._preserve_case(token, suggestion))

        return "".join(corrected)

    @staticmethod
    def _is_word(token: str) -> bool:
        return token.isalpha()

    @staticmethod
    def _skip_token(token: str) -> bool:
        lower = token.lower()
        if len(lower) < 4:
            return True
        if token.isupper() and len(token) <= 4:
            return True
        return False

    @staticmethod
    def _is_reasonable_change(original: str, suggestion: str) -> bool:
        if original == suggestion:
            return False
        if abs(len(original) - len(suggestion)) > 2:
            return False
        return True

    @staticmethod
    def _preserve_case(original: str, suggestion: str) -> str:
        if original.isupper():
            return suggestion.upper()
        if original.istitle():
            return suggestion.title()
        return suggestion


def average_confidence(confidences: Iterable[float]) -> float | None:
    values = [value for value in confidences if 0.0 <= value <= 1.0]
    if not values:
        return None
    return sum(values) / len(values)

