from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.ocr_worker import OcrDetection, OcrWorker


def build_detection(text: str, cx: float, cy: float, confidence: float = 0.9) -> OcrDetection:
    bbox = [
        (cx - 40.0, cy - 12.0),
        (cx + 40.0, cy - 12.0),
        (cx + 40.0, cy + 12.0),
        (cx - 40.0, cy + 12.0),
    ]
    return OcrDetection(
        text=text,
        confidence=confidence,
        bbox=bbox,
        center=(cx, cy),
    )


class OcrTrackingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.worker = OcrWorker(
            video_path=Path("fake.mp4"),
            sample_fps=1.0,
            model_paths={"paddle_home": Path(".")},
            enable_language_correction=False,
            use_gpu=False,
        )

    def test_separa_textos_simultaneos_en_pistas_distintas(self) -> None:
        frame_diagonal = 2000.0
        detections = [
            build_detection("Give me my glasses back!", 520.0, 870.0),
            build_detection("Orange", 420.0, 210.0),
        ]

        self.assertEqual(self.worker._update_tracks(0, detections, frame_diagonal), [])
        self.assertEqual(self.worker._update_tracks(1000, detections, frame_diagonal), [])
        self.assertEqual(self.worker._update_tracks(2000, [], frame_diagonal), [])
        self.assertEqual(self.worker._update_tracks(3000, [], frame_diagonal), [])
        closed = self.worker._update_tracks(4000, [], frame_diagonal)

        self.assertEqual(len(closed), 2)
        self.assertSetEqual(
            {item.text for item in closed},
            {"Give me my glasses back!", "Orange"},
        )

    def test_cambio_texto_misma_posicion_cierra_segmento_previo(self) -> None:
        frame_diagonal = 2000.0
        base_detection = build_detection("This is line", 640.0, 830.0)
        variant_detection = build_detection("This is Iine", 640.0, 830.0)
        next_detection = build_detection("Next subtitle", 640.0, 830.0)

        self.assertEqual(self.worker._update_tracks(0, [base_detection], frame_diagonal), [])
        self.assertEqual(self.worker._update_tracks(1000, [variant_detection], frame_diagonal), [])

        closed = self.worker._update_tracks(2000, [next_detection], frame_diagonal)
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].text, "This is line")
        self.assertEqual(closed[0].start_ms, 0)
        self.assertEqual(closed[0].end_ms, 2000)

        tail = self.worker._flush_tracks(3000)
        self.assertEqual(len(tail), 1)
        self.assertEqual(tail[0].text, "Next subtitle")

    def test_filtra_ruido_con_simbolos(self) -> None:
        self.assertTrue(self.worker._is_noise_text("@@12##", 0.95))
        self.assertTrue(self.worker._is_noise_text("12345", 0.90))
        self.assertFalse(self.worker._is_noise_text("Give me my glasses back!", 0.62))

    def test_agrupa_palabras_de_misma_linea(self) -> None:
        detections = [
            build_detection("Give", 380.0, 840.0),
            build_detection("me", 470.0, 841.0),
            build_detection("my", 540.0, 840.0),
            build_detection("glasses", 650.0, 842.0),
        ]
        merged = self.worker._merge_line_detections(detections)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "Give me my glasses")

    def test_no_agrupa_textos_lejanos_misma_altura(self) -> None:
        detections = [
            build_detection("Orange", 220.0, 220.0),
            build_detection("Subtitle", 1260.0, 220.0),
        ]
        merged = self.worker._merge_line_detections(detections)
        self.assertEqual(len(merged), 2)
        self.assertSetEqual({item.text for item in merged}, {"Orange", "Subtitle"})

    def test_agrupa_lineas_de_un_mismo_globo(self) -> None:
        lines = [
            build_detection("The cookies look ready.", 1150.0, 360.0),
            build_detection("I hope my son's girlfriend likes", 1150.0, 420.0),
            build_detection("them.", 1150.0, 478.0),
        ]
        merged = self.worker._merge_bubble_detections(lines)
        self.assertEqual(len(merged), 1)
        self.assertEqual(
            merged[0].text,
            "The cookies look ready. I hope my son's girlfriend likes them.",
        )

    def test_no_mezcla_globos_distintos(self) -> None:
        lines = [
            build_detection("Orange", 320.0, 260.0),
            build_detection("is not funny!", 320.0, 320.0),
            build_detection("Give it back!", 1300.0, 290.0),
        ]
        merged = self.worker._merge_bubble_detections(lines)
        self.assertEqual(len(merged), 2)
        self.assertSetEqual(
            {item.text for item in merged},
            {"Orange is not funny!", "Give it back!"},
        )

    def test_descarta_segmento_de_palabra_unica_sospechosa(self) -> None:
        detections = [build_detection("NMN", 620.0, 860.0)]
        frame_diagonal = 2000.0

        self.assertEqual(self.worker._update_tracks(0, detections, frame_diagonal), [])
        self.assertEqual(self.worker._update_tracks(1000, [], frame_diagonal), [])
        self.assertEqual(self.worker._update_tracks(2000, [], frame_diagonal), [])
        dropped = self.worker._update_tracks(3000, [], frame_diagonal)
        self.assertEqual(dropped, [])

    def test_conserva_palabra_unica_valida(self) -> None:
        detections = [build_detection("Wait", 630.0, 850.0)]
        frame_diagonal = 2000.0

        self.assertEqual(self.worker._update_tracks(0, detections, frame_diagonal), [])
        self.assertEqual(self.worker._update_tracks(1000, [], frame_diagonal), [])
        self.assertEqual(self.worker._update_tracks(2000, [], frame_diagonal), [])
        kept = self.worker._update_tracks(3000, [], frame_diagonal)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].text, "Wait")


if __name__ == "__main__":
    unittest.main()
