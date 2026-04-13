from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QPainter, QPaintEvent
from PyQt6.QtWidgets import QCheckBox


class ToggleSwitch(QCheckBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setText("")
        self.setFixedSize(48, 28)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(48, 28)

    def paintEvent(self, event: QPaintEvent) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)

        rect = self.rect().adjusted(1, 1, -1, -1)
        radius = rect.height() / 2
        knob_size = rect.height() - 6
        knob_y = rect.y() + 3
        knob_x = rect.x() + 3 if not self.isChecked() else rect.right() - knob_size - 2

        if not self.isEnabled():
            track_color = QColor("#1a2230")
            knob_color = QColor("#5b6474")
        elif self.isChecked():
            track_color = QColor("#345777")
            knob_color = QColor("#d9e5f3")
        else:
            track_color = QColor("#2a3340")
            knob_color = QColor("#b8c2d0")

        painter.setBrush(track_color)
        painter.drawRoundedRect(rect, radius, radius)

        painter.setBrush(knob_color)
        painter.drawEllipse(int(knob_x), int(knob_y), int(knob_size), int(knob_size))
        painter.end()

    def hitButton(self, pos) -> bool:  # type: ignore[override]
        # Permite alternar el switch en toda su superficie, no solo en un área interna.
        return self.rect().contains(pos)
