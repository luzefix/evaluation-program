"""Placeholder tabs for methods that are planned but not implemented yet."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlaceholderTab(QWidget):
    def __init__(self, title: str, text: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lab = QLabel(f"<h2>{title}</h2><p>{text}</p>")
        lab.setWordWrap(True)
        lab.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lay.addWidget(lab)


PLANNED = [
    ("MPMS", "Magnetisation M(T), M(H) from Quantum Design MPMS .dat files: diamagnetic substrate "
     "background subtraction, emu → µB/f.u. and emu/cm³, T_C / T_N, coercive field, remanence."),
    ("Ellipsometry", "Ψ/Δ display and analysis. The instrument and its export format are still to be defined."),
    ("TEM", "To be discussed."),
]
