import sys
from PySide6.QtWidgets import QApplication
from .windows import MainWindow


def editor():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())
