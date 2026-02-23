"""Entry point for the Weave GUI editor."""

import sys

from PySide6.QtWidgets import QApplication, QMainWindow

from .canvas import Canvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Weave Editor")
        self.canvas = Canvas(self)
        self.setCentralWidget(self.canvas)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
