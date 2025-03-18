from PySide6.QtWidgets import QMainWindow
from .canvas import Canvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Weave Editor")
        self.canvas = Canvas(self)
        self.setCentralWidget(self.canvas)
