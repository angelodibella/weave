from PySide6.QtWidgets import QMainWindow
from .code_editor_canvas import CodeEditorCanvas


class MainEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Weave QEC Code Editor")
        self.canvas = CodeEditorCanvas(self)
        self.setCentralWidget(self.canvas)
