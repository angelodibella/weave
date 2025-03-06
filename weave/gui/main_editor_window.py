from PySide6.QtWidgets import QMainWindow, QToolBar, QFileDialog
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
from .code_editor_canvas import CodeEditorCanvas


class MainEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Weave QEC Code Editor")
        self.canvas = CodeEditorCanvas(self)
        self.setCentralWidget(self.canvas)
        self._create_toolbar()

    def _create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        add_node_action = QAction("Add Node", self)
        add_node_action.triggered.connect(self.canvas.enable_add_node_mode)
        toolbar.addAction(add_node_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self._save_code)
        toolbar.addAction(save_action)

        load_action = QAction("Load", self)
        load_action.triggered.connect(self._load_code)
        toolbar.addAction(load_action)

    def _save_code(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save QEC Code", "", "JSON Files (*.json)")
        if filename:
            self.canvas.save_to_file(filename)

    def _load_code(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load QEC Code", "", "JSON Files (*.json)")
        if filename:
            self.canvas.load_from_file(filename)
