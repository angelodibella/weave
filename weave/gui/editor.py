import sys
from PySide6.QtWidgets import QApplication
from .main_editor_window import MainEditorWindow


def main():
    app = QApplication(sys.argv)
    window = MainEditorWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
