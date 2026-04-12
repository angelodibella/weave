"""Entry point for the Weave GUI editor.

The main window hosts the interactive canvas and a status bar that
shows live node/edge/graph counts. The ``wv`` console entry point
launches this module.
"""

import sys

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QStatusBar

from .canvas import Canvas


class MainWindow(QMainWindow):
    """The top-level window for the Weave visual editor."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Weave Editor")
        self.canvas = Canvas(self)
        self.setCentralWidget(self.canvas)

        # Status bar with live counts.
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_label = QLabel()
        self._status_bar.addPermanentWidget(self._status_label)

        # Connect model signals to status update.
        self.canvas.model.model_changed.connect(self._update_status)
        self.canvas.model.graph_detected.connect(self._update_status)
        self.canvas.model.graph_removed.connect(self._update_status)
        self._update_status()

    def _update_status(self) -> None:
        """Refresh the status bar with current canvas statistics."""
        model = self.canvas.model
        n_nodes = len(model.nodes)
        n_edges = len(model.edges)
        n_graphs = len(model.graphs)

        parts = [f"Nodes: {n_nodes}", f"Edges: {n_edges}"]
        if n_graphs:
            parts.append(f"Graphs: {n_graphs}")

        # If exactly one graph is detected, show its code parameters.
        if n_graphs == 1:
            try:
                from . import code_bridge

                code = code_bridge.graph_to_css_code(model, model.graphs[0].node_ids)
                n_data = len(code.data_qubits)
                k = code.k
                crossings = len(code.crossings)
                parts.append(f"[[{n_data}, {k}]]")
                if crossings:
                    parts.append(f"Crossings: {crossings}")
            except Exception:
                pass

        grid_indicator = " | Grid: ON" if self.canvas.grid_mode else ""
        self._status_label.setText("  |  ".join(parts) + grid_indicator)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Weave Editor")
    app.setOrganizationName("weave")

    window = MainWindow()
    window.resize(1000, 700)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
