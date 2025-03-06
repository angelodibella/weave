import json
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtCore import Qt, QPoint


class CodeEditorCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.nodes = []  # List of dicts: {'pos': (x, y), 'type': 'B' or other types}
        self.edges = []  # List of tuples: (node_index_1, node_index_2)
        self.current_mode = None  # Modes: 'add', 'select', etc.
        self.selected_node = None

    def enable_add_node_mode(self):
        self.current_mode = 'add'

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(QColor("black"), 2)
        painter.setPen(pen)

        # Draw nodes (as circles)
        for node in self.nodes:
            x, y = node['pos']
            radius = 10
            painter.drawEllipse(QPoint(x, y), radius, radius)

        # Draw edges (as lines)
        for edge in self.edges:
            p1 = self.nodes[edge[0]]['pos']
            p2 = self.nodes[edge[1]]['pos']
            painter.drawLine(p1[0], p1[1], p2[0], p2[1])

    def mousePressEvent(self, event):
        if self.current_mode == 'add':
            # Add a new node at the click position
            new_node = {'pos': (event.position().x(), event.position().y()), 'type': 'B'}
            self.nodes.append(new_node)
            self.update()  # Redraw the canvas

    # You can also add mouseMoveEvent, mouseReleaseEvent for drag-and-drop
    # and keyPressEvent for shortcuts (like delete, etc.).

    def save_to_file(self, filename):
        # Serialize the internal model to JSON
        data = {'nodes': self.nodes, 'edges': self.edges}
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_from_file(self, filename):
        # Load and update the internal model
        with open(filename, 'r') as f:
            data = json.load(f)
        self.nodes = data.get('nodes', [])
        self.edges = data.get('edges', [])
        self.update()
