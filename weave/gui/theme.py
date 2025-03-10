from PySide6.QtGui import QColor


class ThemeManager:
    """Theme manager for consistent styling across the application."""

    def __init__(self, dark_mode=False):
        self.dark_mode = dark_mode
        self._define_colors()

    def _define_colors(self):
        """Define color palettes for light and dark modes."""
        if self.dark_mode:
            # Dark theme colors.
            self.background = QColor("#212121")
            self.foreground = QColor("#FFFFFF")
            self.grid = QColor("#5C5C5C")
            self.selected = QColor("#3D7EFF")
            self.node_qubit = QColor("#464646")
            self.node_z_stabilizer = QColor("#2D5D6C")
            self.node_x_stabilizer = QColor("#6B0E1E")
            self.node_bit = QColor("#FFFFFF")
            self.node_parity = QColor("#FFFFFF")
            self.menu_bg = QColor(45, 45, 45, 230)
            self.menu_selected = QColor(70, 70, 70, 230)
            self.menu_separator = QColor(100, 100, 100)
        else:
            # Light theme colors.
            self.background = QColor("#FFFFFF")
            self.foreground = QColor("#000000")
            self.grid = QColor("#A3A3A3")
            self.selected = QColor("#1976D2")
            self.node_qubit = QColor("#D3D3D3")
            self.node_z_stabilizer = QColor("#ADD8E6")
            self.node_x_stabilizer = QColor("#FFC0CB")
            self.node_bit = QColor("#000000")
            self.node_parity = QColor("#000000")
            self.menu_bg = QColor(245, 245, 245, 230)
            self.menu_selected = QColor(230, 230, 230, 230)
            self.menu_separator = QColor(220, 220, 220)

    def set_dark_mode(self, dark_mode):
        if self.dark_mode != dark_mode:
            self.dark_mode = dark_mode
            self._define_colors()

    def get_node_color(self, node_type):
        if node_type == "qubit":
            return self.node_qubit
        elif node_type == "Z_stabilizer":
            return self.node_z_stabilizer
        elif node_type == "X_stabilizer":
            return self.node_x_stabilizer
        elif node_type == "bit":
            return self.node_bit
        elif node_type == "parity_check":
            return self.node_parity
        return self.foreground

    def get_menu_style(self, is_context_menu=False):
        """Get the style sheet for menus."""
        text_color = self.foreground.name()

        # Use a narrower width for context menus.
        min_width = "150px" if is_context_menu else "200px"

        return f"""
            QMenu {{
                background-color: {self.menu_bg.name()};
                border: none;
                border-radius: 8px;
                padding: 5px;
                margin: 0;
                font-family: "Segoe UI", "Helvetica Neue", sans-serif;
                font-size: 12px;
                color: {text_color};
                min-width: {min_width};
            }}
            QMenu::item {{
                padding: 8px 20px;
                background-color: transparent;
                border: none;
                border-radius: 5px;
                color: {text_color};
            }}
            QMenu::item:selected {{
                background-color: {self.menu_selected.name()};
                color: {text_color};
            }}
            QMenu::separator {{
                height: 1px;
                background: {self.menu_separator.name()};
                margin: 5px 10px;
            }}
        """
