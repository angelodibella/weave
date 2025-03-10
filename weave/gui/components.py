from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QPen, QColor, QRadialGradient, QPainterPath
from PySide6.QtCore import Qt, QPointF, QRectF, QPropertyAnimation, QEasingCurve, Signal, Property


class ToggleSwitch(QWidget):
    toggled = Signal(bool)

    def __init__(self, checked=False, parent=None):
        super().__init__(parent)
        self._checked = checked
        self._toggle_value = 1.0 if checked else 0.0

        self._animation = QPropertyAnimation(self, b"toggle_value")
        self._animation.setDuration(100)
        self._animation.setEasingCurve(QEasingCurve.InOutQuad)

        self.setFixedSize(40, 20)
        # Prevent this widget from closing parent menu.
        self.setAttribute(Qt.WA_NoMousePropagation, True)

    def get_value(self):
        return self._toggle_value

    def set_value(self, value):
        self._toggle_value = value
        self.update()

    # Define the Qt property for animation.
    toggle_value = Property(float, get_value, set_value)

    def setChecked(self, checked):
        if self._checked == checked:
            return

        self._checked = checked

        # Force animation to stop and restart.
        self._animation.stop()
        self._animation.setStartValue(self._toggle_value)
        self._animation.setEndValue(1.0 if checked else 0.0)
        self._animation.start()

    def mousePressEvent(self, event):
        # Toggle on press and make sure to trigger animation.
        event.accept()
        self._checked = not self._checked

        # Explicitly start animation with correct values.
        self._animation.stop()
        self._animation.setStartValue(self._toggle_value)
        self._animation.setEndValue(1.0 if self._checked else 0.0)
        self._animation.start()

        # Emit signal after animation setup.
        self.toggled.emit(self._checked)

    def mouseReleaseEvent(self, event):
        # Consume the event to prevent it from propagating.
        event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()

        # Calculate rounded rectangle path for proper shadow and background.
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, rect.width(), rect.height()), rect.height() / 2, rect.height() / 2)

        # Draw shadow.
        painter.save()
        painter.translate(0, 1)
        painter.setBrush(QColor(0, 0, 0, 30))
        painter.setPen(Qt.NoPen)
        painter.drawPath(path)
        painter.restore()

        # Draw background.
        if self._checked:
            color = QColor("#3D7EFF")  # Modern blue
        else:
            color = QColor("#bfc0c0")
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawPath(path)

        # Calculate knob position using animation value.
        knob_radius = (rect.height() - 4) / 2
        knob_x = 2 + (rect.width() - 2 * knob_radius - 4) * self._toggle_value

        # Ensure knob is vertically centered.
        knob_center = QPointF(knob_x + knob_radius, rect.height() / 2)

        # Draw knob shadow.
        painter.setBrush(QColor(0, 0, 0, 20))
        painter.drawEllipse(knob_center + QPointF(0, 0.5), knob_radius, knob_radius)

        # Draw knob with slight gradient.
        gradient = QRadialGradient(knob_center, knob_radius * 1.2)
        gradient.setColorAt(0, QColor("#ffffff"))
        gradient.setColorAt(1, QColor("#f0f0f0"))
        painter.setBrush(gradient)
        painter.drawEllipse(knob_center, knob_radius, knob_radius)


class MenuIcon(QWidget):
    """Hamburger menu icon with animation support."""

    clicked = Signal()

    def __init__(self, parent=None, theme_manager=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.setFixedSize(30, 30)
        self.setCursor(Qt.PointingHandCursor)
        self._is_open = False

        # Initialize the attribute with the same name as property.
        self._open_state = 0.0

        self._animation = QPropertyAnimation(self, b"open_state")
        self._animation.setDuration(150)
        self._animation.setEasingCurve(QEasingCurve.InOutQuad)

    def get_open_state(self):
        return self._open_state

    def set_open_state(self, value):
        self._open_state = value
        self.update()

    # Define the Qt property for animation
    open_state = Property(float, get_open_state, set_open_state)

    def setOpen(self, is_open):
        if self._is_open == is_open:
            return

        self._is_open = is_open
        self._animation.stop()
        self._animation.setStartValue(self._open_state)
        self._animation.setEndValue(1.0 if is_open else 0.0)
        self._animation.start()

    def mousePressEvent(self, event):
        self.clicked.emit()
        event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Get foreground color from theme if available.
        pen_color = self.theme_manager.foreground if self.theme_manager else QColor("black")

        pen = QPen(pen_color, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)

        # Center of the widget.
        center_x = self.width() / 2
        center_y = self.height() / 2

        # Line length and spacing.
        line_length = 14
        line_spacing = 5

        # Animations from hamburger to X.
        if self._open_state == 0:
            # Draw three lines.
            painter.drawLine(center_x - line_length / 2, center_y - line_spacing,
                             center_x + line_length / 2, center_y - line_spacing)
            painter.drawLine(center_x - line_length / 2, center_y,
                             center_x + line_length / 2, center_y)
            painter.drawLine(center_x - line_length / 2, center_y + line_spacing,
                             center_x + line_length / 2, center_y + line_spacing)
        else:
            # Top line transforms to \ part of X.
            painter.drawLine(
                center_x - line_length / 2,
                center_y - line_spacing + self._open_state * (line_spacing * 2),
                center_x + line_length / 2,
                center_y + line_spacing - self._open_state * (line_spacing * 2)
            )

            # Middle line fades out.
            if self._open_state < 1.0:
                pen.setColor(QColor(pen_color.red(), pen_color.green(), pen_color.blue(),
                                    int(255 * (1 - self._open_state))))
                painter.setPen(pen)
                painter.drawLine(
                    center_x - line_length / 2 + self._open_state * (line_length / 2),
                    center_y,
                    center_x + line_length / 2 - self._open_state * (line_length / 2),
                    center_y
                )
                pen.setColor(pen_color)
                painter.setPen(pen)

            # Bottom line transforms to / part of X.
            painter.drawLine(
                center_x - line_length / 2,
                center_y + line_spacing - self._open_state * (line_spacing * 2),
                center_x + line_length / 2,
                center_y - line_spacing + self._open_state * (line_spacing * 2)
            )
