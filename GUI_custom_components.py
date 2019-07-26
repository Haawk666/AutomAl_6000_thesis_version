# By Haakon Tvedt @ NTNU
"""Module container for low-level custom GUI-elements"""

# Program imports:
import GUI_settings
# External imports:
from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np

# ----------
# Graphic elements:
# ----------


class InteractiveColumn(QtWidgets.QGraphicsEllipseItem):

    """A general interactive graphical element that is meant to represent atomic columns in the GUI.

    Inherits PyQt5.QtWidgets.QGraphicsEllipseItem()
    """

    def __init__(self, ui_obj=None, i=-1, r=5, scale_factor=1):

        """Initialize with optional reference to a MainUI object.

        parameters
        ----------
        ui_obj : GUI.MainUI, optional
            Reference MainUI object.
        i : int, optional
            An integer that is a reference to the index of the relative vertex in
            ui_obj.project_instance.graph.vertices[i]. If i == -1, it is interpreted as no reference.
        """

        self.ui_obj = ui_obj
        self.r = r
        self.i = i
        self.scale_factor = scale_factor
        self.vertex = self.ui_obj.project_instance.graph.vertices[self.i]
        self.center_coor = self.vertex.real_coor()
        self.center_coor = scale_factor * self.center_coor[0] - np.round(self.r / 2), scale_factor * self.center_coor[1] - np.round(self.r / 2)

        super().__init__(0, 0, self.r, self.r)

        self.moveBy(self.center_coor[0], self.center_coor[1])
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)

    def mouseReleaseEvent(self, event: 'QtWidgets.QGraphicsEllipseItem.mouseReleaseEvent'):
        """Pass a mouse release event on to the ui_obj reference object"""
        self.ui_obj.column_selected(self.i)
        # super().__init__()


class InteractivePosColumn(InteractiveColumn):

    def __init__(self, *args):
        """Initialize a positional interactive column.

        Inherits GUI_custom_components.InteractiveColumn. Is used to highlight atomic positions."""
        super().__init__(*args)

        self.selected_pen = GUI_settings.pen_selected_2
        self.selected_brush = GUI_settings.brush_selected_2
        self.unselected_pen = GUI_settings.pen_atom_pos
        self.unselected_brush = GUI_settings.brush_atom_pos
        self.hidden_pen = GUI_settings.pen_atom_pos_hidden
        self.hidden_brush = GUI_settings.brush_atom_pos_hidden
        self.set_style()

    def set_style(self):
        """Set the appearance of the shape"""
        if self.i == self.ui_obj.selected_column:
            self.setPen(self.selected_pen)
            self.setBrush(self.selected_brush)
        else:
            if self.vertex.show_in_overlay:
                self.setPen(self.hidden_pen)
                self.setBrush(self.hidden_brush)
            else:
                self.setPen(self.unselected_pen)
                self.setBrush(self.hidden_brush)


class InteractiveOverlayColumn(InteractiveColumn):

    def __init__(self, *args):
        """Initialize a positional interactive column.

        Inherits GUI_custom_components.InteractiveColumn. Is used to highlight atomic positions."""
        super().__init__(*args)

        self.pen_cu = GUI_settings.pen_cu
        self.pen_si = GUI_settings.pen_si
        self.pen_zn = GUI_settings.pen_zn
        self.pen_al = GUI_settings.pen_al
        self.pen_mg = GUI_settings.pen_mg
        self.pen_ag = GUI_settings.pen_ag
        self.pen_un = GUI_settings.pen_un
        self.selected_pen = GUI_settings.pen_selected_1
        self.brush_cu = GUI_settings.brush_cu
        self.brush_si = GUI_settings.brush_si
        self.brush_al = GUI_settings.brush_al
        self.brush_zn = GUI_settings.brush_zn
        self.brush_mg = GUI_settings.brush_mg
        self.brush_ag = GUI_settings.brush_ag
        self.brush_un = GUI_settings.brush_un
        self.brush_selected = GUI_settings.brush_selected_1
        self.brush_black = GUI_settings.brush_black

        self.set_style()

    def set_style(self):
        """Set the appearance of the shape"""
        if self.i == self.ui_obj.selected_column:
            self.setPen(self.selected_pen)
            self.setBrush(self.brush_selected)
        else:
            self.setBrush(self.brush_black)
            if self.vertex.h_index == 0:
                self.setPen(self.pen_si)
                if self.vertex.level == 0:
                    self.setBrush(self.brush_si)
            elif self.vertex.h_index == 1:
                self.setPen(self.pen_cu)
                if self.vertex.level == 0:
                    self.setBrush(self.brush_cu)
            elif self.vertex.h_index == 2:
                self.setPen(self.pen_zn)
                if self.vertex.level == 0:
                    self.setBrush(self.brush_zn)
            elif self.vertex.h_index == 3:
                self.setPen(self.pen_al)
                if self.vertex.level == 0:
                    self.setBrush(self.brush_al)
            elif self.vertex.h_index == 4:
                self.setPen(self.pen_ag)
                if self.vertex.level == 0:
                    self.setBrush(self.brush_ag)
            elif self.vertex.h_index == 5:
                self.setPen(self.pen_mg)
                if self.vertex.level == 0:
                    self.setBrush(self.brush_mg)
            elif self.vertex.h_index == 6:
                self.setPen(self.pen_un)
                if self.vertex.level == 0:
                    self.setBrush(self.brush_un)
            else:
                print('TODO: Logger')
            if not self.vertex.show_in_overlay:
                self.hide()
            else:
                self.show()


class InteractiveGraphColumn(InteractiveColumn):

    def __init__(self, *args):
        """Initialize a positional interactive column.

        Inherits GUI_custom_components.InteractiveColumn. Is used to highlight atomic positions."""
        super().__init__(*args)

        self.selected_pen = GUI_settings.pen_selected_1
        self.unselected_pen = GUI_settings.pen_graph
        self.level_0_brush = GUI_settings.brush_graph_0
        self.level_1_brush = GUI_settings.brush_graph_1

        self.set_style()

    def set_style(self):
        """Set the appearance of the shape"""
        if self.i == self.ui_obj.selected_column:
            self.setPen(self.selected_pen)
        else:
            self.setPen(self.unselected_pen)
        if self.vertex.level == 0:
            self.setBrush(self.level_0_brush)
        else:
            self.setBrush(self.level_1_brush)


class Arrow(QtWidgets.QGraphicsItemGroup):

    def __init__(self, *args, i=0, j=1, p1=(0, 0), p2=(1, 1), r=1, scale_factor=1, consistent=False, dislocation=False, chb=None):
        super().__init__(*args)

        self.inconsistent_pen = GUI_settings.pen_inconsistent_edge
        self.dislocation_pen = GUI_settings.pen_dislocation_edge
        self.normal_pen = GUI_settings.pen_edge

        self.chb = chb

        self.consistent = consistent
        self.dislocation = dislocation
        self.scale_factor = scale_factor
        self.r = r
        self.p1 = p1
        self.p2 = p2
        self.arrow = None, None
        self.i = i
        self.j = j

        self.make_arrow_obj()
        self.set_style()

    def set_style(self):
        if self.consistent and not self.dislocation:
            self.childItems()[0].setPen(self.normal_pen)
            self.childItems()[0].show()
            self.childItems()[1].hide()
        elif self.dislocation and self.consistent:
            self.childItems()[0].setPen(self.dislocation_pen)
            self.childItems()[0].show()
            self.childItems()[1].setPen(self.dislocation_pen)
            self.childItems()[1].show()
        else:
            self.childItems()[0].setPen(self.inconsistent_pen)
            self.childItems()[0].show()
            self.childItems()[1].setPen(self.inconsistent_pen)
            self.childItems()[1].show()

        if self.chb is not None:
            if not self.chb.isChecked() and not self.consistent:
                self.hide()
            else:
                self.show()
        else:
            self.show()

    def make_arrow_obj(self):

        r_2 = QtCore.QPointF(self.scale_factor * self.p2[0], self.scale_factor * self.p2[1])
        r_1 = QtCore.QPointF(self.scale_factor * self.p1[0], self.scale_factor * self.p1[1])

        r_vec = r_2 - r_1
        r_mag = np.sqrt((r_2.x() - r_1.x()) ** 2 + (r_2.y() - r_1.y()) ** 2)
        factor = self.r / (r_mag * 2)

        k_2 = r_1 + (1 - factor) * r_vec

        theta = np.pi / 4

        l_3 = - factor * QtCore.QPointF(r_vec.x() * np.cos(theta) + r_vec.y() * np.sin(theta), - r_vec.x() * np.sin(theta) + r_vec.y() * np.cos(theta))
        l_3 = k_2 + l_3
        l_4 = - factor * QtCore.QPointF(r_vec.x() * np.cos(-theta) + r_vec.y() * np.sin(-theta), - r_vec.x() * np.sin(-theta) + r_vec.y() * np.cos(-theta))
        l_4 = k_2 + l_4

        tri_2 = (k_2, l_3, l_4)

        poly_2 = QtGui.QPolygonF(tri_2)

        line = QtWidgets.QGraphicsLineItem(self.scale_factor * self.p1[0],
                                           self.scale_factor * self.p1[1],
                                           self.scale_factor * self.p2[0],
                                           self.scale_factor * self.p2[1])
        head_2 = QtWidgets.QGraphicsPolygonItem(poly_2)

        self.addToGroup(line)
        self.addToGroup(head_2)
        self.setZValue(-1)


class ScaleBar(QtWidgets.QGraphicsItemGroup):

    def __init__(self, *args, length=2, scale=5, r=10, height=512):
        """Initialize a positional interactive column.

        Inherits GUI_custom_components.InteractiveColumn. Is used to highlight atomic positions."""
        super().__init__(*args)

        self.length = length
        self.height = height
        self.nano_scale = scale / 1000
        self.r = r
        self.color_brush = GUI_settings.brush_white
        self.make()

    def make(self):
        p1 = 0, 0
        p2 = self.length / self.nano_scale - GUI_settings.pen_scalebar.width(), 0
        line = QtWidgets.QGraphicsLineItem(p1[0], p1[1], p2[0], p2[1])
        line.setPen(GUI_settings.pen_scalebar)

        text = QtWidgets.QGraphicsSimpleTextItem()
        text.setText('{} nm'.format(self.length))
        text.setFont(GUI_settings.font_scalebar)
        text.setPen(GUI_settings.white_pen)
        text.setBrush(GUI_settings.brush_white)
        rect = text.boundingRect()
        text.setX(p2[0] / 2 - rect.width() / 2)
        text.setY(- rect.height() - GUI_settings.pen_scalebar.width())

        self.addToGroup(line)
        self.addToGroup(text)

        self.moveBy(self.r * 6, self.height - self.r * 6)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)


class Legend(QtWidgets.QGraphicsItemGroup):

    def __init__(self, *args):
        super().__init__(*args)


class Overlay(QtWidgets.QWidget):

    def __init__(self, parent=None):

        QtWidgets.QWidget.__init__(self, parent)
        palette = QtGui.QPalette(self.palette())
        palette.setColor(palette.Background, QtCore.Qt.transparent)
        self.setPalette(palette)

    def paintEvent(self, event):

        painter = QtGui.QPainter()
        painter.begin(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(event.rect(), QtGui.QBrush(QtGui.QColor(255, 255, 255, 127)))
        painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))

        for i in range(6):
            if (self.counter / 5) % 6 == i:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(127 + (self.counter % 5) * 32, 127, 127)))
            else:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(127, 127, 127)))
            painter.drawEllipse(
                self.width() / 2 + 30 * np.cos(2 * np.pi * i / 6.0) - 10,
                self.height() / 2 + 30 * np.sin(2 * np.pi * i / 6.0) - 10,
                20, 20)

        painter.end()

    def showEvent(self, event):

        self.timer = self.startTimer(50)
        self.counter = 0

    def timerEvent(self, event):

        self.counter += 1
        self.update()
        if self.counter == 60:
            self.killTimer(self.timer)
            self.hide()

# ----------
# Convenience re-implementations:
# ----------


class SmallButton(QtWidgets.QPushButton):

    def __init__(self, *args, trigger_func=None):
        super().__init__(*args)

        self.trigger_func = trigger_func
        self.clicked.connect(trigger_func)
        self.setMaximumHeight(15)
        self.setMaximumWidth(50)
        self.setFont(GUI_settings.font_tiny)


class MediumButton(QtWidgets.QPushButton):

    def __init__(self, *args, trigger_func=None):
        super().__init__(*args)

        self.trigger_func = trigger_func
        self.clicked.connect(trigger_func)
        self.setMaximumHeight(20)
        self.setMaximumWidth(200)
        self.setFont(GUI_settings.font_tiny)


class SetButton(QtWidgets.QPushButton):

    def __init__(self, obj, trigger_func=None):
        super().__init__('Set', obj)

        self.trigger_func = trigger_func
        self.clicked.connect(trigger_func)
        self.setMaximumHeight(15)
        self.setMaximumWidth(30)
        self.setFont(GUI_settings.font_tiny)


class SetButtonLayout(QtWidgets.QHBoxLayout):

    def __init__(self, *args, obj=None, trigger_func=None, label=None):
        super().__init__(*args)

        self.addWidget(SetButton(obj, trigger_func))
        self.addWidget(label)
        self.addStretch()


class GroupBox(QtWidgets.QGroupBox):

    def __init__(self, title, menu_action=None):
        super().__init__(title)

        self.menu_action = menu_action
        self.setStyleSheet('QGroupBox { font-weight: bold; } ')
        self.visible = True

    def set_state(self):
        if self.visible:
            self.menu_action.blockSignals(True)
            self.menu_action.setChecked(True)
            self.menu_action.blockSignals(False)
            for widget in self.children():
                if widget is not None and widget.isWidgetType():
                    widget.show()
        else:
            self.menu_action.blockSignals(True)
            self.menu_action.setChecked(False)
            self.menu_action.blockSignals(False)
            for widget in self.children():
                if widget is not None and widget.isWidgetType():
                    widget.hide()

    def set_visible(self):
        self.visible = True
        self.set_state()

    def set_hidden(self):
        self.visible = False
        self.set_state()

    def toggle(self):
        self.visible = not self.visible
        self.set_state()

    def mouseDoubleClickEvent(self, *args):
        self.toggle()

