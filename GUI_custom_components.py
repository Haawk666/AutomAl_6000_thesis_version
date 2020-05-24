# By Haakon Tvedt @ NTNU
"""Module container for low-level custom GUI-elements"""

# Program imports:
import GUI_settings
# External imports:
from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import logging
# Instantiate logger:
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ----------
# Graphic elements:
# ----------


class InteractiveColumn(QtWidgets.QGraphicsEllipseItem):

    """A general interactive graphical element that is meant to represent atomic columns in the GUI.

    Inherits PyQt5.QtWidgets.QGraphicsEllipseItem()
    """

    def __init__(self, ui_obj=None, vertex=None, scale_factor=1, r=5, movable=True, selectable=True):

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
        if vertex is not None:
            self.r = r
            self.i = vertex.i
            self.center_coor = vertex.im_pos()
        else:
            self.r = 5
            self.i = -1
            self.center_coor = (0, 0)
        self.scale_factor = scale_factor
        self.vertex = vertex

        super().__init__(0, 0, self.r, self.r)

        if movable:
            offset = self.boundingRect().center()
            self.moveBy(self.center_coor[0] - offset.x(), self.center_coor[1] - offset.y())
        if selectable:
            self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)

    def get_vertex_pos(self):
        offset = self.boundingRect().center()
        pos = (self.x() + offset.x(), self.y() + offset.y())
        return pos

    def mouseReleaseEvent(self, event: 'QtWidgets.QGraphicsEllipseItem.mouseReleaseEvent'):
        """Pass a mouse release event on to the ui_obj reference object"""
        self.ui_obj.column_selected(self.i)

    def mousePressEvent(self, *args, **kwargs):
        pass


class InteractivePosColumn(InteractiveColumn):

    def __init__(self, ui_obj=None, vertex=None, scale_factor=1, movable=True, r=16):
        """Initialize a positional interactive column.

        Inherits GUI_custom_components.InteractiveColumn. Is used to highlight atomic positions."""
        super().__init__(ui_obj=ui_obj, vertex=vertex, scale_factor=scale_factor, movable=movable, r=r)

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

    def __init__(
            self, ui_obj=None, vertex=None, scale_factor=1, movable=True, selectable=True, r=16,
            pen=QtGui.QPen(QtCore.Qt.red), brush=QtGui.QBrush(QtCore.Qt.red)
    ):
        """Initialize a positional interactive column.

        Inherits GUI_custom_components.InteractiveColumn. Is used to highlight atomic positions."""
        super().__init__(ui_obj=ui_obj, vertex=vertex, scale_factor=scale_factor, movable=movable, selectable=selectable, r=r)

        self.pen = pen
        self.brush = brush
        self.selected_pen = QtGui.QPen(QtCore.Qt.darkCyan)
        self.selected_pen.setWidth(6)
        self.selected_brush = QtGui.QBrush(QtCore.Qt.darkCyan)

        self.set_style()

    def set_style(self):
        """Set the appearance of the shape"""
        if self.i == self.ui_obj.selected_column:
            self.setPen(self.selected_pen)
            self.setBrush(self.selected_brush)
        else:
            self.setPen(self.pen)
            self.setBrush(self.brush)
        if self.vertex is not None:
            if not self.vertex.show_in_overlay:
                self.hide()
            else:
                self.show()
        else:
            self.show()


class InteractiveGraphColumn(InteractiveColumn):

    def __init__(self, ui_obj=None, vertex=None, scale_factor=1, r=16):
        """Initialize a positional interactive column.

        Inherits GUI_custom_components.InteractiveColumn. Is used to highlight atomic positions."""
        super().__init__(ui_obj=ui_obj, vertex=vertex, scale_factor=scale_factor, movable=True, selectable=True, r=r)

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
        if self.vertex.zeta == 0:
            self.setBrush(self.level_0_brush)
        else:
            self.setBrush(self.level_1_brush)


class Arrow(QtWidgets.QGraphicsItemGroup):

    def __init__(self, *args, i=0, j=1, p1=(0, 0), p2=(1, 1), r=1, scale_factor=1, dual_arc=False, co_planar=False):
        super().__init__(*args)

        self.inconsistent_pen = GUI_settings.pen_inconsistent_edge
        self.dislocation_pen = GUI_settings.pen_dislocation_edge
        self.normal_pen = GUI_settings.pen_edge

        self.dual_arc = dual_arc
        self.co_planar = co_planar
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
        if self.dual_arc and not self.co_planar:
            self.childItems()[0].setPen(self.normal_pen)
            self.childItems()[0].show()
            self.childItems()[1].hide()
        elif self.dual_arc and self.co_planar:
            self.childItems()[0].setPen(self.dislocation_pen)
            self.childItems()[0].show()
            self.childItems()[1].setPen(self.dislocation_pen)
            self.childItems()[1].show()
        else:
            self.childItems()[0].setPen(self.inconsistent_pen)
            self.childItems()[0].show()
            self.childItems()[1].setPen(self.inconsistent_pen)
            self.childItems()[1].show()

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


class DistanceArrow(QtWidgets.QGraphicsItemGroup):

    def __init__(self, *args, color=None, i=0, j=1, p1=(0, 0), p2=(1, 1), r=1, scale_factor=1):
        super().__init__(*args)

        self.inconsistent_pen = GUI_settings.pen_inconsistent_edge
        self.dislocation_pen = GUI_settings.pen_dislocation_edge
        self.normal_pen = GUI_settings.pen_edge

        if color is None:
            self.color = QtGui.QColor(255, 255, 255, 255)
        else:
            self.color = QtGui.QColor(color[0], color[1], color[2], color[3])
        self.pen = QtGui.QPen(self.color)
        self.pen.setWidth(3)

        self.scale_factor = scale_factor
        self.r = r
        self.p1 = p1
        self.p2 = p2
        self.i = i
        self.j = j

        self.make_arrow_obj()
        self.set_style()

    def set_style(self):

            self.childItems()[0].setPen(self.pen)
            self.childItems()[0].show()
            self.show()

    def make_arrow_obj(self):

        line = QtWidgets.QGraphicsLineItem(self.scale_factor * self.p1[0],
                                           self.scale_factor * self.p1[1],
                                           self.scale_factor * self.p2[0],
                                           self.scale_factor * self.p2[1])

        self.addToGroup(line)
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


class MeshDetail(QtWidgets.QGraphicsItemGroup):

    def __init__(self, *args, mesh=None):
        super().__init__(*args)
        self.mesh = mesh
        if self.mesh is not None:
            self.make()

    def make(self):
        num_corners = self.mesh.order
        text = QtWidgets.QGraphicsSimpleTextItem()
        text.setText('{}'.format(num_corners))
        text.setFont(GUI_settings.font_mesh_details)
        if num_corners == 4:
            text.setPen(GUI_settings.pen_boarder)
        else:
            text.setPen(GUI_settings.pen_skinny_red)
        text.setBrush(GUI_settings.brush_black)
        rect = text.boundingRect()
        text.setX(2 * self.mesh.cm[0] - rect.width() / 2)
        text.setY(2 * self.mesh.cm[1] - rect.height() / 2)

        self.addToGroup(text)

        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)

    def mouseReleaseEvent(self, event: 'QtWidgets.QGraphicsItemGroup.mouseReleaseEvent'):
        print('{}'.format(self.mesh.mesh_index))
        # super().__init__()


class RgbaSelector(QtWidgets.QGroupBox):

    def __init__(self, *args, r=100, g=100, b=100, a=100):
        super().__init__(*args)

        self.r_box = QtWidgets.QSpinBox()
        self.r_box.setMaximum(255)
        self.r_box.setMinimum(0)
        if 0 <= r <= 255:
            self.r_box.setValue(r)
        else:
            self.r_box.setValue(100)

        self.g_box = QtWidgets.QSpinBox()
        self.g_box.setMaximum(255)
        self.g_box.setMinimum(0)
        if 0 <= g <= 255:
            self.g_box.setValue(g)
        else:
            self.g_box.setValue(100)

        self.b_box = QtWidgets.QSpinBox()
        self.b_box.setMaximum(255)
        self.b_box.setMinimum(0)
        if 0 <= b <= 255:
            self.b_box.setValue(b)
        else:
            self.b_box.setValue(100)

        self.a_box = QtWidgets.QSpinBox()
        self.a_box.setMaximum(255)
        self.a_box.setMinimum(0)
        if 0 <= a <= 255:
            self.a_box.setValue(a)
        else:
            self.a_box.setValue(100)

        self.lbl_r = QtWidgets.QLabel('R:')
        self.lbl_g = QtWidgets.QLabel('G:')
        self.lbl_b = QtWidgets.QLabel('B:')
        self.lbl_a = QtWidgets.QLabel('A:')

        layout = QtWidgets.QGridLayout()

        layout.addWidget(self.lbl_r, 0, 0)
        layout.addWidget(self.lbl_g, 0, 1)
        layout.addWidget(self.lbl_b, 0, 2)
        layout.addWidget(self.lbl_a, 0, 3)
        layout.addWidget(self.r_box, 1, 0)
        layout.addWidget(self.g_box, 1, 1)
        layout.addWidget(self.b_box, 1, 2)
        layout.addWidget(self.a_box, 1, 3)

        self.setLayout(layout)

        self.setTitle('')

    def get_triple(self):
        triple = (
            self.r_box.value(),
            self.g_box.value(),
            self.b_box.value()
        )
        return triple

    def set_triple(self, triple):
        self.r_box.setValue(triple[0])
        self.g_box.setValue(triple[1])
        self.b_box.setValue(triple[2])


class CenterBox(QtWidgets.QGroupBox):

    def __init__(self, *args, widget=None):
        super().__init__(*args)

        self.widget = widget

        h_layout = QtWidgets.QHBoxLayout()
        h_layout.addStretch()
        h_layout.addWidget(self.widget)
        h_layout.addStretch()

        v_layout = QtWidgets.QVBoxLayout()
        v_layout.addStretch()
        v_layout.addLayout(h_layout)
        v_layout.addStretch()

        self.setLayout(v_layout)

        self.setStyleSheet("border:0;")

        self.widget.setStyleSheet('')


class HCenterBox(QtWidgets.QGroupBox):

    def __init__(self, *args, widget=None):
        super().__init__(*args)

        self.widget = widget

        h_layout = QtWidgets.QHBoxLayout()
        h_layout.addStretch()
        h_layout.addWidget(self.widget)
        h_layout.addStretch()

        self.setLayout(h_layout)

        self.setFlat(True)
        self.setStyleSheet("border:0;")


class VCenterBox(QtWidgets.QGroupBox):

    def __init__(self, *args, widget=None):
        super().__init__(*args)

        self.widget = widget

        v_layout = QtWidgets.QVBoxLayout()
        v_layout.addStretch()
        v_layout.addWidget(self.widget)
        v_layout.addStretch()

        self.setLayout(v_layout)

        self.setFlat(True)
        self.setStyleSheet("border:0;")


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


