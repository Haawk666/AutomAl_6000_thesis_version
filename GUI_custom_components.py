# By Haakon Tvedt @ NTNU
"""Module container for low-level custom GUI-elements"""

from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import GUI_settings


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
        self.vertex = self.ui_obj.project_instance.graph.vertices[self.i]
        self.center_coor = self.vertex.real_coor()
        self.center_coor = scale_factor * self.center_coor[0] - np.round(self.r / 2), scale_factor * self.center_coor[1] - np.round(self.r / 2)

        super().__init__(0, 0, self.r, self.r)

        self.moveBy(self.center_coor[0], self.center_coor[1])
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)

    def mouseReleaseEvent(self, event: 'QtWidgets.QGraphicsEllipseItem.mouseReleaseEvent'):

        """Pass a mouse release event on to the ui_obj reference object"""
        self.ui_obj.column_selected(self.i)


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


class Arrow:

    def __init__(self, p1, p2, r, scale_factor, consistent, dislocation):

        self.inconsistent_pen = GUI_settings.pen_inconsistent_edge
        self.dislocation_pen = GUI_settings.pen_dislocation_edge
        self.normal_pen = GUI_settings.pen_edge

        self.arrow = None, None
        self.make_arrow_obj(p1, p2, r, scale_factor)
        self.set_style(consistent, dislocation)

    def set_style(self, consistent, dislocation):

        if not consistent:
            self.arrow[0].setPen(self.inconsistent_pen)
            self.arrow[1].setPen(self.inconsistent_pen)
            self.arrow[1].show()
        elif dislocation:
            self.arrow[0].setPen(self.dislocation_pen)
            self.arrow[1].setPen(self.dislocation_pen)
            self.arrow[1].show()
        else:
            self.arrow[0].setPen(self.normal_pen)
            self.arrow[1].setPen(self.normal_pen)
            self.arrow[1].hide()

    def make_arrow_obj(self, p1, p2, r, scale_factor):

        r_2 = QtCore.QPointF(scale_factor * p2[0], scale_factor * p2[1])
        r_1 = QtCore.QPointF(scale_factor * p1[0], scale_factor * p1[1])

        r_vec = r_2 - r_1
        r_mag = np.sqrt((r_2.x() - r_1.x()) ** 2 + (r_2.y() - r_1.y()) ** 2)
        factor = r / (r_mag * 2)

        k_1 = r_1 + factor * r_vec
        k_2 = r_1 + (1 - factor) * r_vec

        theta = np.pi / 4

        l_1 = factor * QtCore.QPointF(r_vec.x() * np.cos(theta) + r_vec.y() * np.sin(theta),
                                      - r_vec.x() * np.sin(theta) + r_vec.y() * np.cos(theta))
        l_1 = k_1 + l_1

        l_2 = factor * QtCore.QPointF(r_vec.x() * np.cos(-theta) + r_vec.y() * np.sin(-theta),
                                      - r_vec.x() * np.sin(-theta) + r_vec.y() * np.cos(-theta))
        l_2 = k_1 + l_2

        l_3 = - factor * QtCore.QPointF(r_vec.x() * np.cos(theta) + r_vec.y() * np.sin(theta),
                                        - r_vec.x() * np.sin(theta) + r_vec.y() * np.cos(theta))
        l_3 = k_2 + l_3

        l_4 = - factor * QtCore.QPointF(r_vec.x() * np.cos(-theta) + r_vec.y() * np.sin(-theta),
                                        - r_vec.x() * np.sin(-theta) + r_vec.y() * np.cos(-theta))
        l_4 = k_2 + l_4

        tri_1 = (k_1, l_1, l_2)
        tri_2 = (k_2, l_3, l_4)

        poly_1 = QtGui.QPolygonF(tri_1)
        poly_2 = QtGui.QPolygonF(tri_2)

        line = QtWidgets.QGraphicsLineItem(scale_factor * p1[0],
                                           scale_factor * p1[1],
                                           scale_factor * p2[0],
                                           scale_factor * p2[1])
        head_1 = QtWidgets.QGraphicsPolygonItem(poly_1)
        head_2 = QtWidgets.QGraphicsPolygonItem(poly_2)

        self.arrow = line, head_2


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

    def __init__(self, title):
        super().__init__(title)

        self.setStyleSheet('QGroupBox { font-weight: bold; } ')

        self.shadow_box = QtWidgets.QGroupBox(title)
        self.shadow_box.setStyleSheet('QGroupBox { font-weight: bold; } ')
        self.shadow_box.hide()

        self.visible = True
        self.show()

    def toggle(self):

        if self.visible:
            self.visible = False
            self.hide()
            self.shadow_box.show()
        else:
            self.visible = True
            self.show()
            self.shadow_box.hide()

    def set_visible(self):

        self.visible = True
        self.show()
        self.shadow_box.hide()

    def set_hidden(self):

        self.visible = False
        self.hide()
        self.shadow_box.show()








