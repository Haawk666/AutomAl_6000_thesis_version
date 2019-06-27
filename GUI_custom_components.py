# By Haakon Tvedt @ NTNU
"""Module container for low-level custom GUI-elements"""

from PyQt5 import QtWidgets, QtGui, QtCore
import legacy_GUI
import sys
import numpy as np
import mat_op
import core
import GUI_elements
import utils


class InteractiveColumn(QtWidgets.QGraphicsEllipseItem):

    """A general interactive graphical element that is meant to represent atomic columns in the GUI.

    Inherits PyQt5.QtWidgets.QGraphicsEllipseItem()
    """

    def __init__(self, ui_obj=None, i=-1):

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
        self.r = ui_obj.project_instance.r
        self.i = i
        self.vertex = self.ui_obj.project_instance.grapg.vertices[self.i]
        self.center_coor = self.vertex.real_coor()
        self.center_coor[0] -= self.r
        self.center_coor[1] -= self.r

        super().__init__(0, 0, 2 * self.r, 2 * self.r)

        self.moveBy(self.center_coor[0], self.center_coor[1])

    def mouseReleaseEvent(self, event: 'QtWidgets.QGraphicsEllipseItem.mouseReleaseEvent'):

        """Pass a mouse release event on to the ui_obj reference object"""
        self.ui_obj.column_selected()


class InteractivePosColumn(InteractiveColumn):

    def __init__(self, *args):
        """Initialize a positional interactive column.

        Inherits GUI_custom_components.InteractiveColumn. Is used to highlight atomic positions."""
        super().__init__(*args)

        self.selected_pen = QtGui.QPen(QtCore.Qt.yellow)
        self.selected_pen.setWidth(3)
        self.transparent_brush = QtGui.QBrush(QtCore.Qt.transparent)
        self.unselected_pen = QtGui.QPen(QtCore.Qt.red)
        self.unselected_pen.setWidth(1)
        self.hidden_pen = QtGui.QPen(QtCore.Qt.darkRed)
        self.hidden_pen.setWidth(1)

        self.set_style()

    def set_style(self):
        """Set the appearance of the shape"""
        if self.i == self.ui_obj.selected_column:
            self.setPen(self.selected_pen)
        else:
            if self.vertex.show_in_overlay:
                self.setPen(self.hidden_pen)
            else:
                self.setPen(self.unselected_pen)
        self.setBrush(self.transparent_brush)


class InteractiveOverlayColumn(InteractiveColumn):

    def __init__(self, *args):
        """Initialize a positional interactive column.

        Inherits GUI_custom_components.InteractiveColumn. Is used to highlight atomic positions."""
        super().__init__(*args)

        self.brush_black = QtGui.QBrush(QtCore.Qt.black)

        self.pen_al = QtGui.QPen(QtCore.Qt.green)
        self.pen_al.setWidth(5)
        self.brush_al = QtGui.QBrush(QtCore.Qt.green)

        self.pen_mg = QtGui.QPen(QtGui.QColor(143, 0, 255))
        self.pen_mg.setWidth(5)
        self.brush_mg = QtGui.QBrush(QtGui.QColor(143, 0, 255))

        self.pen_si = QtGui.QPen(QtCore.Qt.red)
        self.pen_si.setWidth(5)
        self.brush_si = QtGui.QBrush(QtCore.Qt.red)

        self.pen_cu = QtGui.QPen(QtCore.Qt.yellow)
        self.pen_cu.setWidth(5)
        self.brush_cu = QtGui.QBrush(QtCore.Qt.yellow)

        self.pen_zn = QtGui.QPen(QtGui.QColor(100, 100, 100))
        self.pen_zn.setWidth(5)
        self.brush_zn = QtGui.QBrush(QtGui.QColor(100, 100, 100))

        self.pen_ag = QtGui.QPen(QtGui.QColor(200, 200, 200))
        self.pen_ag.setWidth(5)
        self.brush_ag = QtGui.QBrush(QtGui.QColor(200, 200, 200))

        self.pen_un = QtGui.QPen(QtCore.Qt.blue)
        self.pen_un.setWidth(5)
        self.brush_un = QtGui.QBrush(QtCore.Qt.blue)

        self.selected_pen = QtGui.QPen(QtCore.Qt.darkCyan)
        self.selected_pen.setWidth(6)
        self.brush_selected = QtGui.QBrush(QtCore.Qt.darkCyan)

        self.set_style()

    def set_style(self):
        """Set the appearance of the shape"""
        if self.i == self.ui_obj.selected_column:
            self.setPen(self.selected_pen)
            self.setBrush(self.brush_selected)
        else:
            self.setBrush(self.brush_black)
            if self.vertex.h_index == 0:
                self.setPen(self.pen_cu)
                if self.vertex.level == 0:
                    self.setBrush(self.brush_cu)
            elif self.vertex.h_index == 1:
                self.setPen(self.pen_si)
                if self.vertex.level == 0:
                    self.setBrush(self.brush_si)
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

        self.selected_pen = QtGui.QPen(QtCore.Qt.blue)
        self.selected_pen.setWidth(3)
        self.transparent_brush = QtGui.QBrush(QtCore.Qt.white)
        self.opaque_brush = QtGui.QBrush(QtCore.Qt.black)
        self.selected_brush = QtGui.QBrush(QtCore.Qt.blue)
        self.unselected_pen = QtGui.QPen(QtCore.Qt.black)
        self.unselected_pen.setWidth(1)

        self.set_style()

    def set_style(self):
        """Set the appearance of the shape"""
        if self.i == self.ui_obj.selected_column:
            self.setPen(self.selected_pen)
            self.setBrush(self.selected_brush)
        else:
            self.setPen(self.unselected_pen)
            if self.vertex.level == 0:
                self.setBrush(self.transparent_brush)
            else:
                self.setBrush(self.opaque_brush)


class Arrow:

    def __init__(self, p1, p2, r, scale_factor, consistent, dislocation):

        self.inconsistent_pen = QtGui.QPen(QtCore.Qt.red)
        self.inconsistent_pen.setWidth(3)
        self.dislocation_pen = QtGui.QPen(QtCore.Qt.blue)
        self.dislocation_pen.setWidth(3)
        self.normal_pen = QtGui.QPen(QtCore.Qt.black)
        self.normal_pen.setWidth(1)

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

        r_2 = QtCore.QPointF(2 * scale_factor * p2[0], 2 * scale_factor * p2[1])
        r_1 = QtCore.QPointF(2 * scale_factor * p1[0], 2 * scale_factor * p1[1])

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

        line = QtWidgets.QGraphicsLineItem(2 * scale_factor * p1[0],
                                           2 * scale_factor * p1[1],
                                           2 * scale_factor * p2[0],
                                           2 * scale_factor * p2[1])
        head_1 = QtWidgets.QGraphicsPolygonItem(poly_1)
        head_2 = QtWidgets.QGraphicsPolygonItem(poly_2)

        self.arrow = line, head_2




