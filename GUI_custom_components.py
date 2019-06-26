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

class arrow():

    def make_arrow_obj(self, p1, p2, r, scale_factor, consistent):

        r_2 = QtCore.QPointF(2 * scale_factor * p2[0], 2 * scale_factor * p2[1])
        r_1 = QtCore.QPointF(2 * scale_factor * p1[0], 2 * scale_factor * p1[1])

        r_vec = r_2 - r_1
        r_mag = np.sqrt((r_2.x() - r_1.x()) ** 2 + (r_2.y() - r_1.y()) ** 2)
        factor = r / (r_mag * 2)

        k_1 = r_1 + factor * r_vec
        k_2 = r_1 + (1 - factor) * r_vec

        theta = np.pi / 4
        self.red_pen.setWidth(3)

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

        if consistent:
            pen = self.pen_connection
            brush = self.brush_connection
        else:
            pen = self.red_pen
            brush = self.red_brush

        line.setPen(pen)
        head_1.setPen(pen)
        head_2.setBrush(brush)

        return line, head_2




