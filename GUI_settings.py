# By Haakon Tvedt @ NTNU
"""Module container for style settings of the GUI"""


from PyQt5 import QtGui, QtCore

font_tiny = QtGui.QFont()
font_tiny.setPixelSize(9)

pen_boarder = QtGui.QPen(QtCore.Qt.black)
pen_boarder.setWidth(1)

brush_black = QtGui.QBrush(QtCore.Qt.black)

pen_al = QtGui.QPen(QtCore.Qt.green)
pen_al.setWidth(5)
brush_al = QtGui.QBrush(QtCore.Qt.green)

pen_mg = QtGui.QPen(QtGui.QColor(143, 0, 255))
pen_mg.setWidth(5)
brush_mg = QtGui.QBrush(QtGui.QColor(143, 0, 255))

pen_si = QtGui.QPen(QtCore.Qt.red)
pen_si.setWidth(5)
brush_si = QtGui.QBrush(QtCore.Qt.red)

pen_cu = QtGui.QPen(QtCore.Qt.yellow)
pen_cu.setWidth(5)
brush_cu = QtGui.QBrush(QtCore.Qt.yellow)

pen_zn = QtGui.QPen(QtGui.QColor(100, 100, 100))
pen_zn.setWidth(5)
brush_zn = QtGui.QBrush(QtGui.QColor(100, 100, 100))

pen_ag = QtGui.QPen(QtGui.QColor(200, 200, 200))
pen_ag.setWidth(5)
brush_ag = QtGui.QBrush(QtGui.QColor(200, 200, 200))

pen_un = QtGui.QPen(QtCore.Qt.blue)
pen_un.setWidth(5)
brush_un = QtGui.QBrush(QtCore.Qt.blue)

pen_selected_1 = QtGui.QPen(QtCore.Qt.darkCyan)
pen_selected_1.setWidth(6)
brush_selected_1 = QtGui.QBrush(QtCore.Qt.darkCyan)

pen_selected_2 = QtGui.QPen(QtCore.Qt.yellow)
pen_selected_2.setWidth(3)
brush_selected_2 = QtGui.QBrush(QtCore.Qt.transparent)

pen_atom_pos = QtGui.QPen(QtCore.Qt.red)
pen_atom_pos.setWidth(1)
brush_atom_pos = QtGui.QBrush(QtCore.Qt.transparent)

pen_atom_pos_hidden = QtGui.QPen(QtCore.Qt.red)
pen_atom_pos_hidden.setWidth(1)
brush_atom_pos_hidden = QtGui.QBrush(QtCore.Qt.transparent)

pen_graph = QtGui.QPen(QtCore.Qt.black)
pen_graph.setWidth(1)
brush_graph_0 = QtGui.QBrush(QtCore.Qt.black)
brush_graph_1 = QtGui.QBrush(QtCore.Qt.white)

pen_edge = QtGui.QPen(QtCore.Qt.black)
pen_edge.setWidth(1)
pen_inconsistent_edge = QtGui.QPen(QtCore.Qt.red)
pen_inconsistent_edge.setWidth(3)
pen_dislocation_edge = QtGui.QPen(QtCore.Qt.blue)
pen_dislocation_edge.setWidth(3)

