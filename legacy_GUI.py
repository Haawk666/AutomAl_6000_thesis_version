# UI.py
# -------------------------------------------
# This file contains the user interface functionality. MainUi calls on SuchSoftware in core.py, but SuchSoftware never
# calls on MainUi. This is an attempt to keep SuchSoftware independent and thus callable in other hypothetical contexts,
# all though this inhibits the possibility to display "progress information" to the user when SuchSoftware is running
# tasks...

from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import numpy as np
import numpy.core._dtype_ctypes  # This is needed because of a bug in pyinstaller
import mat_op
import core
import legacy_GUI_elements
import utils
import dev_module


class MainUI(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.version = [0, 0, 0]

        # Initialize in an 'empty state'
        self.project_instance = core.SuchSoftware('empty')
        self.project_loaded = False
        self.savefile = None
        self.control_instance = None
        self.selected_column = -1
        self.previous_selected_column = -1

        self.previous_pos_obj = None
        self.previous_overlay_obj = None
        self.previous_vertex_obj = None

        # Lists of graphical objects
        dummy_instance = legacy_GUI_elements.InteractivePosColumn(0, 0, 0, 0)
        self.pos_objects = np.ndarray([1], dtype=type(dummy_instance))
        dummy_instance = legacy_GUI_elements.InteractiveOverlayColumn(0, 0, 0, 0)
        self.overlay_objects = np.ndarray([1], dtype=type(dummy_instance))
        dummy_instance = legacy_GUI_elements.InteractiveGraphVertex(0, 0, 0, 0)
        self.vertex_objects = np.ndarray([1], dtype=type(dummy_instance))
        self.sub_vertex_objects = np.ndarray([1], dtype=type(dummy_instance))

        self.boarder_line_objects = np.ndarray([1], dtype=type(QtWidgets.QGraphicsLineItem()))
        self.boarder_line_objects[0] = QtWidgets.QGraphicsLineItem(1.0, 2.0, 3.0, 4.0)
        self.eye_line_objects = np.ndarray([1], dtype=type(QtWidgets.QGraphicsLineItem()))
        self.eye_line_objects[0] = QtWidgets.QGraphicsLineItem(1.0, 2.0, 3.0, 4.0)
        self.flower_line_objects = np.ndarray([1], dtype=type(QtWidgets.QGraphicsLineItem()))
        self.flower_line_objects[0] = QtWidgets.QGraphicsLineItem(1.0, 2.0, 3.0, 4.0)
        self.neighbour_line_objects = np.ndarray([1], dtype=type(QtWidgets.QGraphicsLineItem()))
        self.neighbour_line_objects[0] = QtWidgets.QGraphicsLineItem(1.0, 2.0, 3.0, 4.0)

        # Some predefined pens and brushes
        self.red_brush = QtGui.QBrush(QtCore.Qt.red)
        self.yellow_brush = QtGui.QBrush(QtCore.Qt.yellow)
        self.black_pen = QtGui.QPen(QtCore.Qt.black)
        self.red_pen = QtGui.QPen(QtCore.Qt.red)
        self.dark_red_pen = QtGui.QPen(QtCore.Qt.darkRed)
        self.green_pen = QtGui.QPen(QtCore.Qt.green)
        self.blue_pen = QtGui.QPen(QtCore.Qt.blue)
        self.yellow_pen = QtGui.QPen(QtCore.Qt.yellow)
        self.yellow_pen.setWidth(3)

        # Species pens and brushes
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
        self.pen_problem = QtGui.QPen(QtGui.QColor(200, 300, 40))
        self.pen_problem.setWidth(6)
        self.pen_boarder = QtGui.QPen(QtCore.Qt.white)
        self.pen_boarder.setWidth(6)
        self.pen_structure = QtGui.QPen(QtCore.Qt.white)
        self.pen_structure.setWidth(3)
        self.pen_connection = QtGui.QPen(QtCore.Qt.black)
        self.pen_connection.setWidth(1)
        self.brush_connection = QtGui.QBrush(QtCore.Qt.black)

        # Atomic graph pens and brushes
        self.pen_consistent = QtGui.QPen(QtCore.Qt.green)
        self.pen_consistent.setWidth(4)
        self.pen_inconsistent_popular = QtGui.QPen(QtCore.Qt.yellow)
        self.pen_inconsistent_popular.setWidth(4)
        self.pen_inconsistent_unpopular = QtGui.QPen(QtCore.Qt.red)
        self.pen_inconsistent_unpopular.setWidth(4)
        self.brush_level_0 = QtGui.QBrush(QtCore.Qt.white)
        self.brush_level_1 = QtGui.QBrush(QtCore.Qt.black)
        self.brush_background_black = QtGui.QBrush(QtCore.Qt.black)
        self.brush_background_grey = QtGui.QBrush(QtGui.QColor(80, 80, 80))

        # A header font
        self.font_header = QtGui.QFont()
        self.font_header.setBold(True)

        # A tiny font
        self.font_tiny = QtGui.QFont()
        self.font_tiny.setPixelSize(9)

        # Create menu bar
        legacy_GUI_elements.MenuBar(self.menuBar(), self)

        # Generate elements
        self.setWindowTitle(
            'AACC - Automatic Atomic Column Characterizer - By Haakon Tvedt @ NTNU. Version alpha 2')
        self.resize(1500, 900)
        self.move(50, 30)
        self.statusBar().showMessage('Ready')

        # Generate central widget
        self.graphic = QtGui.QPixmap('Images\\no_image.png')
        self.graphicScene_1 = QtWidgets.QGraphicsScene()
        self.graphicScene_1.addPixmap(self.graphic)
        self.graphicScene_2 = QtWidgets.QGraphicsScene()
        self.graphicScene_2.addPixmap(self.graphic)
        self.graphicScene_3 = QtWidgets.QGraphicsScene()
        self.graphicScene_3.addPixmap(self.graphic)
        self.graphicScene_4 = QtWidgets.QGraphicsScene()
        self.graphicScene_4.addPixmap(self.graphic)
        self.graphicScene_5 = QtWidgets.QGraphicsScene()
        self.graphicScene_5.addPixmap(self.graphic)
        self.graphicScene_6 = QtWidgets.QGraphicsScene()
        self.graphicScene_6.addPixmap(self.graphic)
        self.graphicScene_7 = QtWidgets.QGraphicsScene()
        self.graphicScene_7.addPixmap(self.graphic)
        self.graphicsView_1 = legacy_GUI_elements.ZGraphicsView(self.graphicScene_1, self.key_press)
        self.graphicsView_2 = legacy_GUI_elements.ZGraphicsView(self.graphicScene_2, self.key_press)
        self.graphicsView_3 = legacy_GUI_elements.ZGraphicsView(self.graphicScene_3, self.key_press)
        self.graphicsView_4 = legacy_GUI_elements.ZGraphicsView(self.graphicScene_4, self.key_press)
        self.graphicsView_5 = legacy_GUI_elements.ZGraphicsView(self.graphicScene_5, self.key_press)
        self.graphicsView_6 = legacy_GUI_elements.ZGraphicsView(self.graphicScene_6, self.key_press)
        self.graphicsView_7 = legacy_GUI_elements.ZGraphicsView(self.graphicScene_7, self.key_press)

        self.tabs = QtWidgets.QTabWidget()

        self.tab_1 = self.tabs.addTab(self.graphicsView_1, 'Raw image')
        self.tab_2 = self.tabs.addTab(self.graphicsView_2, 'Atomic positions')
        self.tab_3 = self.tabs.addTab(self.graphicsView_3, 'Overlay composition')
        self.tab_6 = self.tabs.addTab(self.graphicsView_6, 'Atomic graph')
        self.tab_7 = self.tabs.addTab(self.graphicsView_7, 'Atomic sub-graph')
        self.tab_4 = self.tabs.addTab(self.graphicsView_4, 'Search matrix')
        self.tab_5 = self.tabs.addTab(self.graphicsView_5, 'FFT image')

        self.setCentralWidget(self.tabs)

        # Generate control window

        self.control_window = legacy_GUI_elements.ControlWindow(obj=self)

        self.info_display_area = QtWidgets.QScrollArea()
        self.info_display_area.setWidget(self.control_window)
        self.info_display_area.setWidgetResizable(True)
        self.info_display_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

        self.info_display = QtWidgets.QDockWidget()
        self.info_display.setWidget(self.info_display_area)
        self.info_display.setWindowTitle('Control window')
        self.info_display.setMinimumWidth(300)

        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.info_display)

        self.terminal_window = legacy_GUI_elements.Terminal()

        self.btn_save_log = QtWidgets.QPushButton('Save log', self)
        self.btn_save_log.clicked.connect(self.save_log_trigger)
        self.btn_save_log.setMaximumHeight(15)
        self.btn_save_log.setMaximumWidth(50)
        self.btn_save_log.setFont(self.font_tiny)

        self.btn_clear_log = QtWidgets.QPushButton('Clear log', self)
        self.btn_clear_log.clicked.connect(self.clear_log_trigger)
        self.btn_clear_log.setMaximumHeight(15)
        self.btn_clear_log.setMaximumWidth(50)
        self.btn_clear_log.setFont(self.font_tiny)

        self.terminal_btns_layout = QtWidgets.QHBoxLayout()
        self.terminal_btns_layout.addWidget(self.btn_save_log)
        self.terminal_btns_layout.addWidget(self.btn_clear_log)
        self.terminal_btns_layout.addStretch()

        self.terminal_display_layout = QtWidgets.QVBoxLayout()
        self.terminal_display_layout.addLayout(self.terminal_btns_layout)
        self.terminal_display_layout.addWidget(self.terminal_window)

        self.terminal_display_area = QtWidgets.QScrollArea()
        self.terminal_display_area.setLayout(self.terminal_display_layout)
        self.terminal_display_area.setWidgetResizable(True)
        self.terminal_display_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.terminal_display = QtWidgets.QDockWidget()
        self.terminal_display.setWidget(self.terminal_display_area)
        self.terminal_display.setWindowTitle('Terminal stream')
        self.terminal_display.setMinimumWidth(400)

        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.terminal_display)

        self.toggle_debug_mode_trigger(False)
        self.debug_mode = False
        self.deselect_trigger()

        # Display
        self.show()

        # Intro
        self.report('Welcome to AACC by Haakon Tvedt', force=True)
        self.report('    GUI version: {}.{}.{}'.format(self.version[0], self.version[1], self.version[2]), force=True)
        self.report('    core version: {}.{}.{}'.format(core.SuchSoftware.version[0],
                                                        core.SuchSoftware.version[1],
                                                        core.SuchSoftware.version[2]), force=True)

    def keyPressEvent(self, event):

        if event.key() == QtCore.Qt.Key_X:
            if self.tabs.currentIndex() == 6:
                self.tabs.setCurrentIndex(0)
            else:
                self.tabs.setCurrentIndex(self.tabs.currentIndex() + 1)
        if event.key() == QtCore.Qt.Key_Z:
            if self.tabs.currentIndex() == 0:
                self.tabs.setCurrentIndex(6)
            else:
                self.tabs.setCurrentIndex(self.tabs.currentIndex() - 1)

    def key_press(self, key):
        if self.project_loaded and not self.selected_column == -1:
            if self.tabs.currentIndex() == 0:
                pass
            if self.tabs.currentIndex() == 1 or self.tabs.currentIndex() == 2 or self.tabs.currentIndex() == 3:
                if key == QtCore.Qt.Key_1:
                    self.set_species(3)
                elif key == QtCore.Qt.Key_2:
                    self.set_species(5)
                elif key == QtCore.Qt.Key_3:
                    self.set_species(0)
                elif key == QtCore.Qt.Key_4:
                    self.set_species(1)
                elif key == QtCore.Qt.Key_Plus:
                    if self.project_instance.graph.vertices[self.selected_column].level == 0:
                        self.set_level(1)
                    elif self.project_instance.graph.vertices[self.selected_column].level == 1:
                        self.set_level(2)
                    elif self.project_instance.graph.vertices[self.selected_column].level == 2:
                        self.set_level(0)
                    else:
                        self.report('Something is wrong! Could not set column level...', force=True)
                elif key == QtCore.Qt.Key_W and self.control_window.chb_move.isChecked():
                    self.pos_objects[self.selected_column].moveBy(0.0, -1.0)
                    self.overlay_objects[self.selected_column].moveBy(0.0, -1.0)
                    self.vertex_objects[self.selected_column].moveBy(0.0, -1.0)
                elif key == QtCore.Qt.Key_S and self.control_window.chb_move.isChecked():
                    self.pos_objects[self.selected_column].moveBy(0.0, 1.0)
                    self.overlay_objects[self.selected_column].moveBy(0.0, 1.0)
                    self.vertex_objects[self.selected_column].moveBy(0.0, 1.0)
                elif key == QtCore.Qt.Key_A and self.control_window.chb_move.isChecked():
                    self.pos_objects[self.selected_column].moveBy(-1.0, 0.0)
                    self.overlay_objects[self.selected_column].moveBy(-1.0, 0.0)
                    self.vertex_objects[self.selected_column].moveBy(-1.0, 0.0)
                elif key == QtCore.Qt.Key_D and self.control_window.chb_move.isChecked():
                    self.pos_objects[self.selected_column].moveBy(1.0, 0.0)
                    self.overlay_objects[self.selected_column].moveBy(1.0, 0.0)
                    self.vertex_objects[self.selected_column].moveBy(1.0, 0.0)
                elif key == QtCore.Qt.Key_V:
                    self.project_instance.graph.vertices[self.selected_column].flag_2 = True
            if self.tabs.currentIndex() == 4:
                pass
            if self.tabs.currentIndex() == 5:
                pass
            if self.tabs.currentIndex() == 6:
                pass

    def receive_console_output(self, string, update):
        if not string == '':
            self.terminal_window.appendPlainText(string)
            self.terminal_window.repaint()
        else:
            self.terminal_window.repaint()
        if update:
            self.update_display()

    def report(self, string, force=False):
        if self.debug_mode or force:
            self.terminal_window.appendPlainText('GUI: ' + string)

    # Menu triggers:
    def new_trigger(self):

        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Select dm3', '')
        if filename[0]:
            self.statusBar().showMessage('Working...')
            self.project_instance = core.SuchSoftware(filename[0], self.receive_console_output)
            self.control_instance = None
            self.project_loaded = True
            self.update_display()
            self.report('Generated instance from {}'.format(filename[0]), force=True)
        else:
            self.statusBar().showMessage('Ready')

    def open_trigger(self):

        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '')
        if filename[0]:
            self.statusBar().showMessage('Working...')
            self.project_instance = core.SuchSoftware.load(filename[0])
            if self.project_instance is not None:
                self.project_instance.debug_obj = self.receive_console_output
                if self.control_window.debug_box.isVisible():
                    self.project_instance.debug_mode = True
                else:
                    self.project_instance.debug_mode = False
                self.control_instance = None
                self.project_loaded = True
                self.savefile = filename[0]
                self.update_display()
                self.report('Loaded {}'.format(filename[0]), force=True)
            else:
                self.report('File was not loaded. Something must have gone wrong!', force=True)
        else:
            self.statusBar().showMessage('Ready')

    def save_trigger(self):

        if self.savefile is None:
            filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', '')
        else:
            filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', self.savefile)

        if filename[0]:
            self.statusBar().showMessage('Working...')
            self.report('Saving project', force=True)
            self.project_instance.save(filename[0])
            self.update_display()
            self.report('Saved project to {}'.format(filename[0]), force=True)
        else:
            self.statusBar().showMessage('Ready')

    def save_log_trigger(self):
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', '')
        if filename[0]:
            self.statusBar().showMessage('Working...')
            self.report('Saving log file...', force=True)
            string = self.terminal_window.toPlainText()
            with open(filename[0], 'w') as f:
                for line in iter(string.splitlines()):
                    f.write(line)
            f.close()
            self.report('Saved log to {}'.format(filename[0]), force=True)

    def clear_log_trigger(self):
        self.terminal_window.clear()

    def gen_sub_graph(self):
        self.graphicScene_7 = QtWidgets.QGraphicsScene()
        self.draw_atomic_sub_graph(2)
        self.graphicsView_7.setScene(self.graphicScene_7)

    def close_trigger(self):

        self.statusBar().showMessage('Working...')
        self.cancel_move_trigger()
        self.deselect_trigger()
        self.control_window.empty_display()
        self.report('Closed project', force=True)

    def exit_trigger(self):
        self.close()

    def ad_hoc_trigger(self):

        pass

    def set_threshold_trigger(self):

        if self.project_loaded:
            threshold, ok_pressed = QtWidgets.QInputDialog.getDouble(self, "Set",
                                                                     "Threshold value (decimal between 0 and 1):",
                                                                     self.project_instance.threshold, 0, 1, 5)
            if ok_pressed:
                self.project_instance.threshold = threshold
                self.control_window.lbl_detection_threshold.setText(
                    'Detection threshold value: ' + str(self.project_instance.threshold))

    def set_search_size_trigger(self):

        if self.project_loaded:
            search_size, ok_pressed = QtWidgets.QInputDialog.getInt(self, "Set", "Search size:",
                                                                    self.project_instance.search_size, 0, 100000, 100)
            if ok_pressed:
                self.project_instance.search_size = search_size
                self.control_window.lbl_search_size.setText('Search size: ' + str(self.project_instance.search_size))

    def set_scale_trigger(self):

        if self.project_loaded:
            scale, ok_pressed = QtWidgets.QInputDialog.getDouble(self, "Set", "Image scale (pm/pixel):",
                                                                 self.project_instance.scale, 0, 10000, 4)
            if ok_pressed:
                self.statusBar().showMessage('Working...')
                self.project_instance.scale = scale
                self.project_instance.r = int(100 / scale)
                self.project_instance.overhead = int(6 * (self.r / 10))
                self.control_window.lbl_scale.setText('Scale (pm / pixel): ' + str(self.project_instance.scale))
                self.control_window.lbl_atomic_radii.setText('Approx atomic radii (pixels): ' +
                                                             str(self.project_instance.r))
                self.control_window.lbl_overhead_radii.setText('Overhead (pixels): ' +
                                                               str(self.project_instance.overhead))
                self.project_instance.redraw_search_mat()
                self.update_central_widget()

    def set_alloy_trigger(self):

        if self.project_loaded:

            items = ('Al-Mg-Si-(Cu)', 'Al-Mg-Si')
            item, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Set", "Alloy:", items, 0, False)

            if ok_pressed and item:

                if item == 'Al-Mg-Si-(Cu)':
                    self.project_instance.alloy = 0
                elif item == 'Al-Mg-Si':
                    self.project_instance.alloy = 1
                else:
                    print('Error!')

                self.project_instance.set_alloy_mat()

                self.control_window.lbl_alloy.setText('Alloy: ' + item)

    def set_indices_trigger(self):

        if self.project_loaded and not self.selected_column == -1:

            if self.project_instance.graph.vertices[self.selected_column].neighbour_indices is not None:

                dialog = legacy_GUI_elements.SetIndicesDialog()
                dialog.reference_object(self, self.selected_column)
                dialog.gen_layout()
                dialog.exec_()

    def set_indices_2_trigger(self):

        if self.project_loaded and not self.selected_column == -1:

            if self.project_instance.graph.vertices[self.selected_column].neighbour_indices is not None:

                dialog = legacy_GUI_elements.SetIndicesManuallyDialog()
                dialog.reference_object(self, self.selected_column)
                dialog.gen_layout()
                dialog.exec_()

    def set_start_trigger(self):
        if self.project_instance is not None:
            if not self.selected_column == -1:
                self.project_instance.starting_index = self.selected_column
                self.control_window.lbl_starting_index.setText('Default starting index: ' + str(self.selected_column))
                self.report('Default starting index set to {}'.format(self.selected_column), force=True)

    def set_perturb_mode_trigger(self):

        if self.project_loaded:
            message = QtWidgets.QMessageBox()
            message.setText('Not implemented yet')
            message.exec_()

    def set_std_1_trigger(self):

        if self.project_loaded:
            std_1, ok_pressed = QtWidgets.QInputDialog.getDouble(self, "Set", "Standard deviation 1:",
                                                                 self.project_instance.dist_1_std, 0, 10000, 2)
            if ok_pressed:
                self.project_instance.dist_1_std = std_1
                self.control_window.lbl_std_1.setText('Standard deviation 1: ' + str(self.project_instance.dist_1_std))

    def set_std_2_trigger(self):

        if self.project_loaded:
            std_2, ok_pressed = QtWidgets.QInputDialog.getDouble(self, "Set", "Standard deviation 2:",
                                                                 self.project_instance.dist_2_std, 0, 10000, 2)
            if ok_pressed:
                self.project_instance.dist_2_std = std_2
                self.control_window.lbl_std_2.setText('Standard deviation 2: ' + str(self.project_instance.dist_2_std))

    def set_std_3_trigger(self):

        if self.project_loaded:
            std_3, ok_pressed = QtWidgets.QInputDialog.getDouble(self, "Set", "Standard deviation 3:",
                                                                 self.project_instance.dist_3_std, 0, 10000, 2)
            if ok_pressed:
                self.project_instance.dist_3_std = std_3
                self.control_window.lbl_std_3.setText('Standard deviation 3: ' + str(self.project_instance.dist_3_std))

    def set_std_4_trigger(self):

        if self.project_loaded:
            std_4, ok_pressed = QtWidgets.QInputDialog.getDouble(self, "Set", "Standard deviation 4:",
                                                                 self.project_instance.dist_4_std, 0, 10000, 2)
            if ok_pressed:
                self.project_instance.dist_4_std = std_4
                self.control_window.lbl_std_4.setText('Standard deviation 4: ' + str(self.project_instance.dist_4_std))

    def set_std_5_trigger(self):

        if self.project_loaded:
            std_5, ok_pressed = QtWidgets.QInputDialog.getDouble(self, "Set", "Standard deviation 5:",
                                                                 self.project_instance.dist_5_std, 0, 10000, 2)
            if ok_pressed:
                self.project_instance.dist_5_std = std_5
                self.control_window.lbl_std_5.setText('Standard deviation 5: ' + str(self.project_instance.dist_5_std))

    def set_std_8_trigger(self):

        if self.project_loaded:
            std_8, ok_pressed = QtWidgets.QInputDialog.getDouble(self, "Set", "Standard deviation 8:",
                                                                 self.project_instance.dist_8_std, 0, 10000, 2)
            if ok_pressed:
                self.project_instance.dist_8_std = std_8
                self.control_window.lbl_std_8.setText('Standard deviation 8: ' + str(self.project_instance.dist_8_std))

    def set_cert_threshold_trigger(self):

        if self.project_loaded:
            threshold, ok_pressed = QtWidgets.QInputDialog.getDouble(self, "Set",
                                                                     "Certainty threshold (decimal between 0 and 1):",
                                                                     self.project_instance.certainty_threshold, 0, 1, 3)
            if ok_pressed:
                self.project_instance.certainty_threshold = threshold
                self.control_window.lbl_cert_threshold.setText(
                    'Certainty threshold: ' + str(self.project_instance.certainty_threshold))

    def view_image_title_trigger(self):

        if self.project_loaded:
            self.report(str(self.project_instance.filename_full), force=True)
            message = QtWidgets.QMessageBox()
            message.setText(str(self.project_instance.filename_full))
            message.exec_()

    def show_stats_trigger(self):

        if self.project_loaded:
            self.project_instance.image_report()

    def toggle_image_control_trigger(self, state):

        if state:
            self.control_window.image_box.show()
            self.control_window.image_box_empty.hide()
        else:
            self.control_window.image_box.hide()
            self.control_window.image_box_empty.show()

    def toggle_alg_1_control_trigger(self, state):

        if state:
            self.control_window.alg_1_box.show()
            self.control_window.alg_1_box_empty.hide()
        else:
            self.control_window.alg_1_box.hide()
            self.control_window.alg_1_box_empty.show()

    def toggle_alg_2_control_trigger(self, state):

        if state:
            self.control_window.alg_2_box.show()
            self.control_window.alg_2_box_empty.hide()
        else:
            self.control_window.alg_2_box.hide()
            self.control_window.alg_2_box_empty.show()

    def toggle_column_control_trigger(self, state):

        if state:
            self.control_window.column_box.show()
            self.control_window.column_box_empty.hide()
        else:
            self.control_window.column_box.hide()
            self.control_window.column_box_empty.show()

    def toggle_overlay_control_trigger(self, state):

        if state:
            self.control_window.overlay_box.show()
            self.control_window.overlay_box_empty.hide()
        else:
            self.control_window.overlay_box.hide()
            self.control_window.overlay_box_empty.show()

    def find_column_trigger(self):

        if self.project_loaded:
            index, ok_pressed = QtWidgets.QInputDialog.getInt(self, "Set", "Find column by index:", 0, 0, 100000, 1)
            if ok_pressed:
                if index < self.project_instance.num_columns:
                    self.pos_objects[index].mouseReleaseEvent(QtWidgets.QGraphicsEllipseItem.mouseReleaseEvent)

    def set_species_trigger(self):

        if self.project_loaded and not (self.selected_column == -1):

            items = ('Al', 'Mg', 'Si', 'Cu', 'Un')
            item, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Set", "Species", items, 0, False)

            if ok_pressed and item:

                if item == 'Al':
                    h = 3
                elif item == 'Si':
                    h = 0
                elif item == 'Mg':
                    h = 5
                elif item == 'Cu':
                    h = 1
                else:
                    h = 6

                self.set_species(h)

    def set_species(self, h):

        if self.project_loaded and not self.selected_column == -1:

            self.project_instance.graph.vertices[self.selected_column].force_species(h)

            self.control_window.lbl_column_species.setText(
                'Atomic species: ' + self.project_instance.graph.vertices[self.selected_column].atomic_species)
            self.control_window.lbl_confidence.setText(
                'Confidence: ' + str(self.project_instance.graph.vertices[self.selected_column].confidence))
            self.project_instance.graph.vertices[self.selected_column].flag_1 = False
            self.overlay_objects[self.selected_column] = \
                self.set_species_colors(self.overlay_objects[self.selected_column], self.selected_column)

            self.control_window.draw_histogram()

            self.overlay_objects[self.selected_column] = self.set_species_colors(
                self.overlay_objects[self.selected_column], self.selected_column)

    def set_level_trigger(self):
        if self.project_loaded and not (self.selected_column == -1):
            items = ('0', '1', 'other')
            item, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Set", "Level", items, 0, False)
            if ok_pressed and item:
                if item == '0':
                    self.set_level(0)
                elif item == '1':
                    self.set_level(1)
                elif item == 'other':
                    level = self.project_instance.graph.vertices[self.selected_column].anti_level()
                    self.set_level(level)

    def set_level(self, level):
        self.project_instance.graph.vertices[self.selected_column].level = level
        self.control_window.lbl_column_level.setText(
            'Level: ' + str(self.project_instance.graph.vertices[self.selected_column].level))
        self.overlay_objects[self.selected_column] = self.set_species_colors(
            self.overlay_objects[self.selected_column], self.selected_column)
        if level == 0:
            self.vertex_objects[self.selected_column].setBrush(self.brush_level_0)
        else:
            self.vertex_objects[self.selected_column].setBrush(self.brush_level_1)

    def continue_detection_trigger(self):
        if self.project_loaded:
            items = ('s', 't', 'other')
            item, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Set", "Search type", items, 0, False)
            if ok_pressed and item:
                self.statusBar().showMessage('Working...')
                self.project_instance.redraw_search_mat()
                self.project_instance.column_detection(item)
                self.update_display()

    def restart_detection_trigger(self):

        if self.project_loaded:
            self.statusBar().showMessage('Working...')
            self.deselect_trigger()
            self.project_instance.reset_graph()
            self.previous_pos_obj = None
            self.previous_overlay_obj = None
            self.update_display()

    def continue_analysis_trigger(self):

        if self.project_loaded and not self.selected_column == -1:

            strings = ['0 - Full column characterization algorithm with legacy untangling',
                       '1 - Full column characterization algorithm with experimental untangling',
                       '2 - Run spatial mapping',
                       '3 - Apply angle statistics',
                       '4 - Apply intensity statistics',
                       '5 - Run particle detection',
                       '6 - Set levels',
                       '7 - Redraw edges',
                       '8 - Run legacy weak untangling',
                       '9 - Run legacy strong untangling',
                       '10 - Run experimental weak untangling',
                       '11 - Run experimental strong untangling',
                       '12 - Reset probability vectors',
                       '13 - Reset user-set columns',
                       '14 - Search for intersections',
                       '15 - Experimental',
                       '16 - Experimental angle score',
                       '17 - Experimental levels',
                       '18 - Find edge columns']

            string, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Set", "Search step", strings, 0, False)
            if ok_pressed and strings:
                self.statusBar().showMessage('Analyzing... This may take a long time...')
                sys.setrecursionlimit(10000)
                choice = -1
                for k in range(0, len(strings)):
                    if string == strings[k]:
                        choice = k
                if not choice == -1:
                    self.project_instance.column_characterization(self.selected_column, choice)
                    self.update_display()
                else:
                    self.report('Invalid selection. Was not able to start column detection.', force=True)

    def restart_analysis_trigger(self):

        self.project_instance.reset_vertex_properties()

        self.project_instance.precipitate_boarder = np.ndarray([1], dtype=int)
        self.project_instance.boarder_size = 0
        self.project_instance.graph.reset_all_flags()
        self.update_display()
        self.statusBar().showMessage('Reset all prob_vector\'s')

    def invert_levels_trigger(self):

        if self.project_loaded:
            self.project_instance.graph.invert_levels()
            self.update_central_widget()

    def image_correction_trigger(self):

        if self.project_loaded:
            message = QtWidgets.QMessageBox()
            message.setText('Not implemented yet')
            message.exec_()

    def image_filter_trigger(self):

        if self.project_loaded:
            message = QtWidgets.QMessageBox()
            message.setText('Not implemented yet')
            message.exec_()

    def image_adjustments_trigger(self):

        if self.project_loaded:
            message = QtWidgets.QMessageBox()
            message.setText('Not implemented yet')
            message.exec_()

    def export_data_trigger(self):

        if self.project_loaded:
            message = QtWidgets.QMessageBox()
            message.setText('Not implemented yet')
            message.exec_()

    def export_raw_image_trigger(self):

        if self.project_loaded:

            filename = QtWidgets.QFileDialog.getSaveFileName(self, "Save image",
                                                             '',
                                                             "PNG (*.png);;BMP Files (*.bmp);;JPEG (*.JPEG)")
            if filename[0]:
                rect_f = self.graphicScene_1.sceneRect()
                print(str(type(rect_f.size().toSize())))
                img = QtGui.QImage(rect_f.size().toSize(), QtGui.QImage.Format_ARGB32)
                img.fill(QtCore.Qt.white)
                p = QtGui.QPainter(img)
                self.graphicScene_1.render(p, target=QtCore.QRectF(img.rect()), source=rect_f)
                p.end()
                saving = img.save(filename[0])
                print("Saving Pass" if saving else "Saving Not Pass")

    def export_column_position_image_trigger(self):

        if self.project_loaded:

            filename = QtWidgets.QFileDialog.getSaveFileName(self, "Save image",
                                                             '',
                                                             "PNG (*.png);;BMP Files (*.bmp);;JPEG (*.JPEG)")
            if filename[0]:
                rect_f = self.graphicScene_2.sceneRect()
                print(str(type(rect_f.size().toSize())))
                img = QtGui.QImage(rect_f.size().toSize(), QtGui.QImage.Format_ARGB32)
                img.fill(QtCore.Qt.white)
                p = QtGui.QPainter(img)
                self.graphicScene_2.render(p, target=QtCore.QRectF(img.rect()), source=rect_f)
                p.end()
                saving = img.save(filename[0])
                print("Saving Pass" if saving else "Saving Not Pass")

    def export_overlay_image_trigger(self):

        if self.project_loaded:

            filename = QtWidgets.QFileDialog.getSaveFileName(self, "Save image",
                                                             '',
                                                             "PNG (*.png);;BMP Files (*.bmp);;JPEG (*.JPEG)")
            if filename[0]:
                rect_f = self.graphicScene_3.sceneRect()
                print(str(type(rect_f.size().toSize())))
                img = QtGui.QImage(rect_f.size().toSize(), QtGui.QImage.Format_ARGB32)
                img.fill(QtCore.Qt.white)
                p = QtGui.QPainter(img)
                self.graphicScene_3.render(p, target=QtCore.QRectF(img.rect()), source=rect_f)
                p.end()
                saving = img.save(filename[0])
                print("Saving Pass" if saving else "Saving Not Pass")

    def export_atomic_graph_trigger(self):

        if self.project_loaded:

            filename = QtWidgets.QFileDialog.getSaveFileName(self, "Save image",
                                                             '',
                                                             "PNG (*.png);;BMP Files (*.bmp);;JPEG (*.JPEG)")
            if filename[0]:
                rect_f = self.graphicScene_6.sceneRect()
                print(str(type(rect_f.size().toSize())))
                img = QtGui.QImage(rect_f.size().toSize(), QtGui.QImage.Format_ARGB32)
                img.fill(QtCore.Qt.white)
                p = QtGui.QPainter(img)
                self.graphicScene_6.render(p, target=QtCore.QRectF(img.rect()), source=rect_f)
                p.end()
                saving = img.save(filename[0])
                print("Saving Pass" if saving else "Saving Not Pass")

    def toggle_debug_mode_trigger(self, state):

        if state:
            self.control_window.debug_box.show()
            self.draw_connections(self.selected_column)
            self.project_instance.debug_mode = True
        else:
            self.control_window.debug_box.hide()
            temp = self.selected_column
            self.selected_column = -1
            self.draw_connections(0)
            self.selected_column = temp
            self.project_instance.debug_mode = False

    def add_mark_trigger(self):

        if self.project_loaded and not self.selected_column == -1:
            self.project_instance.vertex_report(self.selected_column)

    def invert_precipitate_columns_trigger(self):

        if self.project_loaded:

            self.project_instance.graph.invert_levels(True)

    def set_control_file_trigger(self):

        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open control file for comparison', '')
        if filename[0]:
            self.statusBar().showMessage('Working...')
            print(filename[0])
            self.control_instance = core.SuchSoftware.load(filename[0])
            self.statusBar().showMessage('Ready')
        else:
            self.control_instance = None
            self.statusBar().showMessage('Ready')

    def run_benchmark_trigger(self):

        dev_module.accumulate_statistics()

    def display_deviations_trigger(self):

        if self.project_loaded and self.project_instance.num_columns > 0 and self.control_instance is not None:

            deviations = 0
            symmetry_deviations = 0
            flags = 0
            correct_flags = 0
            erroneous_flags = 0
            popular = 0
            unpopular = 0

            for x in range(0, self.project_instance.num_columns):

                if self.project_instance.graph.vertices[x].h_index == self.control_instance.graph.vertices[x].h_index:
                    pass
                elif self.project_instance.graph.vertices[x].h_index == 0 and self.control_instance.graph.vertices[x].h_index == 1:
                    deviations = deviations + 1
                elif self.project_instance.graph.vertices[x].h_index == 1 and self.control_instance.graph.vertices[x].h_index == 0:
                    deviations = deviations + 1
                else:
                    deviations = deviations + 1
                    symmetry_deviations += 1

                if self.project_instance.graph.vertices[x].is_unpopular or self.project_instance.graph.vertices[x].is_popular:

                    flags = flags + 1
                    if self.project_instance.graph.vertices[x].h_index == self.control_instance.graph.vertices[x].h_index:
                        erroneous_flags = erroneous_flags + 1
                    else:
                        correct_flags = correct_flags + 1
                    if self.project_instance.graph.vertices[x].is_unpopular:
                        unpopular = unpopular + 1
                    if self.project_instance.graph.vertices[x].is_popular:
                        popular = popular + 1

            undetected_errors = deviations - correct_flags

            print('Deviations: ' + str(deviations))
            message = QtWidgets.QMessageBox()
            message.setText('Flags: ' + str(flags) + '\nDeviations: ' + str(deviations) + '\nPercentage: ' + str(
                deviations / self.project_instance.num_columns) + '\nSymmetry deviations: ' + str(symmetry_deviations) + '\nCorrect flags: ' + str(correct_flags) +
                '\nErroneous flags: ' + str(erroneous_flags) + '\nundetected errors: ' + str(undetected_errors) +
                '\nPopular: ' + str(popular) + '\nUnpopular: ' + str(unpopular))
            message.exec_()

    def test_consistency_trigger(self):

        # Highjacked

        if self.project_loaded:

            for x in range(0, self.project_instance.num_columns):

                self.project_instance.graph.vertices[x].level = 0

    def clear_flags_trigger(self):

        if self.project_loaded and self.project_instance.num_columns > 0:

            self.project_instance.graph.reset_all_flags()
            self.update_central_widget()

    @ staticmethod
    def there_is_no_help_trigger():

        message = QtWidgets.QMessageBox()
        message.setText('Mental Helses Hjelpetelefon er åpen døgnet rundt på 116 123.')
        message.exec_()

    def toggle_precipitate_trigger(self):

        if self.project_loaded and not self.selected_column == -1:

            if self.control_window.chb_precipitate_column.isChecked():

                self.project_instance.graph.vertices[self.selected_column].is_in_precipitate = True

            else:

                self.project_instance.graph.vertices[self.selected_column].is_in_precipitate = False

            self.project_instance.summarize_stats()

    def toggle_show_trigger(self):

        if self.selected_column == -1:
            pass
        else:
            self.project_instance.graph.vertices[self.selected_column].show_in_overlay =\
                not self.project_instance.graph.vertices[self.selected_column].show_in_overlay
            if self.project_instance.graph.vertices[self.selected_column].show_in_overlay:
                self.overlay_objects[self.selected_column].show()
            else:
                self.overlay_objects[self.selected_column].hide()

    def move_trigger(self):

        self.control_window.mode_move(True)

    def cancel_move_trigger(self):

        self.control_window.mode_move(False)

    def set_position_trigger(self):
        """This function is a mess and needs re-implementing."""

        self.statusBar().showMessage('Working...')
        self.pos_objects[self.selected_column].setFlag(QtWidgets.QGraphicsItem.ItemIsPanel)

        x_coor = self.pos_objects[self.selected_column].x() + self.project_instance.r
        y_coor = self.pos_objects[self.selected_column].y() + self.project_instance.r
        self.pos_objects[self.selected_column].x_0 = np.floor(self.pos_objects[self.selected_column].x() + self.project_instance.r)
        self.pos_objects[self.selected_column].y_0 = np.floor(self.pos_objects[self.selected_column].y() + self.project_instance.r)
        r = self.project_instance.r

        self.project_instance.graph.vertices[self.selected_column].real_coor_x = x_coor
        self.project_instance.graph.vertices[self.selected_column].real_coor_y = y_coor
        self.project_instance.graph.vertices[self.selected_column].im_coor_x = int(np.floor(x_coor))
        self.project_instance.graph.vertices[self.selected_column].im_coor_y = int(np.floor(y_coor))

        if self.project_instance.im_width - r > x_coor > r and self.project_instance.im_height - r > y_coor > r:

            self.project_instance.graph.vertices[self.selected_column].avg_gamma, self.project_instance.graph.vertices[
                self.selected_column].peak_gamma = mat_op.average(self.project_instance.im_mat, int(x_coor), int(y_coor), r)

        else:

            self.project_instance.im_mat = mat_op.gen_framed_mat(self.project_instance.im_mat, r)
            self.project_instance.graph.vertices[self.selected_column].avg_gamma, self.project_instance.graph.vertices[
                self.selected_column].peak_gamma = mat_op.average(self.project_instance.im_mat, x_coor + r, y_coor + r,
                                                                  r)
            self.project_instance.im_mat = mat_op.gen_de_framed_mat(self.project_instance.im_mat, r)

        self.project_instance.redraw_centre_mat()

        self.control_window.lbl_column_x_pos.setText('x: ' + str(self.pos_objects[self.selected_column].x_0))
        self.control_window.lbl_column_y_pos.setText('y: ' + str(self.pos_objects[self.selected_column].y_0))
        self.control_window.lbl_column_peak_gamma.setText(
            'Peak gamma: ' + str(self.project_instance.graph.vertices[self.selected_column].peak_gamma))
        self.control_window.lbl_column_avg_gamma.setText(
            'Avg gamma: ' + str(self.project_instance.graph.vertices[self.selected_column].avg_gamma))
        self.control_window.lbl_neighbours.setText('Nearest neighbours: ' + str(self.project_instance.graph.vertices[
            self.selected_column].neighbour_indices))

        self.cancel_move_trigger()

    def delete_trigger(self):

        if self.project_loaded:
            message = QtWidgets.QMessageBox()
            message.setText('Not implemented yet')
            message.exec_()

    def deselect_trigger(self):

        if self.previous_pos_obj is None:
            pass
        else:
            self.pos_objects[self.selected_column].unselect()

        if self.previous_overlay_obj is None:
            pass
        else:
            self.overlay_objects[self.selected_column].unselect()

        self.selected_column = -1

        self.draw_connections(0)

        self.control_window.lbl_column_index.setText('Column index: ')
        self.control_window.lbl_column_x_pos.setText('x: ')
        self.control_window.lbl_column_y_pos.setText('y: ')
        self.control_window.lbl_column_peak_gamma.setText('Peak gamma: ')
        self.control_window.lbl_column_avg_gamma.setText('Avg gamma: ')
        self.control_window.lbl_column_species.setText('Atomic species: ')
        self.control_window.lbl_column_level.setText('Level: ')
        self.control_window.lbl_confidence.setText('Confidence: ')
        self.control_window.lbl_neighbours.setText('Nearest neighbours: ')
        self.control_window.lbl_central_variance.setText('Central angle variance: ')

        self.control_window.btn_new.setDisabled(True)
        self.control_window.btn_deselect.setDisabled(True)
        self.control_window.btn_delete.setDisabled(True)
        self.control_window.btn_set_species.setDisabled(True)
        self.control_window.btn_set_level.setDisabled(True)
        self.control_window.btn_find_column.setDisabled(True)
        self.control_window.chb_precipitate_column.setDisabled(True)
        self.control_window.chb_show.setDisabled(True)
        self.control_window.chb_move.setDisabled(True)

        self.control_window.chb_show.blockSignals(True)
        self.control_window.chb_show.setChecked(False)
        self.control_window.chb_show.blockSignals(False)
        self.control_window.chb_move.blockSignals(True)
        self.control_window.chb_move.setChecked(False)
        self.control_window.chb_move.blockSignals(False)

        self.control_window.btn_set_move.setDisabled(True)
        self.control_window.btn_cancel_move.setDisabled(True)

        self.control_window.draw_histogram()

    def new_column_trigger(self):

        if self.project_loaded:
            message = QtWidgets.QMessageBox()
            message.setText('Not implemented yet')
            message.exec_()

    def toggle_graph_detail_trigger(self):

        self.draw_atomic_graph(1)
        self.update_display()

    def set_style_trigger(self):

        if self.project_loaded:
            message = QtWidgets.QMessageBox()
            message.setText('Not implemented yet')
            message.exec_()

    def toggle_si_trigger(self):

        if self.project_loaded:
            for x in range(0, self.project_instance.num_columns):
                if self.project_instance.graph.vertices[x].h_index == 0:
                    if self.control_window.chb_si_columns.isChecked():
                        self.project_instance.graph.vertices[x].show_in_overlay = True
                        self.overlay_objects[x].show()
                    else:
                        self.project_instance.graph.vertices[x].show_in_overlay = False
                        self.overlay_objects[x].hide()

    def toggle_cu_trigger(self):

        if self.project_loaded:
            for x in range(0, self.project_instance.num_columns):
                if self.project_instance.graph.vertices[x].h_index == 1:
                    if self.control_window.chb_cu_columns.isChecked():
                        self.project_instance.graph.vertices[x].show_in_overlay = True
                        self.overlay_objects[x].show()
                    else:
                        self.project_instance.graph.vertices[x].show_in_overlay = False
                        self.overlay_objects[x].hide()

    def toggle_al_trigger(self):

        if self.project_loaded:
            for x in range(0, self.project_instance.num_columns):
                if self.project_instance.graph.vertices[x].h_index == 3:
                    if self.control_window.chb_al_columns.isChecked():
                        if self.control_window.chb_al_mesh.isChecked():
                            self.project_instance.graph.vertices[x].show_in_overlay = True
                            self.overlay_objects[x].show()
                        else:
                            if self.project_instance.graph.vertices[x].is_in_precipitate:
                                self.project_instance.graph.vertices[x].show_in_overlay = True
                                self.overlay_objects[x].show()
                            else:
                                self.project_instance.graph.vertices[x].show_in_overlay = False
                                self.overlay_objects[x].hide()
                    else:
                        self.project_instance.graph.vertices[x].show_in_overlay = False
                        self.overlay_objects[x].hide()

    def toggle_ag_trigger(self):

        if self.project_loaded:
            for x in range(0, self.project_instance.num_columns):
                if self.project_instance.graph.vertices[x].h_index == 4:
                    if self.control_window.chb_ag_columns.isChecked():
                        self.project_instance.graph.vertices[x].show_in_overlay = True
                        self.overlay_objects[x].show()
                    else:
                        self.project_instance.graph.vertices[x].show_in_overlay = False
                        self.overlay_objects[x].hide()

    def toggle_mg_trigger(self):

        if self.project_loaded:
            for x in range(0, self.project_instance.num_columns):
                if self.project_instance.graph.vertices[x].h_index == 5:
                    if self.control_window.chb_mg_columns.isChecked():
                        self.project_instance.graph.vertices[x].show_in_overlay = True
                        self.overlay_objects[x].show()
                    else:
                        self.project_instance.graph.vertices[x].show_in_overlay = False
                        self.overlay_objects[x].hide()

    def toggle_un_trigger(self):

        if self.project_loaded:
            for x in range(0, self.project_instance.num_columns):
                if self.project_instance.graph.vertices[x].h_index == 6:
                    if self.control_window.chb_un_columns.isChecked():
                        self.project_instance.graph.vertices[x].show_in_overlay = True
                        self.overlay_objects[x].show()
                    else:
                        self.project_instance.graph.vertices[x].show_in_overlay = False
                        self.overlay_objects[x].hide()

    def toggle_column_trigger(self):

        if self.project_loaded:
            set_state = self.control_window.chb_columns.isChecked()

            self.control_window.chb_si_columns.setChecked(set_state)
            self.toggle_si_trigger()
            self.control_window.chb_cu_columns.setChecked(set_state)
            self.toggle_cu_trigger()
            self.control_window.chb_al_columns.setChecked(set_state)
            self.toggle_al_trigger()
            self.control_window.chb_ag_columns.setChecked(set_state)
            self.toggle_ag_trigger()
            self.control_window.chb_mg_columns.setChecked(set_state)
            self.toggle_mg_trigger()
            self.control_window.chb_un_columns.setChecked(set_state)
            self.toggle_un_trigger()

    def toggle_al_mesh_trigger(self):

        if self.project_loaded:

            if not self.control_window.chb_al_mesh.isChecked():

                for x in range(0, self.project_instance.num_columns):

                    if self.project_instance.graph.vertices[x].is_in_precipitate:
                        self.project_instance.graph.vertices[x].show_in_overlay = True
                        self.overlay_objects[x].show()
                    else:
                        self.project_instance.graph.vertices[x].show_in_overlay = False
                        self.overlay_objects[x].hide()

            else:

                self.control_window.chb_columns.setChecked(True)
                self.toggle_column_trigger()

    def select_column(self, i):

        if i == -1:
            self.deselect_trigger()
        else:

            self.control_window.lbl_column_index.setText('Column index: ' + str(i))
            self.control_window.lbl_column_x_pos.setText('x: ' + str(self.project_instance.graph.vertices[i].im_coor_x))
            self.control_window.lbl_column_y_pos.setText('y: ' + str(self.project_instance.graph.vertices[i].im_coor_y))
            self.control_window.lbl_column_peak_gamma.setText(
                'Peak gamma: ' + str(self.project_instance.graph.vertices[i].peak_gamma))
            self.control_window.lbl_column_avg_gamma.setText(
                'Avg gamma: ' + str(self.project_instance.graph.vertices[i].avg_gamma))
            self.control_window.lbl_column_species.setText(
                'Atomic species: ' + self.project_instance.graph.vertices[i].atomic_species)
            self.control_window.lbl_column_level.setText('Level: ' + str(self.project_instance.graph.vertices[i].level))
            self.control_window.lbl_confidence.setText(
                'Confidence: ' + str(self.project_instance.graph.vertices[i].confidence))
            self.control_window.lbl_symmetry_confidence.setText(
                'Symmetry confidence: ' + str(
                    self.project_instance.graph.vertices[self.selected_column].symmetry_confidence))
            self.control_window.lbl_level_confidence.setText(
                'Level confidence: ' + str(self.project_instance.graph.vertices[self.selected_column].level_confidence))
            self.control_window.lbl_neighbours.setText(
                'Nearest neighbours: ' + str(self.project_instance.graph.vertices[i].neighbour_indices))
            rotation_partners, angles, variance = self.project_instance.graph.calc_central_angle_variance(
                self.selected_column)
            self.control_window.lbl_central_variance.setText('Central angle variance: ' + str(variance))

            self.control_window.btn_new.setDisabled(False)
            self.control_window.btn_deselect.setDisabled(False)
            self.control_window.btn_delete.setDisabled(False)
            self.control_window.btn_set_species.setDisabled(False)
            self.control_window.btn_set_level.setDisabled(False)
            self.control_window.btn_find_column.setDisabled(False)
            self.control_window.chb_precipitate_column.setDisabled(False)
            self.control_window.chb_show.setDisabled(False)
            self.control_window.chb_move.setDisabled(False)

            self.control_window.chb_show.blockSignals(True)
            self.control_window.chb_show.setChecked(self.project_instance.graph.vertices[i].show_in_overlay)
            self.control_window.chb_show.blockSignals(False)
            self.control_window.chb_precipitate_column.blockSignals(True)
            self.control_window.chb_precipitate_column.setChecked(
                self.project_instance.graph.vertices[i].is_in_precipitate)
            self.control_window.chb_precipitate_column.blockSignals(False)
            self.control_window.chb_move.blockSignals(True)
            self.control_window.chb_move.setChecked(False)
            self.control_window.chb_move.blockSignals(False)

            self.control_window.btn_set_move.setDisabled(True)
            self.control_window.btn_cancel_move.setDisabled(True)

            self.selected_column = i

            if self.control_window.debug_box.isVisible():
                self.draw_connections(self.selected_column)
                self.graphicsView_3.setScene(self.graphicScene_3)

            self.control_window.draw_histogram()

    def column_selected(self, i):

        if i == -1:
            self.deselect_trigger()
        else:

            self.control_window.lbl_column_index.setText('Column index: ' + str(i))
            self.control_window.lbl_column_x_pos.setText('x: ' + str(self.project_instance.graph.vertices[i].im_coor_x))
            self.control_window.lbl_column_y_pos.setText('y: ' + str(self.project_instance.graph.vertices[i].im_coor_y))
            self.control_window.lbl_column_peak_gamma.setText(
                'Peak gamma: ' + str(self.project_instance.graph.vertices[i].peak_gamma))
            self.control_window.lbl_column_avg_gamma.setText('Avg gamma: ' + str(self.project_instance.graph.vertices[i].avg_gamma))
            self.control_window.lbl_column_species.setText(
                'Atomic species: ' + self.project_instance.graph.vertices[i].atomic_species)
            self.control_window.lbl_column_level.setText('Level: ' + str(self.project_instance.graph.vertices[i].level))
            self.control_window.lbl_confidence.setText('Confidence: ' + str(self.project_instance.graph.vertices[i].confidence))
            self.control_window.lbl_symmetry_confidence.setText(
                'Symmetry confidence: ' + str(
                    self.project_instance.graph.vertices[self.selected_column].symmetry_confidence))
            self.control_window.lbl_level_confidence.setText(
                'Level confidence: ' + str(self.project_instance.graph.vertices[self.selected_column].level_confidence))
            self.control_window.lbl_neighbours.setText('Nearest neighbours: ' + str(self.project_instance.graph.vertices[i].neighbour_indices))
            rotation_partners, angles, variance = self.project_instance.graph.calc_central_angle_variance(self.selected_column)
            self.control_window.lbl_central_variance.setText('Central angle variance: ' + str(variance))

            self.control_window.btn_new.setDisabled(False)
            self.control_window.btn_deselect.setDisabled(False)
            self.control_window.btn_delete.setDisabled(False)
            self.control_window.btn_set_species.setDisabled(False)
            self.control_window.btn_set_level.setDisabled(False)
            self.control_window.btn_find_column.setDisabled(False)
            self.control_window.chb_precipitate_column.setDisabled(False)
            self.control_window.chb_show.setDisabled(False)
            self.control_window.chb_move.setDisabled(False)

            self.control_window.chb_show.blockSignals(True)
            self.control_window.chb_show.setChecked(self.project_instance.graph.vertices[i].show_in_overlay)
            self.control_window.chb_show.blockSignals(False)
            self.control_window.chb_precipitate_column.blockSignals(True)
            self.control_window.chb_precipitate_column.setChecked(self.project_instance.graph.vertices[i].is_in_precipitate)
            self.control_window.chb_precipitate_column.blockSignals(False)
            self.control_window.chb_move.blockSignals(True)
            self.control_window.chb_move.setChecked(False)
            self.control_window.chb_move.blockSignals(False)

            self.control_window.btn_set_move.setDisabled(True)
            self.control_window.btn_cancel_move.setDisabled(True)

            if not self.selected_column == -1 and self.control_window.debug_box.isVisible() and not self.project_instance.graph.vertices[self.selected_column].neighbour_indices == []:

                for x in range(0, self.project_instance.graph.vertices[self.selected_column].n()):

                    if self.project_instance.graph.vertices[self.selected_column].neighbour_indices[x] == i:

                        corners, angles, vectors = self.project_instance.graph.find_mesh(self.selected_column, i)
                        print(str(len(corners)) + ': ' + str(i) + ' ' + str(self.selected_column))

            self.previous_selected_column, self.selected_column  = self.selected_column, i

            if self.control_window.debug_box.isVisible():
                self.draw_connections(self.selected_column)
                self.graphicsView_3.setScene(self.graphicScene_3)

            self.control_window.draw_histogram()

    def update_central_widget(self):

        # Buffer background images
        mat_op.im_out_static(self.project_instance.im_mat.astype(np.float64), 'Images\Outputs\Buffers\\raw_image.png')
        mat_op.im_out_static(self.project_instance.search_mat.astype(np.float64), 'Images\Outputs\Buffers\search_image.png')
        mat_op.im_out_static(self.project_instance.fft_im_mat.astype(np.float64), 'Images\Outputs\Buffers\FFT.png')

        # Draw raw image tab
        self.graphic = QtGui.QPixmap('Images\\Outputs\\Buffers\\raw_image.png')
        self.graphicScene_1 = QtWidgets.QGraphicsScene()
        self.graphicScene_1.addPixmap(self.graphic)
        self.graphicsView_1.setScene(self.graphicScene_1)

        # Draw atomic positions tab
        self.graphicScene_2 = QtWidgets.QGraphicsScene()
        self.draw_positions()
        self.graphicsView_2.setScene(self.graphicScene_2)

        # Draw overlay image tab
        self.graphicScene_3 = QtWidgets.QGraphicsScene()
        self.draw_overlay()
        self.draw_boarder()
        self.neighbour_line_objects = np.ndarray([1], dtype=type(QtWidgets.QGraphicsLineItem()))
        self.neighbour_line_objects[0] = QtWidgets.QGraphicsLineItem(1.0, 2.0, 3.0, 4.0)
        self.draw_connections()
        self.graphicsView_3.setScene(self.graphicScene_3)

        # Draw atomic graph
        self.graphicScene_6 = QtWidgets.QGraphicsScene()
        self.draw_atomic_graph(1)
        self.graphicsView_6.setScene(self.graphicScene_6)

        # Draw atomic sub-graph

        # Draw search matrix tab
        self.graphic = QtGui.QPixmap('Images\\Outputs\\Buffers\\search_image.png')
        self.graphicScene_4 = QtWidgets.QGraphicsScene()
        self.graphicScene_4.addPixmap(self.graphic)
        self.graphicsView_4.setScene(self.graphicScene_4)

        # Draw FFT
        self.graphic = QtGui.QPixmap('Images\\Outputs\\Buffers\\FFT.png')
        self.graphicScene_5 = QtWidgets.QGraphicsScene()
        self.graphicScene_5.addPixmap(self.graphic)
        self.graphicsView_5.setScene(self.graphicScene_5)

        # Finit!
        self.statusBar().showMessage('Ready')

    def update_display(self):

        self.update_central_widget()

        self.column_selected(self.selected_column)

        # Update labels
        self.control_window.lbl_num_detected_columns.setText('Number of detected columns: ' + str(self.project_instance.num_columns))
        self.control_window.lbl_image_width.setText('Image width (pixels): ' + str(self.project_instance.im_width))
        self.control_window.lbl_image_height.setText('Image height (pixels): ' + str(self.project_instance.im_height))
        self.control_window.lbl_atomic_radii.setText('Approx atomic radii (pixels): ' + str(self.project_instance.r))
        self.control_window.lbl_detection_threshold.setText('Detection threshold value: ' + str(self.project_instance.threshold))
        self.control_window.lbl_search_matrix_peak.setText('Search matrix peak: ' + str(np.max(self.project_instance.search_mat)))
        self.control_window.lbl_search_size.setText('Search size: ' + str(self.project_instance.search_size))
        self.control_window.lbl_scale.setText('Scale (pm / pixel): ' + str(self.project_instance.scale))
        self.control_window.lbl_overhead_radii.setText('Overhead (pixels): ' + str(self.project_instance.overhead))

        chi = self.project_instance.graph.calc_chi()
        avg_species_confidence = self.project_instance.graph.calc_avg_species_confidence()
        avg_symmetry_confidence = self.project_instance.graph.calc_avg_symmetry_confidence()
        avg_level_confidence = self.project_instance.graph.calc_avg_level_confidence()
        avg_variance = self.project_instance.graph.calc_avg_central_angle_variance()

        self.control_window.lbl_chi.setText('Chi: ' + str(chi))
        self.control_window.lbl_avg_species_confidence.setText('Average species confidence: ' + str(avg_species_confidence))
        self.control_window.lbl_avg_symmetry_confidence.setText('Average symmetry confidence: ' + str(avg_symmetry_confidence))
        self.control_window.lbl_avg_level_confidence.setText('Average level confidence: ' + str(avg_level_confidence))
        self.control_window.lbl_avg_variance.setText('Average angle variance: ' + str(avg_variance))

        if self.project_instance.alloy == 0:
            self.control_window.lbl_alloy.setText('Alloy: Al-Mg-Si-(Cu)')
        elif self.project_instance.alloy == 1:
            self.control_window.lbl_alloy.setText('Alloy: Al-Mg-Si')
        else:
            self.control_window.lbl_alloy.setText('Alloy: Unknown')
        self.control_window.lbl_starting_index.setText(
            'Default starting index: ' + str(self.project_instance.starting_index))
        self.control_window.lbl_std_1.setText('Standard deviation 1: ' + str(self.project_instance.dist_1_std))
        self.control_window.lbl_std_2.setText('Standard deviation 2: ' + str(self.project_instance.dist_2_std))
        self.control_window.lbl_std_3.setText('Standard deviation 3: ' + str(self.project_instance.dist_3_std))
        self.control_window.lbl_std_4.setText('Standard deviation 4: ' + str(self.project_instance.dist_4_std))
        self.control_window.lbl_std_5.setText('Standard deviation 5: ' + str(self.project_instance.dist_5_std))
        self.control_window.lbl_std_8.setText('Standard deviation 8: ' + str(self.project_instance.dist_8_std))
        self.control_window.lbl_cert_threshold.setText('Certainty threshold: ' + str(self.project_instance.certainty_threshold))

    def draw_positions(self):

        self.graphic = QtGui.QPixmap('Images\\Outputs\\Buffers\\raw_image.png')
        self.graphicScene_2.addPixmap(self.graphic)

        r = self.project_instance.r

        dummy_instance = legacy_GUI_elements.InteractivePosColumn(0, 0, 0, 0)
        self.pos_objects = np.ndarray([1], dtype=type(dummy_instance))

        if self.project_instance.num_columns > 0:
            for i in range(0, self.project_instance.num_columns):

                custom_ellipse_pos = legacy_GUI_elements.InteractivePosColumn(0, 0, 2 * r, 2 * r)
                custom_ellipse_pos.moveBy(self.project_instance.graph.vertices[i].real_coor_x - r,
                                          self.project_instance.graph.vertices[i].real_coor_y - r)
                custom_ellipse_pos.reference_object(self, i)

                if self.project_instance.graph.vertices[i].show_in_overlay:
                    custom_ellipse_pos.setPen(self.red_pen)
                else:
                    custom_ellipse_pos.setPen(self.dark_red_pen)

                custom_ellipse_pos.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)

                if i == 0:
                    self.pos_objects[0] = custom_ellipse_pos
                else:
                    self.pos_objects = np.append(self.pos_objects, custom_ellipse_pos)

                self.graphicScene_2.addItem(custom_ellipse_pos)

            if not self.selected_column == -1:
                self.pos_objects[self.selected_column].select()

    def draw_overlay(self):

        if self.project_loaded:

            self.graphic = QtGui.QPixmap('Images\\Outputs\\Buffers\\raw_image.png')

            if self.control_window.chb_raw_image.isChecked():
                self.graphicScene_3.addPixmap(self.graphic)

            if self.control_window.chb_black_background.isChecked():
                self.graphicScene_3.setBackgroundBrush(self.brush_black)

            r = self.project_instance.r

            dummy_instance = legacy_GUI_elements.InteractiveOverlayColumn(0, 0, 0, 0)
            self.overlay_objects = np.ndarray([1], dtype=type(dummy_instance))

            if self.project_instance.num_columns > 0:
                for i in range(0, self.project_instance.num_columns):

                    custom_ellipse_overlay = legacy_GUI_elements.InteractiveOverlayColumn(0, 0, r, r)
                    custom_ellipse_overlay.moveBy(self.project_instance.graph.vertices[i].im_coor_x - np.round(r / 2),
                                                  self.project_instance.graph.vertices[i].im_coor_y - np.round(r / 2))
                    custom_ellipse_overlay.reference_object(self, i)
                    custom_ellipse_overlay = self.set_species_colors(custom_ellipse_overlay, i)

                    custom_ellipse_overlay.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)

                    if i == 0:
                        self.overlay_objects[0] = custom_ellipse_overlay
                    else:
                        self.overlay_objects = np.append(self.overlay_objects, custom_ellipse_overlay)

                    self.graphicScene_3.addItem(custom_ellipse_overlay)

                if not self.selected_column == -1:
                    self.overlay_objects[self.selected_column].select()

    def set_species_colors(self, column_object, i):

        if self.project_instance.graph.vertices[i].atomic_species == 'Al':
            column_object.setPen(self.pen_al)
            column_object.setBrush(self.brush_al)
        elif self.project_instance.graph.vertices[i].atomic_species == 'Mg':
            column_object.setPen(self.pen_mg)
            column_object.setBrush(self.brush_mg)
        elif self.project_instance.graph.vertices[i].atomic_species == 'Si':
            column_object.setPen(self.pen_si)
            column_object.setBrush(self.brush_si)
        elif self.project_instance.graph.vertices[i].atomic_species == 'Cu':
            column_object.setPen(self.pen_cu)
            column_object.setBrush(self.brush_cu)
        elif self.project_instance.graph.vertices[i].atomic_species == 'Zn':
            column_object.setPen(self.pen_zn)
            column_object.setBrush(self.brush_zn)
        elif self.project_instance.graph.vertices[i].atomic_species == 'Ag':
            column_object.setPen(self.pen_ag)
            column_object.setBrush(self.brush_ag)
        elif self.project_instance.graph.vertices[i].atomic_species == 'Un':
            column_object.setPen(self.pen_un)
            column_object.setBrush(self.brush_un)
        else:
            column_object.setPen(self.pen_un)
            column_object.setBrush(self.brush_un)

        if self.project_instance.graph.vertices[i].level == 0:
            pass
        else:
            column_object.setBrush(self.brush_black)

        if self.control_window.debug_box.isVisible():
            if self.control_instance is not None and not self.project_instance.graph.vertices[i].h_index == self.control_instance.graph.vertices[i].h_index:
                column_object.setPen(self.pen_un)

            # if self.project_instance.columns[i].is_unpopular:
                # column_object.setPen(self.pen_problem)

            # if self.project_instance.columns[i].is_popular:
                # column_object.setPen(self.pen_ag)

        if self.project_instance.graph.vertices[i].show_in_overlay:
            column_object.show()
        else:
            column_object.hide()

        return column_object

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

    def draw_atomic_graph(self, scale_factor):

        if self.project_loaded and self.project_instance.num_columns > 0:

            r = self.project_instance.r * scale_factor

            # Draw edges
            self.project_instance.graph.redraw_edges()
            self.red_pen.setWidth(3)

            for i in range(0, self.project_instance.graph.num_edges):

                if not self.project_instance.graph.edges[i].vertex_a.neighbour_indices == []:

                    p1 = (self.project_instance.graph.edges[i].vertex_a.real_coor_x,
                          self.project_instance.graph.edges[i].vertex_a.real_coor_y)
                    p2 = (self.project_instance.graph.edges[i].vertex_b.real_coor_x,
                          self.project_instance.graph.edges[i].vertex_b.real_coor_y)

                    consistent = self.project_instance.graph.edges[i].is_consistent_edge

                    line, head_1 = self.make_arrow_obj(p1, p2, r, scale_factor, consistent)

                    if self.control_window.chb_graph.isChecked():
                        self.graphicScene_6.addItem(line)
                        if not self.project_instance.graph.edges[i].is_consistent_edge:
                            self.graphicScene_6.addItem(head_1)
                    else:
                        if consistent:
                            self.graphicScene_6.addItem(line)

            # Draw vertices

            dummy_instance = legacy_GUI_elements.InteractiveGraphVertex(0, 0, 0, 0)
            self.vertex_objects = np.ndarray([1], dtype=type(dummy_instance))

            for i in range(0, self.project_instance.num_columns):

                custom_ellipse_overlay = self.draw_vertex(r, scale_factor, i)

                if i == 0:
                    self.vertex_objects[0] = custom_ellipse_overlay
                else:
                    self.vertex_objects = np.append(self.vertex_objects, custom_ellipse_overlay)

                self.graphicScene_6.addItem(custom_ellipse_overlay)

            if not self.selected_column == -1:
                self.vertex_objects[self.selected_column].select()
            self.red_pen.setWidth(1)

    def draw_vertex(self, r, scale_factor, i):

        custom_ellipse_overlay = legacy_GUI_elements.InteractiveGraphVertex(0, 0, r, r)
        custom_ellipse_overlay.moveBy(
            2 * scale_factor * self.project_instance.graph.vertices[i].real_coor_x - r / 2,
            2 * scale_factor * self.project_instance.graph.vertices[i].real_coor_y - r / 2)
        custom_ellipse_overlay.reference_object(self, i)

        if self.project_instance.graph.vertices[i].level == 0:
            custom_ellipse_overlay.setBrush(self.brush_level_0)
        else:
            custom_ellipse_overlay.setBrush(self.brush_level_1)

        custom_ellipse_overlay.setPen(self.black_pen)
        custom_ellipse_overlay.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)

        return custom_ellipse_overlay

    def draw_atomic_sub_graph(self, scale_factor):

        if self.project_loaded and self.project_instance.num_columns > 0 and not self.selected_column == -1:

            if not self.project_instance.graph.vertices[self.selected_column].neighbour_indices == []:

                sub_graph = self.project_instance.graph.get_atomic_configuration(self.selected_column)
                r = self.project_instance.r * scale_factor

                # Draw edges

                for num, edge in enumerate(sub_graph.edges):

                    p1 = (edge.vertex_a.real_coor_x, edge.vertex_a.real_coor_y)
                    p2 = (edge.vertex_b.real_coor_x, edge.vertex_b.real_coor_y)

                    line, head = self.make_arrow_obj(p1, p2, r, scale_factor, True)

                    self.graphicScene_7.addItem(line)
                    self.graphicScene_7.addItem(head)

                self.red_pen.setWidth(1)

                # Draw vertices

                for num, vertex in enumerate(sub_graph.vertices):

                    custom_ellipse = self.draw_vertex(r, scale_factor, vertex.i)
                    custom_ellipse.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
                    custom_ellipse.setFlag(QtWidgets.QGraphicsItem.ItemIsPanel, True)

                    self.graphicScene_7.addItem(custom_ellipse)

                self.red_pen.setWidth(1)

                # Draw angles

                self.report('Sub-graph centered on vertex {}; Numerical summary:------------'.format(self.selected_column), force=True)
                central_angles = []
                m_max = 0

                for m, mesh in enumerate(sub_graph.meshes):

                    m_max = m

                    central_angles.append(mesh.angles[0])

                    self.report('Mesh {}:'.format(m), force=True)
                    self.report('    Is consistent: {}'.format(str(mesh.test_consistency())), force=True)
                    self.report('    Sum of angles: {}'.format(str(sum(mesh.angles))), force=True)
                    self.report('    Variance of angles: {}'.format(utils.variance(mesh.angles)), force=True)
                    self.report('    Symmetry prob vector from central angle: {}'.format(str([0, 0, 0])), force=True)
                    self.report('    corners: {}'.format(mesh.vertex_indices), force=True)

                    self.report('    Angles:', force=True)

                    for i, corner in enumerate(mesh.vertices):

                        p1 = corner.real_coor()
                        p2 = (p1[0] + 0.3 * r * mesh.angle_vectors[i][0], p1[1] + 0.3 * r * mesh.angle_vectors[i][1])

                        arrow, head = self.make_arrow_obj(p1, p2, r, scale_factor, False)

                        angle = mesh.angles[i]

                        angle_text = QtWidgets.QGraphicsSimpleTextItem()
                        angle_text.setText('a{}{}'.format(m, i))
                        angle_text.setFont(self.font_tiny)
                        rect = angle_text.boundingRect()
                        angle_text.setX(2 * scale_factor * p2[0] - 0.5 * rect.width())
                        angle_text.setY(2 * scale_factor * p2[1] - 0.5 * rect.height())

                        self.graphicScene_7.addItem(angle_text)
                        self.graphicScene_7.addItem(arrow)
                        self.report('        a{}{} = {}'.format(m, i, angle), force=True)

                self.report('Mean central angle: {}'.format(str(sum(central_angles) / (m_max + 1))), force=True)
                self.report('Central angle variance: {}'.format(utils.variance(central_angles)), force=True)
                self.report('Symmetry prob vector from mean central angle: {}'.format(str([0, 0, 0])), force=True)

    def draw_boarder(self):

        pass

    def plot_variance_trigger(self):

        if self.project_instance is not None:

            self.project_instance.plot_variances()

    def plot_angles_trigger(self):

        if self.project_instance is not None:

            self.project_instance.plot_min_max_angles()

    def draw_connections(self, index=-1):

        if index == -1:

            if self.project_loaded and self.project_instance.num_columns > 0 and self.control_window.chb_neighbours.isChecked():

                for x in range(0, self.project_instance.num_columns):

                    if not self.project_instance.graph.vertices[x].h_index == 6 and self.project_instance.graph.vertices[x].show_in_overlay and not self.project_instance.graph.vertices[x].neighbour_indices == []:

                        n = 3

                        if self.project_instance.graph.vertices[x].h_index == 0 or self.project_instance.graph.vertices[x].h_index == 1:
                            n = 3
                        elif self.project_instance.graph.vertices[x].h_index == 3:
                            n = 4
                        elif self.project_instance.graph.vertices[x].h_index == 5:
                            n = 5

                        for y in range(0, n):

                            i_1 = x
                            i_2 = self.project_instance.graph.vertices[x].neighbour_indices[y]
                            line = QtWidgets.QGraphicsLineItem(self.project_instance.graph.vertices[i_1].im_coor_x,
                                                               self.project_instance.graph.vertices[i_1].im_coor_y,
                                                               self.project_instance.graph.vertices[i_2].im_coor_x,
                                                               self.project_instance.graph.vertices[i_2].im_coor_y)
                            line.setPen(self.pen_connection)

                            self.graphicScene_3.addItem(line)

        else:

            if self.project_loaded and self.project_instance.num_columns > 0 and not self.project_instance.graph.vertices[index].neighbour_indices == []:

                for j in range(0, self.neighbour_line_objects.shape[0]):
                    self.neighbour_line_objects[j].hide()

                if self.selected_column == -1:
                    pass
                else:

                    x = index

                    n = 6

                    self.neighbour_line_objects = np.ndarray([n], dtype=type(QtWidgets.QGraphicsLineItem()))

                    for y in range(0, n):

                        i_1 = x
                        i_2 = self.project_instance.graph.vertices[x].neighbour_indices[y]
                        line = QtWidgets.QGraphicsLineItem(self.project_instance.graph.vertices[i_1].im_coor_x,
                                                           self.project_instance.graph.vertices[i_1].im_coor_y,
                                                           self.project_instance.graph.vertices[i_2].im_coor_x,
                                                           self.project_instance.graph.vertices[i_2].im_coor_y)

                        self.pen_connection.setWidth(2)
                        line.setPen(self.pen_connection)
                        self.pen_connection.setWidth(1)
                        if y == 0:
                            self.red_pen.setWidth(2)
                            line.setPen(self.red_pen)
                            self.red_pen.setWidth(1)
                        elif y == 1:
                            self.green_pen.setWidth(2)
                            line.setPen(self.green_pen)
                            self.green_pen.setWidth(1)
                        elif y == 2:
                            self.blue_pen.setWidth(2)
                            line.setPen(self.blue_pen)
                            self.blue_pen.setWidth(1)
                        elif y == 3:
                            line.setPen(self.yellow_pen)
                        elif y == 5:
                            self.black_pen.setWidth(2)
                            line.setPen(self.black_pen)
                            self.black_pen.setWidth(1)

                        self.neighbour_line_objects[y] = line
                        self.graphicScene_3.addItem(line)




