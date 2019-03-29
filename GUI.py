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
import GUI_elements


class MainUi(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        # Initialize in an 'empty state'
        self.project_instance = core.SuchSoftware('empty')
        self.project_loaded = False
        self.savefile = None
        self.control_instance = None
        self.selected_column = -1

        self.previous_pos_obj = None
        self.previous_overlay_obj = None

        # Lists of graphical objects
        dummy_instance = GUI_elements.InteractivePosColumn(0, 0, 0, 0)
        self.pos_objects = np.ndarray([1], dtype=type(dummy_instance))
        dummy_instance = GUI_elements.InteractiveOverlayColumn(0, 0, 0, 0)
        self.overlay_objects = np.ndarray([1], dtype=type(dummy_instance))

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
        GUI_elements.MenuBar(self.menuBar(), self)

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
        self.graphicsView_1 = QtWidgets.QGraphicsView(self.graphicScene_1)
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.graphicScene_2)
        self.graphicsView_3 = QtWidgets.QGraphicsView(self.graphicScene_3)
        self.graphicsView_4 = QtWidgets.QGraphicsView(self.graphicScene_4)
        self.graphicsView_5 = QtWidgets.QGraphicsView(self.graphicScene_5)
        self.graphicsView_6 = QtWidgets.QGraphicsView(self.graphicScene_6)
        self.graphicsView_7 = QtWidgets.QGraphicsView(self.graphicScene_7)

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

        self.control_window = GUI_elements.ControlWindow(obj=self)

        self.info_display_area = QtWidgets.QScrollArea()
        self.info_display_area.setWidget(self.control_window)
        self.info_display_area.setWidgetResizable(True)
        self.info_display_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

        self.info_display = QtWidgets.QDockWidget()
        self.info_display.setWidget(self.info_display_area)
        self.info_display.setWindowTitle('Control window')
        self.info_display.setMinimumWidth(300)

        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.info_display)

        self.toggle_debug_mode_trigger(False)
        self.deselect_trigger()

        # Display
        self.show()

    # Menu triggers:
    def new_trigger(self):

        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Select dm3', '')
        if filename[0]:
            self.statusBar().showMessage('Working...')
            self.project_instance = core.SuchSoftware(filename[0])
            self.control_instance = None
            self.project_loaded = True
            self.update_display()
        else:
            self.statusBar().showMessage('Ready')

    def open_trigger(self):

        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '')
        if filename[0]:
            self.statusBar().showMessage('Working...')
            print(filename[0])
            self.project_instance = core.SuchSoftware.load(filename[0])
            self.control_instance = None
            self.project_loaded = True
            self.savefile = filename[0]
            self.update_display()
        else:
            self.statusBar().showMessage('Ready')

    def save_trigger(self):

        if self.savefile is None:
            filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', '')
        else:
            filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', self.savefile)

        if filename[0]:
            self.statusBar().showMessage('Working...')
            self.project_instance.save(filename[0])
            print('Saved')
            self.update_display()
        else:
            self.statusBar().showMessage('Ready')

    def close_trigger(self):

        self.statusBar().showMessage('Working...')
        self.cancel_move_trigger()
        self.deselect_trigger()
        self.control_window.empty_display()

    @staticmethod
    def exit_trigger():
        QtWidgets.qApp.quit()

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

                dialog = GUI_elements.SetIndicesDialog()
                dialog.reference_object(self, self.selected_column)
                dialog.gen_layout()
                dialog.exec_()

    def set_indices_2_trigger(self):

        if self.project_loaded and not self.selected_column == -1:

            if self.project_instance.graph.vertices[self.selected_column].neighbour_indices is not None:

                dialog = GUI_elements.SetIndicesManuallyDialog()
                dialog.reference_object(self, self.selected_column)
                dialog.gen_layout()
                dialog.exec_()

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
            print(str(self.project_instance.filename_full))
            message = QtWidgets.QMessageBox()
            message.setText(str(self.project_instance.filename_full))
            message.exec_()

    def show_stats_trigger(self):

        if self.project_loaded:
            self.project_instance.summarize_stats()
            print(self.project_instance.display_stats_string)
            print(str(self.project_instance.num_inconsistencies))
            message = QtWidgets.QMessageBox()
            message.setText(self.project_instance.display_stats_string)
            message.exec_()

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

                self.project_instance.graph.vertices[self.selected_column].force_species(h)

                self.control_window.lbl_column_species.setText(
                    'Atomic species: ' + self.project_instance.graph.vertices[self.selected_column].atomic_species)
                self.control_window.lbl_confidence.setText(
                    'Confidence: ' + str(self.project_instance.graph.vertices[self.selected_column].confidence))
                self.project_instance.graph.vertices[self.selected_column].flag_1 = False
                self.overlay_objects[self.selected_column] =\
                    self.set_species_colors(self.overlay_objects[self.selected_column], self.selected_column)

                self.control_window.draw_histogram()

                self.overlay_objects[self.selected_column] = self.set_species_colors(
                    self.overlay_objects[self.selected_column], self.selected_column)

    def set_level_trigger(self):

        if self.project_loaded and not (self.selected_column == -1):
            items = ('down', 'up')
            item, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Set", "Level", items, 0, False)
            if ok_pressed and item:
                if item == 'down':
                    self.project_instance.columns[self.selected_column].level = 0
                else:
                    self.project_instance.columns[self.selected_column].level = 1
                self.control_window.lbl_column_level.setText(
                    'Level: ' + str(self.project_instance.columns[self.selected_column].level))
                self.overlay_objects[self.selected_column] = self.set_species_colors(
                    self.overlay_objects[self.selected_column], self.selected_column)

    def continue_detection_trigger(self):

        if self.project_loaded:
            items = ('s', 't', 'other')
            item, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Set", "Search type", items, 0, False)
            if ok_pressed and item:
                self.statusBar().showMessage('Working...')
                self.project_instance.redraw_search_mat()
                self.project_instance.column_finder(item)
                self.update_display()

    def restart_detection_trigger(self):

        if self.project_loaded:
            self.statusBar().showMessage('Working...')
            self.deselect_trigger()
            self.project_instance.delete_columns()
            self.previous_pos_obj = None
            self.previous_overlay_obj = None
            self.update_display()

    def continue_analysis_trigger(self):

        if self.project_loaded and not self.selected_column == -1:

            if self.control_window.debug_box.isVisible():

                items = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '21', '22', '23', 'other')
                item, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Set", "Search step", items, 0, False)
                if ok_pressed and item:
                    self.statusBar().showMessage('Analyzing... This may take a long time...')
                    sys.setrecursionlimit(10000)
                    self.project_instance.column_analyser(self.selected_column, item)
                    self.update_central_widget()

            else:

                items = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '21', '22', '23', 'other')
                item, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Set", "Search step", items, 0, False)
                if ok_pressed and item:
                    self.statusBar().showMessage('Analyzing... This may take a long time...')
                    sys.setrecursionlimit(10000)
                    self.project_instance.column_analyser(self.selected_column, item)
                    self.update_central_widget()

    def restart_analysis_trigger(self):

        # This is a temporary function:
        for x in range(0, self.project_instance.num_columns):
            self.project_instance.columns[x].confidence = 0.0
            self.project_instance.columns[x].h_index = 6
            self.project_instance.columns[x].is_in_precipitate = False
            self.project_instance.columns[x].set_by_user = False

            for y in range(0, core.SuchSoftware.num_selections):
                self.project_instance.columns[x].prob_vector[y] = 1.0

            self.project_instance.renorm_prop(x)
            self.project_instance.redefine_species(x)
            self.project_instance.columns[x].level = 0

        self.project_instance.precipitate_boarder = np.ndarray([1], dtype=int)
        self.project_instance.boarder_size = 0
        self.project_instance.summarize_stats()
        self.project_instance.reset_all_flags()
        self.update_display()
        self.statusBar().showMessage('Reset all prob_vector\'s')

    def invert_levels_trigger(self):

        if self.project_loaded:
            self.project_instance.invert_levels()
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
        else:
            self.control_window.debug_box.hide()
            temp = self.selected_column
            self.selected_column = -1
            self.draw_connections(0)
            self.selected_column = temp

    def add_mark_trigger(self):

        print('---')
        if self.project_loaded and not self.selected_column == -1:
            string = 'Column report:' +\
                '   Index: ' + str(self.selected_column) +\
                '\n   flag 1: ' + str(self.project_instance.columns[self.selected_column].flag_1) +\
                '\n   flag 2: ' + str(self.project_instance.columns[self.selected_column].flag_2) +\
                '\n   flag 3: ' + str(self.project_instance.columns[self.selected_column].flag_3) +\
                '\n   flag 4: ' + str(self.project_instance.columns[self.selected_column].flag_4) +\
                '\n   Set by User: ' + str(self.project_instance.columns[self.selected_column].set_by_user) +\
                '\n   Is_edge_columns: ' + str(self.project_instance.columns[self.selected_column].is_edge_column) +\
                '\n   Is Popular: ' + str(self.project_instance.columns[self.selected_column].is_popular) +\
                '\n   Is unpopular: ' + str(self.project_instance.columns[self.selected_column].is_unpopular) +\
                '\n   Is in precipitate: ' + str(self.project_instance.columns[self.selected_column].is_in_precipitate) +\
                '\n   h_index: ' + str(self.project_instance.columns[self.selected_column].h_index) +\
                '\n   species: ' + self.project_instance.columns[self.selected_column].atomic_species

            print(string)

    def invert_precipitate_columns_trigger(self):

        if self.project_loaded:

            self.project_instance.invert_levels(True)

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

    def display_deviations_trigger(self):

        if self.project_loaded and self.project_instance.num_columns > 0 and self.control_instance is not None:

            deviations = 0
            flags = 0
            correct_flags = 0
            erroneous_flags = 0
            popular = 0
            unpopular = 0

            for x in range(0, self.project_instance.num_columns):

                if self.project_instance.columns[x].h_index == self.control_instance.columns[x].h_index:
                    pass
                else:
                    deviations = deviations + 1

                if self.project_instance.columns[x].is_unpopular or self.project_instance.columns[x].is_popular:

                    flags = flags + 1
                    if self.project_instance.columns[x].h_index == self.control_instance.columns[x].h_index:
                        erroneous_flags = erroneous_flags + 1
                    else:
                        correct_flags = correct_flags + 1
                    if self.project_instance.columns[x].is_unpopular:
                        unpopular = unpopular + 1
                    if self.project_instance.columns[x].is_popular:
                        popular = popular + 1

            undetected_errors = deviations - correct_flags

            print('Deviations: ' + str(deviations))
            message = QtWidgets.QMessageBox()
            message.setText('Flags: ' + str(flags) + '\nDeviations: ' + str(deviations) + '\nPercentage: ' + str(
                deviations / self.project_instance.num_columns) + '\nCorrect flags: ' + str(correct_flags) +
                '\nErroneous flags: ' + str(erroneous_flags) + '\nundetected errors: ' + str(undetected_errors) +
                '\nPopular: ' + str(popular) + '\nUnpopular: ' + str(unpopular))
            message.exec_()

    def test_consistency_trigger(self):

        # Highjacked

        if self.project_loaded:

            for x in range(0, self.project_instance.num_columns):

                self.project_instance.columns[x].level = 0

    def clear_flags_trigger(self):

        if self.project_loaded and self.project_instance.num_columns > 0:

            self.project_instance.reset_all_flags()
            self.update_central_widget()

    def there_is_no_help_trigger(self):

        message = QtWidgets.QMessageBox()
        message.setText('Mental Helses Hjelpetelefon er åpen døgnet rundt på 116 123.')
        message.exec_()

    def toggle_precipitate_trigger(self):

        if self.project_loaded and not self.selected_column == -1:

            if self.control_window.chb_precipitate_column.isChecked():

                self.project_instance.columns[self.selected_column].is_in_precipitate = True

            else:

                self.project_instance.columns[self.selected_column].is_in_precipitate = False

            self.project_instance.summarize_stats()

    def toggle_show_trigger(self):

        if self.selected_column == -1:
            pass
        else:
            self.project_instance.columns[self.selected_column].show_in_overlay = not self.project_instance.columns[
                self.selected_column].show_in_overlay
            if self.project_instance.columns[self.selected_column].show_in_overlay:
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

        x_coor = self.pos_objects[self.selected_column].x_0
        y_coor = self.pos_objects[self.selected_column].y_0
        r = self.project_instance.r

        self.project_instance.columns[self.selected_column].x = x_coor
        self.project_instance.columns[self.selected_column].y = y_coor

        if self.project_instance.im_width - r > x_coor > r and self.project_instance.im_height - r > y_coor > r:

            self.project_instance.columns[self.selected_column].avg_gamma, self.project_instance.columns[
                self.selected_column].peak_gamma = mat_op.average(self.project_instance.im_mat, x_coor, y_coor, r)

        else:

            self.project_instance.im_mat = mat_op.gen_framed_mat(self.project_instance.im_mat, r)
            self.project_instance.columns[self.selected_column].avg_gamma, self.project_instance.columns[
                self.selected_column].peak_gamma = mat_op.average(self.project_instance.im_mat, x_coor + r, y_coor + r,
                                                                  r)
            self.project_instance.im_mat = mat_op.gen_de_framed_mat(self.project_instance.im_mat, r)

        self.project_instance.redraw_centre_mat()

        self.control_window.lbl_column_x_pos.setText('x: ' + str(self.pos_objects[self.selected_column].x_0))
        self.control_window.lbl_column_y_pos.setText('y: ' + str(self.pos_objects[self.selected_column].y_0))
        self.control_window.lbl_column_peak_gamma.setText(
            'Peak gamma: ' + str(self.project_instance.columns[self.selected_column].peak_gamma))
        self.control_window.lbl_column_avg_gamma.setText(
            'Avg gamma: ' + str(self.project_instance.columns[self.selected_column].avg_gamma))
        self.control_window.lbl_neighbours.setText('Nearest neighbours: ' + str(self.project_instance.columns[self.selected_column].neighbour_indices))

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

    def set_style_trigger(self):

        if self.project_loaded:
            message = QtWidgets.QMessageBox()
            message.setText('Not implemented yet')
            message.exec_()

    def toggle_si_trigger(self):

        if self.project_loaded:
            for x in range(0, self.project_instance.num_columns):
                if self.project_instance.columns[x].h_index == 0:
                    if self.control_window.chb_si_columns.isChecked():
                        self.project_instance.columns[x].show_in_overlay = True
                        self.overlay_objects[x].show()
                    else:
                        self.project_instance.columns[x].show_in_overlay = False
                        self.overlay_objects[x].hide()

    def toggle_cu_trigger(self):

        if self.project_loaded:
            for x in range(0, self.project_instance.num_columns):
                if self.project_instance.columns[x].h_index == 1:
                    if self.control_window.chb_cu_columns.isChecked():
                        self.project_instance.columns[x].show_in_overlay = True
                        self.overlay_objects[x].show()
                    else:
                        self.project_instance.columns[x].show_in_overlay = False
                        self.overlay_objects[x].hide()

    def toggle_al_trigger(self):

        if self.project_loaded:
            for x in range(0, self.project_instance.num_columns):
                if self.project_instance.columns[x].h_index == 3:
                    if self.control_window.chb_al_columns.isChecked():
                        if self.control_window.chb_al_mesh.isChecked():
                            self.project_instance.columns[x].show_in_overlay = True
                            self.overlay_objects[x].show()
                        else:
                            if self.project_instance.columns[x].is_in_precipitate:
                                self.project_instance.columns[x].show_in_overlay = True
                                self.overlay_objects[x].show()
                            else:
                                self.project_instance.columns[x].show_in_overlay = False
                                self.overlay_objects[x].hide()
                    else:
                        self.project_instance.columns[x].show_in_overlay = False
                        self.overlay_objects[x].hide()

    def toggle_ag_trigger(self):

        if self.project_loaded:
            for x in range(0, self.project_instance.num_columns):
                if self.project_instance.columns[x].h_index == 4:
                    if self.control_window.chb_ag_columns.isChecked():
                        self.project_instance.columns[x].show_in_overlay = True
                        self.overlay_objects[x].show()
                    else:
                        self.project_instance.columns[x].show_in_overlay = False
                        self.overlay_objects[x].hide()

    def toggle_mg_trigger(self):

        if self.project_loaded:
            for x in range(0, self.project_instance.num_columns):
                if self.project_instance.columns[x].h_index == 5:
                    if self.control_window.chb_mg_columns.isChecked():
                        self.project_instance.columns[x].show_in_overlay = True
                        self.overlay_objects[x].show()
                    else:
                        self.project_instance.columns[x].show_in_overlay = False
                        self.overlay_objects[x].hide()

    def toggle_un_trigger(self):

        if self.project_loaded:
            for x in range(0, self.project_instance.num_columns):
                if self.project_instance.columns[x].h_index == 6:
                    if self.control_window.chb_un_columns.isChecked():
                        self.project_instance.columns[x].show_in_overlay = True
                        self.overlay_objects[x].show()
                    else:
                        self.project_instance.columns[x].show_in_overlay = False
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

                    if self.project_instance.columns[x].is_in_precipitate:
                        self.project_instance.columns[x].show_in_overlay = True
                        self.overlay_objects[x].show()
                    else:
                        self.project_instance.columns[x].show_in_overlay = False
                        self.overlay_objects[x].hide()

            else:

                self.control_window.chb_columns.setChecked(True)
                self.toggle_column_trigger()

    def column_selected(self, i):

        if i == -1:
            self.deselect_trigger()
        else:

            self.control_window.lbl_column_index.setText('Column index: ' + str(i))
            self.control_window.lbl_column_x_pos.setText('x: ' + str(self.project_instance.columns[i].x))
            self.control_window.lbl_column_y_pos.setText('y: ' + str(self.project_instance.columns[i].y))
            self.control_window.lbl_column_peak_gamma.setText(
                'Peak gamma: ' + str(self.project_instance.columns[i].peak_gamma))
            self.control_window.lbl_column_avg_gamma.setText('Avg gamma: ' + str(self.project_instance.columns[i].avg_gamma))
            self.control_window.lbl_column_species.setText(
                'Atomic species: ' + self.project_instance.columns[i].atomic_species)
            self.control_window.lbl_column_level.setText('Level: ' + str(self.project_instance.columns[i].level))
            self.control_window.lbl_confidence.setText('Confidence: ' + str(self.project_instance.columns[i].confidence))
            self.control_window.lbl_neighbours.setText('Nearest neighbours: ' + str(self.project_instance.columns[i].neighbour_indices))

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
            self.control_window.chb_show.setChecked(self.project_instance.columns[i].show_in_overlay)
            self.control_window.chb_show.blockSignals(False)
            self.control_window.chb_precipitate_column.blockSignals(True)
            self.control_window.chb_precipitate_column.setChecked(self.project_instance.columns[i].is_in_precipitate)
            self.control_window.chb_precipitate_column.blockSignals(False)
            self.control_window.chb_move.blockSignals(True)
            self.control_window.chb_move.setChecked(False)
            self.control_window.chb_move.blockSignals(False)

            self.control_window.btn_set_move.setDisabled(True)
            self.control_window.btn_cancel_move.setDisabled(True)

            if not self.selected_column == -1 and self.control_window.debug_box.isVisible():

                for x in range(0, self.project_instance.columns[self.selected_column].n()):

                    if self.project_instance.columns[self.selected_column].neighbour_indices[x] == i:

                        shape, n = self.project_instance.find_shape(self.selected_column, i)
                        print(str(shape) + ': ' + str(i) + ' ' + str(self.selected_column))

            self.selected_column = i

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
        self.graphicScene_7 = QtWidgets.QGraphicsScene()
        self.draw_atomic_sub_graph(1)
        self.graphicsView_7.setScene(self.graphicScene_7)

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
        if self.project_instance.alloy == 0:
            self.control_window.lbl_alloy.setText('Alloy: Al-Mg-Si-(Cu)')
        elif self.project_instance.alloy == 1:
            self.control_window.lbl_alloy.setText('Alloy: Al-Mg-Si')
        else:
            self.control_window.lbl_alloy.setText('Alloy: Unknown')
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

        dummy_instance = GUI_elements.InteractivePosColumn(0, 0, 0, 0)
        self.pos_objects = np.ndarray([1], dtype=type(dummy_instance))

        if self.project_instance.num_columns > 0:
            for i in range(0, self.project_instance.num_columns):

                custom_ellipse_pos = GUI_elements.InteractivePosColumn(0, 0, 2 * r, 2 * r)
                custom_ellipse_pos.moveBy(self.project_instance.columns[i].x - r,
                                          self.project_instance.columns[i].y - r)
                custom_ellipse_pos.reference_object(self, i)

                if self.project_instance.columns[i].show_in_overlay:
                    custom_ellipse_pos.setPen(self.red_pen)
                else:
                    custom_ellipse_pos.setPen(self.dark_red_pen)

                custom_ellipse_pos.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)

                if i == 0:
                    self.pos_objects[0] = custom_ellipse_pos
                else:
                    self.pos_objects = np.append(self.pos_objects, custom_ellipse_pos)

                self.graphicScene_2.addItem(custom_ellipse_pos)

            if self.selected_column == -1:
                pass
            else:
                self.pos_objects[self.selected_column].select()

    def draw_overlay(self):

        if self.project_loaded:

            self.graphic = QtGui.QPixmap('Images\\Outputs\\Buffers\\raw_image.png')

            if self.control_window.chb_raw_image.isChecked():
                self.graphicScene_3.addPixmap(self.graphic)

            if self.control_window.chb_black_background.isChecked():
                self.graphicScene_3.setBackgroundBrush(self.brush_black)

            r = self.project_instance.r

            dummy_instance = GUI_elements.InteractiveOverlayColumn(0, 0, 0, 0)
            self.overlay_objects = np.ndarray([1], dtype=type(dummy_instance))

            if self.project_instance.num_columns > 0:
                for i in range(0, self.project_instance.num_columns):

                    custom_ellipse_overlay = GUI_elements.InteractiveOverlayColumn(0, 0, r, r)
                    custom_ellipse_overlay.moveBy(self.project_instance.columns[i].x - np.round(r / 2),
                                                  self.project_instance.columns[i].y - np.round(r / 2))
                    custom_ellipse_overlay.reference_object(self, i)
                    custom_ellipse_overlay = self.set_species_colors(custom_ellipse_overlay, i)

                    custom_ellipse_overlay.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)

                    if i == 0:
                        self.overlay_objects[0] = custom_ellipse_overlay
                    else:
                        self.overlay_objects = np.append(self.overlay_objects, custom_ellipse_overlay)

                    self.graphicScene_3.addItem(custom_ellipse_overlay)

    def set_species_colors(self, column_object, i):

        if self.project_instance.columns[i].atomic_species == 'Al':
            column_object.setPen(self.pen_al)
            column_object.setBrush(self.brush_al)
        elif self.project_instance.columns[i].atomic_species == 'Mg':
            column_object.setPen(self.pen_mg)
            column_object.setBrush(self.brush_mg)
        elif self.project_instance.columns[i].atomic_species == 'Si':
            column_object.setPen(self.pen_si)
            column_object.setBrush(self.brush_si)
        elif self.project_instance.columns[i].atomic_species == 'Cu':
            column_object.setPen(self.pen_cu)
            column_object.setBrush(self.brush_cu)
        elif self.project_instance.columns[i].atomic_species == 'Zn':
            column_object.setPen(self.pen_zn)
            column_object.setBrush(self.brush_zn)
        elif self.project_instance.columns[i].atomic_species == 'Ag':
            column_object.setPen(self.pen_ag)
            column_object.setBrush(self.brush_ag)
        elif self.project_instance.columns[i].atomic_species == 'Un':
            column_object.setPen(self.pen_un)
            column_object.setBrush(self.brush_un)
        else:
            column_object.setPen(self.pen_un)
            column_object.setBrush(self.brush_un)

        if self.project_instance.columns[i].level == 0:
            pass
        else:
            column_object.setBrush(self.brush_black)

        if self.control_window.debug_box.isVisible():
            if self.control_instance is not None and not self.project_instance.columns[i].h_index == self.control_instance.columns[i].h_index:
                column_object.setPen(self.pen_un)

            # if self.project_instance.columns[i].is_unpopular:
                # column_object.setPen(self.pen_problem)

            # if self.project_instance.columns[i].is_popular:
                # column_object.setPen(self.pen_ag)

        if self.project_instance.columns[i].show_in_overlay:
            column_object.show()
        else:
            column_object.hide()

        return column_object

    def draw_atomic_graph(self, scale_factor):

        if self.project_loaded and self.project_instance.num_columns > 0:

            r = self.project_instance.r * scale_factor

            # Draw edges
            theta = np.pi / 4
            self.red_pen.setWidth(3)

            for i in range(0, self.project_instance.num_columns):

                i_1 = i

                if self.project_instance.columns[i].neighbour_indices is not None:

                    for y in range(0, self.project_instance.columns[i].n()):

                        i_2 = self.project_instance.columns[i].neighbour_indices[y]

                        r_2 = QtCore.QPointF(2 * scale_factor * self.project_instance.columns[i_1].x, 2 * scale_factor * self.project_instance.columns[i_1].y)
                        r_1 = QtCore.QPointF(2 * scale_factor * self.project_instance.columns[i_2].x, 2 * scale_factor * self.project_instance.columns[i_2].y)

                        r_vec = r_2 - r_1
                        r_mag = np.sqrt((r_2.x() - r_1.x())**2 + (r_2.y() - r_1.y())**2)
                        factor = r / (r_mag * 2)

                        k_1 = r_1 + factor * r_vec
                        k_2 = r_1 + (1 - factor) * r_vec

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
                        # poly_1 = QtGui.QPolygonF(tri_1)
                        poly_2 = QtGui.QPolygonF(tri_2)

                        line = QtWidgets.QGraphicsLineItem(2 * scale_factor * self.project_instance.columns[i_1].x,
                                                           2 * scale_factor * self.project_instance.columns[i_1].y,
                                                           2 * scale_factor * self.project_instance.columns[i_2].x,
                                                           2 * scale_factor * self.project_instance.columns[i_2].y)
                        # head_1 = QtWidgets.QGraphicsPolygonItem(poly_1)
                        head_2 = QtWidgets.QGraphicsPolygonItem(poly_2)

                        is_reciprocated = False
                        is_same_lvl = False

                        for x in range(0, self.project_instance.columns[i_2].n()):

                            if self.project_instance.columns[i_2].neighbour_indices[x] == i:

                                is_reciprocated = True

                        if self.project_instance.columns[i_1].level == self.project_instance.columns[i_2].level:
                            is_same_lvl = True

                        if is_reciprocated and not is_same_lvl:
                            pen = self.pen_connection
                            brush = self.brush_connection
                        else:
                            pen = self.red_pen
                            brush = self.red_brush

                        line.setPen(pen)
                        # head_1.setPen(pen)
                        head_2.setBrush(brush)

                        self.graphicScene_6.addItem(line)
                        if not is_reciprocated or is_same_lvl:
                            self.graphicScene_6.addItem(head_2)

            # Draw vertices

            for i in range(0, self.project_instance.num_columns):

                custom_ellipse_overlay = GUI_elements.InteractiveGraphVertex(0, 0, r, r)
                custom_ellipse_overlay.moveBy(2 * scale_factor * self.project_instance.columns[i].x - np.round(r / 2),
                                              2 * scale_factor * self.project_instance.columns[i].y - np.round(r / 2))
                custom_ellipse_overlay.reference_object(self, i)

                if self.project_instance.columns[i].level == 0:
                    custom_ellipse_overlay.setBrush(self.brush_level_0)
                else:
                    custom_ellipse_overlay.setBrush(self.brush_level_1)

                custom_ellipse_overlay.setPen(self.black_pen)

                custom_ellipse_overlay.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)

                self.graphicScene_6.addItem(custom_ellipse_overlay)

            self.red_pen.setWidth(1)

    def draw_atomic_sub_graph(self, scale_factor):

        if self.project_loaded and self.project_instance.num_columns > 0:

            r = self.project_instance.r * scale_factor

            self.graphicScene_7.setBackgroundBrush(self.brush_background_grey)

            # Draw edges

            for i in range(0, self.project_instance.num_columns):

                i_1 = i

                if self.project_instance.columns[i].neighbour_indices is not None:

                    for y in range(0, 3):

                        i_2 = self.project_instance.columns[i].neighbour_indices[y]
                        line = QtWidgets.QGraphicsLineItem(2 * scale_factor * self.project_instance.columns[i_1].x,
                                                           2 * scale_factor * self.project_instance.columns[i_1].y,
                                                           2 * scale_factor * self.project_instance.columns[i_2].x,
                                                           2 * scale_factor * self.project_instance.columns[i_2].y)

                        is_reciprocated = False

                        for x in range(0, self.project_instance.columns[i_2].n()):

                            if self.project_instance.columns[i_2].neighbour_indices[x] == i:

                                is_reciprocated = True

                        if is_reciprocated:
                            pen = QtGui.QPen(self.pen_connection)
                        else:
                            pen = QtGui.QPen(self.red_pen)

                        if self.project_instance.columns[i_1].level == self.project_instance.columns[i_2].level:
                            pen.setWidth(3)
                        else:
                            pen.setWidth(1)

                        line.setPen(pen)

                        self.graphicScene_7.addItem(line)

            # Draw vertices

            for i in range(0, self.project_instance.num_columns):

                custom_ellipse_overlay = GUI_elements.InteractiveGraphVertex(0, 0, r, r)
                custom_ellipse_overlay.moveBy(2 * scale_factor * self.project_instance.columns[i].x - np.round(r / 2),
                                              2 * scale_factor * self.project_instance.columns[i].y - np.round(r / 2))
                custom_ellipse_overlay.reference_object(self, i)

                if self.project_instance.columns[i].level == 0:
                    custom_ellipse_overlay.setBrush(self.brush_level_0)
                else:
                    custom_ellipse_overlay.setBrush(self.brush_level_1)

                if self.project_instance.columns[i].is_unpopular:
                    custom_ellipse_overlay.setPen(self.pen_inconsistent_unpopular)
                elif self.project_instance.columns[i].is_popular:
                    custom_ellipse_overlay.setPen(self.pen_inconsistent_popular)
                else:
                    custom_ellipse_overlay.setPen(self.pen_consistent)

                custom_ellipse_overlay.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)

                self.graphicScene_7.addItem(custom_ellipse_overlay)

    def draw_boarder(self):

        if self.project_loaded and self.project_instance.boarder_size > 0 and self.control_window.chb_boarders.isChecked():

            for x in range(0, self.project_instance.boarder_size):

                if x == self.project_instance.boarder_size - 1:
                    y = 0
                else:
                    y = x + 1

                i_1 = self.project_instance.precipitate_boarder[x]
                i_2 = self.project_instance.precipitate_boarder[y]
                line = QtWidgets.QGraphicsLineItem(self.project_instance.columns[i_1].x,
                                                   self.project_instance.columns[i_1].y,
                                                   self.project_instance.columns[i_2].x,
                                                   self.project_instance.columns[i_2].y)
                line.setPen(self.pen_boarder)

                self.graphicScene_3.addItem(line)

    def draw_connections(self, index=-1):

        if index == -1:

            if self.project_loaded and self.project_instance.num_columns > 0 and self.control_window.chb_neighbours.isChecked():

                for x in range(0, self.project_instance.num_columns):

                    if not self.project_instance.columns[x].h_index == 6 and self.project_instance.columns[x].show_in_overlay and self.project_instance.columns[x].neighbour_indices is not None:

                        n = 3

                        if self.project_instance.columns[x].h_index == 0 or self.project_instance.columns[x].h_index == 1:
                            n = 3
                        elif self.project_instance.columns[x].h_index == 3:
                            n = 4
                        elif self.project_instance.columns[x].h_index == 5:
                            n = 5

                        for y in range(0, n):

                            i_1 = x
                            i_2 = self.project_instance.columns[x].neighbour_indices[y]
                            line = QtWidgets.QGraphicsLineItem(self.project_instance.columns[i_1].x,
                                                               self.project_instance.columns[i_1].y,
                                                               self.project_instance.columns[i_2].x,
                                                               self.project_instance.columns[i_2].y)
                            line.setPen(self.pen_connection)

                            self.graphicScene_3.addItem(line)

        else:

            if self.project_loaded and self.project_instance.num_columns > 0 and self.project_instance.columns[index].neighbour_indices is not None:

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
                        i_2 = self.project_instance.columns[x].neighbour_indices[y]
                        line = QtWidgets.QGraphicsLineItem(self.project_instance.columns[i_1].x,
                                                           self.project_instance.columns[i_1].y,
                                                           self.project_instance.columns[i_2].x,
                                                           self.project_instance.columns[i_2].y)

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




