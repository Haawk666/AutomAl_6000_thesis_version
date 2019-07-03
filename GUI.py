# By Haakon Tvedt @ NTNU
"""Module containing the Main Window class. This is the top-level GUI and contains the *business logic* of the
interface."""

from PyQt5 import QtWidgets, QtGui, QtCore
import logging
import numpy as np
import core
import GUI_elements
import mat_op

# Instantiate logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.name = 'GUI'


class MainUI(QtWidgets.QMainWindow):
    """Main GUI. Inherits PyQt5.QtWidgets.QMainWindow."""

    def __init__(self, *args):
        super().__init__(*args)

        self.version = [0, 0, 1]

        # Initialize in an 'empty state'
        self.project_instance = core.SuchSoftware('empty')
        self.project_loaded = False
        self.savefile = None
        self.control_instance = None
        self.selected_column = -1
        self.previous_selected_column = -1

        # Tab contents:
        self.no_graphic = QtGui.QPixmap('Images\\no_image.png')
        self.graphic = QtGui.QPixmap('Images\\no_image.png')

        # gs = QGraphicsView
        self.gs_raw_image = GUI_elements.RawImage(ui_obj=self, background=self.no_graphic)
        self.gs_atomic_positions = GUI_elements.AtomicPositions(ui_obj=self, background=self.no_graphic)
        self.gs_overlay_composition = GUI_elements.OverlayComposition(ui_obj=self, background=self.no_graphic)
        self.gs_atomic_graph = GUI_elements.AtomicGraph(ui_obj=self, background=self.no_graphic)
        self.gs_atomic_sub_graph = GUI_elements.AtomicSubGraph(ui_obj=self, background=self.no_graphic)
        self.gs_search_matrix = GUI_elements.RawImage(ui_obj=self, background=self.no_graphic)
        self.gs_fft = GUI_elements.RawImage(ui_obj=self, background=self.no_graphic)

        # gv = QGraphicsView
        self.gv_raw_image = GUI_elements.ZoomGraphicsView(self.gs_raw_image, ui_obj=self, trigger_func=self.key_press_trigger)
        self.gv_atomic_positions = GUI_elements.ZoomGraphicsView(self.gs_atomic_positions, ui_obj=self, trigger_func=self.key_press_trigger)
        self.gv_overlay_composition = GUI_elements.ZoomGraphicsView(self.gs_overlay_composition, ui_obj=self, trigger_func=self.key_press_trigger)
        self.gv_atomic_graph = GUI_elements.ZoomGraphicsView(self.gs_atomic_graph, ui_obj=self, trigger_func=self.key_press_trigger)
        self.gv_atomic_sub_graph = GUI_elements.ZoomGraphicsView(self.gs_atomic_sub_graph, ui_obj=self, trigger_func=self.key_press_trigger)
        self.gv_search_matrix = GUI_elements.ZoomGraphicsView(self.gs_search_matrix, ui_obj=self, trigger_func=self.key_press_trigger)
        self.gv_fft = GUI_elements.ZoomGraphicsView(self.gs_fft, ui_obj=self, trigger_func=self.key_press_trigger)

        # Set up tabs for central widget
        self.tabs = QtWidgets.QTabWidget()

        self.tab_raw_image = self.tabs.addTab(self.gv_raw_image, 'Raw image')
        self.tab_atomic_positions = self.tabs.addTab(self.gv_atomic_positions, 'Atomic positions')
        self.tab_overlay_composition = self.tabs.addTab(self.gv_overlay_composition, 'Overlay composition')
        self.tab_atomic_graph = self.tabs.addTab(self.gv_atomic_graph, 'Atomic graph')
        self.tab_atomic_sub_graph = self.tabs.addTab(self.gv_atomic_sub_graph, 'Atomic sub-graph')
        self.tab_search_matrix = self.tabs.addTab(self.gv_search_matrix, 'Search matrix')
        self.tab_fft = self.tabs.addTab(self.gv_fft, 'FFT image')

        self.setCentralWidget(self.tabs)

        # Create menu bar
        self.menu = GUI_elements.MenuBar(self.menuBar(), self)

        # Create Control window
        self.control_window = GUI_elements.ControlWindow(obj=self)
        self.control_window.debug_box.set_hidden()

        self.control_window_scroll = QtWidgets.QScrollArea()
        self.control_window_scroll.setWidget(self.control_window)
        self.control_window_scroll.setWidgetResizable(True)
        self.control_window_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

        self.control_window_dock = QtWidgets.QDockWidget()
        self.control_window_dock.setWidget(self.control_window_scroll)
        self.control_window_dock.setWindowTitle('Control window')
        self.control_window_dock.setMinimumWidth(300)

        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.control_window_dock)

        # Create terminal window
        self.terminal_window = GUI_elements.Terminal(obj=self)
        self.terminal_window.handler.set_mode(False)
        logger.addHandler(self.terminal_window.handler)

        self.terminal_window_scroll = QtWidgets.QScrollArea()
        self.terminal_window_scroll.setWidget(self.terminal_window)
        self.terminal_window_scroll.setWidgetResizable(True)
        self.terminal_window_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

        self.terminal_window_dock = QtWidgets.QDockWidget()
        self.terminal_window_dock.setWidget(self.terminal_window_scroll)
        self.terminal_window_dock.setWindowTitle('Terminal window')
        self.terminal_window_dock.setMinimumWidth(300)

        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.terminal_window_dock)

        # Generate elements
        self.setWindowTitle('AACC - Automatic Atomic Column Characterizer - By Haakon Tvedt @ NTNU. Version {}.{}.{}'.
                            format(self.version[0], self.version[1], self.version[2]))
        self.resize(1500, 900)
        self.move(50, 30)
        self.statusBar().showMessage('Ready')

        self.debug_mode = False

        # Display
        self.show()

        # Intro
        logger.info('Welcome to AACC by Haakon Tvedt')
        logger.info('GUI version: {}.{}.{}'.format(self.version[0], self.version[1], self.version[2]))
        logger.info('core version: {}.{}.{}\n------------------------'.format(core.SuchSoftware.version[0], core.SuchSoftware.version[1], core.SuchSoftware.version[2]))

    # ----------
    # Business logic methods:
    # ----------

    def set_species(self, h):
        """Set atomic species of selected column"""
        if self.project_loaded and not self.selected_column == -1:
            # Update relevant graphics:
            self.project_instance.graph.vertices[self.selected_column].force_species(h)
            self.gs_overlay_composition.interactive_overlay_objects[self.selected_column].set_style()
            # Update control window info:
            self.control_window.lbl_column_species.setText(
                'Atomic species: ' + self.project_instance.graph.vertices[self.selected_column].atomic_species)
            self.control_window.lbl_confidence.setText(
                'Confidence: ' + str(self.project_instance.graph.vertices[self.selected_column].confidence))
            self.control_window.draw_histogram()
            self.control_window.lbl_prob_vector.setText('Probability vector: {}'.format(self.project_instance.graph.vertices[self.selected_column].prob_vector))

    def set_level(self, level):
        """Set level of selected column"""
        if self.project_loaded and not self.selected_column == -1:
            # Update relevant graphics:
            self.project_instance.graph.vertices[self.selected_column].level = level
            self.gs_overlay_composition.interactive_overlay_objects[self.selected_column].set_style()
            self.gs_atomic_graph.interactive_vertex_objects[self.selected_column].set_style()
            # Update control window info:
            self.control_window.lbl_column_level.setText('Level: {}'.format(level))

    # ----------
    # Self state methods:
    # ----------

    def update_display(self):
        self.update_central_widget()
        self.update_control_window()
        self.statusBar().showMessage('Ready.')

    def update_central_widget(self):
        if self.project_instance is not None:
            mat_op.im_out_static(self.project_instance.im_mat.astype(np.float64), 'Images\Outputs\Buffers\\raw_image.png')
            mat_op.im_out_static(self.project_instance.search_mat.astype(np.float64), 'Images\Outputs\Buffers\search_image.png')
            mat_op.im_out_static(self.project_instance.fft_im_mat.astype(np.float64), 'Images\Outputs\Buffers\FFT.png')
            void = False
        else:
            void = True
        self.update_raw_image(void=void)
        self.update_column_positions(void=void)
        self.update_overlay(void=void)
        self.update_graph()
        self.update_search_matrix(void=void)
        self.update_fft(void=void)

    def update_raw_image(self, void=False):
        if void:
            graphic_ = self.no_graphic
        else:
            graphic_ = QtGui.QPixmap('Images\Outputs\Buffers\\raw_image.png')
        self.gs_raw_image = GUI_elements.RawImage(ui_obj=self, background=graphic_)
        self.gv_raw_image.setScene(self.gs_raw_image)

    def update_column_positions(self, void=False):
        if void:
            graphic_ = self.no_graphic
        else:
            graphic_ = QtGui.QPixmap('Images\Outputs\Buffers\\raw_image.png')
        self.gs_atomic_positions = GUI_elements.AtomicPositions(ui_obj=self, background=graphic_)
        self.gs_atomic_positions.re_draw()
        self.gv_atomic_positions.setScene(self.gs_atomic_positions)

    def update_overlay(self, void=False):
        if void:
            graphic_ = self.no_graphic
        else:
            graphic_ = QtGui.QPixmap('Images\Outputs\Buffers\\raw_image.png')
        self.gs_overlay_composition = GUI_elements.OverlayComposition(ui_obj=self, background=graphic_)
        self.gs_overlay_composition.re_draw()
        self.gv_overlay_composition.setScene(self.gs_overlay_composition)

    def update_graph(self):
        self.gs_atomic_graph = GUI_elements.AtomicGraph(ui_obj=self, scale_factor=2)
        self.gs_atomic_graph.re_draw()
        self.gv_atomic_graph.setScene(self.gs_atomic_graph)

    def update_sub_graph(self):
        pass

    def update_search_matrix(self, void=False):
        if void:
            graphic_ = self.no_graphic
        else:
            graphic_ = QtGui.QPixmap('Images\Outputs\Buffers\search_image.png')
        self.gs_raw_image = GUI_elements.RawImage(ui_obj=self, background=graphic_)
        self.gv_search_matrix.setScene(self.gs_raw_image)

    def update_fft(self, void=False):
        if void:
            graphic_ = self.no_graphic
        else:
            graphic_ = QtGui.QPixmap('Images\Outputs\Buffers\FFT.png')
        self.gs_raw_image = GUI_elements.RawImage(ui_obj=self, background=graphic_)
        self.gv_fft.setScene(self.gs_raw_image)

    def update_control_window(self):
        self.control_window.update_display()

    def column_selected(self, i):
        self.previous_selected_column, self.selected_column = self.selected_column, i
        self.control_window.select_column()
        if not i == -1:
            j = self.previous_selected_column
            if not j == -1:
                self.gs_atomic_positions.interactive_position_objects[j].set_style()
                self.gs_overlay_composition.interactive_overlay_objects[j].set_style()
                self.gs_atomic_graph.interactive_vertex_objects[j].set_style()
            self.gs_atomic_positions.interactive_position_objects[i].set_style()
            self.gs_overlay_composition.interactive_overlay_objects[i].set_style()
            self.gs_atomic_graph.interactive_vertex_objects[i].set_style()

    # ----------
    # Keyboard press methods methods:
    # ----------

    def keyPressEvent(self, event):
        """Handles key-presses when central widget has focus. Used to switch between tabs"""
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

    def key_press_trigger(self, key):
        """Process key-press events from graphic elements"""
        if self.project_loaded and not self.selected_column == -1:
            if self.tabs.currentIndex() == 0:
                pass
            if self.tabs.currentIndex() == 1 or self.tabs.currentIndex() == 2 or self.tabs.currentIndex() == 3:
                if key == QtCore.Qt.Key_1:
                    self.set_species(0)
                elif key == QtCore.Qt.Key_2:
                    self.set_species(1)
                elif key == QtCore.Qt.Key_3:
                    self.set_species(3)
                elif key == QtCore.Qt.Key_4:
                    self.set_species(5)
                elif key == QtCore.Qt.Key_Plus:
                    self.set_level(self.project_instance.graph.vertices[self.selected_column].anti_level())
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
            if self.tabs.currentIndex() == 4:
                pass
            if self.tabs.currentIndex() == 5:
                pass
            if self.tabs.currentIndex() == 6:
                pass

    # ----------
    # Menu triggers:
    # ----------

    def menu_new_trigger(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Select dm3', '')
        if filename[0]:
            self.statusBar().showMessage('Working...')
            self.project_instance = core.SuchSoftware(filename[0])
            self.control_instance = None
            self.project_loaded = True
            self.update_display()
        else:
            self.statusBar().showMessage('Ready')

    def menu_open_trigger(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '')
        if filename[0]:
            self.statusBar().showMessage('Working...')
            self.project_instance = core.SuchSoftware.load(filename[0])
            if self.project_instance is not None:
                if self.control_window.debug_box.visible:
                    self.project_instance.debug_mode = True
                    self.terminal_window.handler.set_mode(True)
                else:
                    self.project_instance.debug_mode = False
                    self.terminal_window.handler.set_mode(False)
                self.control_instance = None
                self.project_loaded = True
                self.savefile = filename[0]
                self.update_display()
            else:
                logger.info('File was not loaded. Something must have gone wrong!')
        else:
            self.statusBar().showMessage('Ready')

    def menu_save_trigger(self):
        if self.savefile is None:
            filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', '')
        else:
            filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', self.savefile)

        if filename[0]:
            self.statusBar().showMessage('Working...')
            self.project_instance.save(filename[0])
            self.update_display()
        else:
            self.statusBar().showMessage('Ready')

    def menu_close_trigger(self):
        self.btn_cancel_move_trigger()
        self.column_selected(-1)
        self.project_instance = None
        self.control_instance = None
        self.project_loaded = False
        self.update_display()

    def menu_exit_trigger(self):
        self.close()

    def menu_view_image_title_trigger(self):
        logger.info(self.project_instance.filename_full)

    def menu_show_stats_trigger(self):
        pass

    def menu_update_display(self):
        self.update_display()

    def menu_toggle_image_control_trigger(self, state):
        if state:
            self.control_window.image_box.set_visible()
        else:
            self.control_window.image_box.set_hidden()

    def menu_toggle_alg_1_control_trigger(self, state):
        if state:
            self.control_window.alg_1_box.set_visible()
        else:
            self.control_window.alg_1_box.set_hidden()

    def menu_toggle_alg_2_control_trigger(self, state):
        if state:
            self.control_window.alg_2_box.set_visible()
        else:
            self.control_window.alg_2_box.set_hidden()

    def menu_toggle_column_control_trigger(self, state):
        if state:
            self.control_window.column_box.set_visible()
        else:
            self.control_window.column_box.set_hidden()

    def menu_toggle_graph_control_trigger(self, state):
        if state:
            self.control_window.graph_box.set_visible()
        else:
            self.control_window.graph_box.set_hidden()

    def menu_toggle_overlay_control_trigger(self, state):
        if state:
            self.control_window.overlay_box.set_visible()
        else:
            self.control_window.overlay_box.set_hidden()

    def menu_image_correction_trigger(self):
        message = QtWidgets.QMessageBox()
        message.setText('Not implemented yet')
        message.exec_()

    def menu_image_filter_trigger(self):
        message = QtWidgets.QMessageBox()
        message.setText('Not implemented yet')
        message.exec_()

    def menu_image_adjustments_trigger(self):
        message = QtWidgets.QMessageBox()
        message.setText('Not implemented yet')
        message.exec_()

    def menu_continue_detection_trigger(self):
        pass

    def menu_restart_detection_trigger(self):
        pass

    def menu_continue_analysis_trigger(self):
        pass

    def menu_restart_analysis_trigger(self):
        pass

    def menu_export_data_trigger(self):
        pass

    def menu_export_raw_image_trigger(self):
        if self.project_loaded:
            filename = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", '', "PNG (*.png);;BMP Files (*.bmp);;JPEG (*.JPEG)")
            if filename[0]:
                self.update_raw_image()
                rect_f = self.gs_raw_image.sceneRect()
                img = QtGui.QImage(rect_f.size().toSize(), QtGui.QImage.Format_ARGB32)
                img.fill(QtCore.Qt.white)
                p = QtGui.QPainter(img)
                self.gs_raw_image.render(p, target=QtCore.QRectF(img.rect()), source=rect_f)
                p.end()
                saved = img.save(filename[0])
                if saved:
                    logger.info('Successfully exported raw image to file!')
                else:
                    logger.error('Could not export image!')

    def menu_export_column_position_image_trigger(self):
        pass

    def menu_export_overlay_image_trigger(self):
        pass

    def menu_export_atomic_graph_trigger(self):
        pass

    def menu_toggle_debug_mode_trigger(self, state):
        if state:
            self.control_window.debug_box.set_visible()
        else:
            self.control_window.debug_box.set_hidden()

    def menu_add_mark_trigger(self):
        logger.info('-------------')

    def menu_clear_flags_trigger(self):
        if self.project_instance is not None:
            self.project_instance.graph.reset_all_flags()

    def menu_set_control_file_trigger(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open control file', '')
        if filename[0]:
            self.control_instance = core.SuchSoftware.load(filename[0])
            if self.control_instance is not None:
                self.project_instance.debug_mode = False
            else:
                logger.info('Control-file was not loaded. Something must have gone wrong!')

    def menu_run_benchmark_trigger(self):
        pass

    def menu_display_deviations_trigger(self):
        pass

    def menu_test_consistency_trigger(self):
        pass

    def menu_invert_precipitate_columns_trigger(self):
        if self.project_loaded:
            self.project_instance.graph.invert_levels()
            self.update_central_widget()
            self.control_window.select_column()

    def menu_ad_hoc_trigger(self):
        pass

    def menu_there_is_no_help_trigger(self):
        message = QtWidgets.QMessageBox()
        message.setText('Not implemented yet')
        message.exec_()

    # ----------
    # Set button triggers:
    # ----------

    def btn_set_threshold_trigger(self):
        if self.project_instance is not None:
            threshold, ok_pressed = QtWidgets.QInputDialog.getDouble(self, "Set", "Threshold value (decimal between 0 and 1):", self.project_instance.threshold, 0, 1, 5)
            if ok_pressed:
                self.project_instance.threshold = threshold
                self.control_window.lbl_detection_threshold.setText('Detection threshold value: {}'.format(self.project_instance.threshold))

    def btn_set_search_size_trigger(self):
        if self.project_instance is not None:
            search_size, ok_pressed = QtWidgets.QInputDialog.getInt(self, "Set", "Search size:",
                                                                    self.project_instance.search_size, 0, 100000, 100)
            if ok_pressed:
                self.project_instance.search_size = search_size
                self.control_window.lbl_search_size.setText('Search size: {}'.format(self.project_instance.search_size))

    def btn_set_scale_trigger(self):
        if self.project_instance is not None:
            scale, ok_pressed = QtWidgets.QInputDialog.getDouble(self, "Set", "Image scale (pm/pixel):", self.project_instance.scale, 0, 10000, 4)
            if ok_pressed:
                self.project_instance.scale = scale
                self.project_instance.r = int(100 / scale)
                self.project_instance.overhead = int(6 * (self.r / 10))
                self.control_window.lbl_scale.setText('Scale (pm / pixel): {}'.format(self.project_instance.scale))
                self.control_window.lbl_atomic_radii.setText('Approx atomic radii (pixels): {}'.format(self.project_instance.r))
                self.control_window.lbl_overhead_radii.setText('Overhead (pixels): {}'.format(self.project_instance.overhead))
                self.project_instance.redraw_search_mat()
                self.update_central_widget()

    def btn_set_alloy_trigger(self):
        if self.project_instance is not None:
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
                self.control_window.lbl_alloy.setText(self.project_instance.alloy_string())

    def btn_set_start_trigger(self):
        pass

    def btn_set_std_1_trigger(self):
        pass

    def btn_set_std_2_trigger(self):
        pass

    def btn_set_std_3_trigger(self):
        pass

    def btn_set_std_4_trigger(self):
        pass

    def btn_set_std_5_trigger(self):
        pass

    def btn_set_std_8_trigger(self):
        pass

    def btn_set_cert_threshold_trigger(self):
        pass

    def btn_find_column_trigger(self):
        pass

    def btn_set_species_trigger(self):
        """Btn-trigger: Run 'set species' dialog."""
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

    def btn_set_level_trigger(self):
        """Btn-trigger: Run 'set level' dialog."""
        if self.project_loaded and not (self.selected_column == -1):

            items = ('0', '1')
            item, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Set", "level", items, 0, False)

            if ok_pressed and item:

                if item == '0':
                    level = 0
                elif item == '1':
                    level = 1
                else:
                    level = 0

                self.set_level(level)

    # ----------
    # Other button triggers:
    # ----------

    def btn_cancel_move_trigger(self):
        pass

    def btn_set_position_trigger(self):
        pass

    def btn_show_stats_trigger(self):
        pass

    def btn_view_image_title_trigger(self):
        pass

    def btn_export_overlay_image_trigger(self):
        pass

    def btn_continue_detection_trigger(self):
        pass

    def btn_restart_detection_trigger(self):
        pass

    def btn_continue_analysis_trigger(self):
        pass

    def btn_restart_analysis_trigger(self):
        pass

    def btn_invert_levels_trigger(self):
        pass

    def btn_delete_trigger(self):
        pass

    def btn_gen_sub_graph(self):
        pass

    def btn_deselect_trigger(self):
        pass

    def btn_new_column_trigger(self):
        pass

    def btn_set_style_trigger(self):
        pass

    def btn_set_indices_trigger(self):
        pass

    def btn_set_indices_2_trigger(self):
        pass

    def btn_set_perturb_mode_trigger(self):
        pass

    def btn_plot_variance_trigger(self):
        pass

    def btn_plot_angles_trigger(self):
        pass

    def btn_save_log_trigger(self):
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', '')
        if filename[0]:
            self.statusBar().showMessage('Working...')
            logger.info('Saving log file...')
            string = self.terminal_window.toPlainText()
            with open(filename[0], 'w') as f:
                for line in iter(string.splitlines()):
                    f.write(line)
            f.close()
            logger.info('Saved log to {}'.format(filename[0]))

    def btn_clear_log_trigger(self):
        self.terminal_window.handler.widget.clear()

    # ----------
    # Checkbox triggers:
    # ----------

    def chb_placeholder_trigger(self):
        pass
