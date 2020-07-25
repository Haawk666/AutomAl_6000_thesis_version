# By Haakon Tvedt @ NTNU
"""Module containing the Main Window class. This is the top-level GUI and contains the *business logic* of the
GUI."""

# Program imports:
import core
import GUI_elements
import GUI_settings
import utils
import test_module
import data_module
# External imports:
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import numpy as np
import logging
# Instantiate logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.name = 'GUI'


class MainUI(QtWidgets.QMainWindow):
    """Main GUI. Inherits PyQt5.QtWidgets.QMainWindow."""

    def __init__(self, *args, settings_file=None):
        super().__init__(*args)

        self.version = [0, 0, 2]
        self.config = settings_file

        # Initialize in an 'empty state'
        self.project_instance = None
        self.savefile = None
        self.control_instance = None
        self.selected_column = -1
        self.previous_selected_column = -1
        self.selection_history = []
        self.perturb_mode = False
        self.lock_views = False
        self.display_mode = 'atomic_species'

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

        # Tab contents:
        self.no_graphic = QtGui.QPixmap('Images\\no_image.png')
        self.graphic = QtGui.QPixmap('Images\\no_image.png')

        # gs = QGraphicsScene
        self.gs_raw_image = GUI_elements.RawImage(ui_obj=self, background=self.no_graphic)
        self.gs_atomic_positions = GUI_elements.AtomicPositions(ui_obj=self, background=self.no_graphic)
        self.gs_overlay_composition = GUI_elements.OverlayComposition(ui_obj=self, background=self.no_graphic)
        self.gs_atomic_graph = GUI_elements.AtomicGraph(ui_obj=self, background=self.no_graphic)
        self.gs_zeta_graph = GUI_elements.AtomicGraph(ui_obj=self, background=self.no_graphic, mode='zeta')
        self.gs_atomic_sub_graph = GUI_elements.AtomicSubGraph(ui_obj=self, background=self.no_graphic)
        self.gs_anti_graph = GUI_elements.AtomicGraph(ui_obj=self, background=self.no_graphic)
        self.gs_info_graph = GUI_elements.InfoGraph(ui_obj=self, background=self.no_graphic)
        self.gs_search_matrix = GUI_elements.RawImage(ui_obj=self, background=self.no_graphic)
        self.gs_fft = GUI_elements.RawImage(ui_obj=self, background=self.no_graphic)

        # gv = QGraphicsView
        self.gv_raw_image = GUI_elements.ZoomGraphicsView(self.gs_raw_image, ui_obj=self, trigger_func=self.key_press_trigger, tab_index=0)
        self.gv_atomic_positions = GUI_elements.ZoomGraphicsView(self.gs_atomic_positions, ui_obj=self, trigger_func=self.key_press_trigger, tab_index=1)
        self.gv_overlay_composition = GUI_elements.ZoomGraphicsView(self.gs_overlay_composition, ui_obj=self, trigger_func=self.key_press_trigger, tab_index=2)
        self.gv_atomic_graph = GUI_elements.ZoomGraphicsView(self.gs_atomic_graph, ui_obj=self, trigger_func=self.key_press_trigger, tab_index=3)
        self.gv_zeta_graph = GUI_elements.ZoomGraphicsView(self.gs_zeta_graph, ui_obj=self, trigger_func=self.key_press_trigger, tab_index=4)
        self.gv_atomic_sub_graph = GUI_elements.ZoomGraphicsView(self.gs_atomic_sub_graph, ui_obj=self, trigger_func=self.key_press_trigger, tab_index=5)
        self.gv_anti_graph = GUI_elements.ZoomGraphicsView(self.gs_anti_graph, ui_obj=self, trigger_func=self.key_press_trigger, tab_index=6)
        self.gv_info_graph = GUI_elements.ZoomGraphicsView(self.gs_info_graph, ui_obj=self, trigger_func=self.key_press_trigger, tab_index=7)
        self.gv_search_matrix = GUI_elements.ZoomGraphicsView(self.gs_search_matrix, ui_obj=self, trigger_func=self.key_press_trigger, tab_index=8)
        self.gv_fft = GUI_elements.ZoomGraphicsView(self.gs_fft, ui_obj=self, trigger_func=self.key_press_trigger, tab_index=9)

        self.gv_list = [
            self.gv_raw_image,
            self.gv_atomic_positions,
            self.gv_overlay_composition,
            self.gv_atomic_graph,
            self.gv_zeta_graph,
            self.gv_atomic_sub_graph,
            self.gv_anti_graph,
            self.gv_info_graph,
            self.gv_search_matrix,
            self.gv_fft
        ]

        # Set up tabs for central widget
        self.tabs = QtWidgets.QTabWidget()

        self.tab_raw_image = self.tabs.addTab(self.gv_raw_image, 'Raw image')
        self.tab_atomic_positions = self.tabs.addTab(self.gv_atomic_positions, 'Atomic positions')
        self.tab_overlay_composition = self.tabs.addTab(self.gv_overlay_composition, 'Overlay composition')
        self.tab_atomic_graph = self.tabs.addTab(self.gv_atomic_graph, 'Atomic graph')
        self.tab_zeta_graph = self.tabs.addTab(self.gv_zeta_graph, 'Zeta graph')
        self.tab_atomic_sub_graph = self.tabs.addTab(self.gv_atomic_sub_graph, 'Atomic sub-graph')
        self.tab_anti_graph = self.tabs.addTab(self.gv_anti_graph, 'Anti-graph')
        self.tab_info_graph = self.tabs.addTab(self.gv_info_graph, 'Info-graph')
        self.tab_search_matrix = self.tabs.addTab(self.gv_search_matrix, 'Search matrix')
        self.tab_fft = self.tabs.addTab(self.gv_fft, 'FFT image')

        self.setCentralWidget(self.tabs)

        # Generate elements
        self.setWindowTitle('AutomAl 6000'.
                            format(self.version[0], self.version[1], self.version[2]))
        self.resize(1500, 900)
        self.move(50, 30)
        self.statusBar().showMessage('Ready')

        self.debug_mode = False

        # Display
        self.show()

        # Intro
        logger.info('Welcome to AutomAl 6000 by Haakon Tvedt')
        logger.info('GUI version: {}.{}.{}'.format(self.version[0], self.version[1], self.version[2]))
        logger.info('core version: {}.{}.{}'.format(core.Project.version[0], core.Project.version[1], core.Project.version[2]))
        logger.info('Build version: Alpha 1.0\n------------------------')

    # ----------
    # Business logic methods:
    # ----------

    def set_species(self, h):
        """Set atomic species of selected column"""
        if self.project_instance is not None and not self.selected_column == -1:
            # Update relevant graphics:
            self.project_instance.graph.set_species(self.selected_column, h)
            self.gs_overlay_composition.re_draw_vertex(self.selected_column)
            self.gs_atomic_graph.redraw_neighbourhood(self.selected_column)
            self.gs_zeta_graph.redraw_neighbourhood(self.selected_column)
            # Update control window info:
            self.control_window.select_column()

    def set_level(self, zeta):
        """Set level of selected column"""
        if self.project_instance is not None and not self.selected_column == -1:
            # Update relevant graphics:
            self.project_instance.graph.set_zeta(self.selected_column, zeta)
            self.gs_overlay_composition.re_draw_vertex(self.selected_column)
            self.gs_atomic_graph.interactive_vertex_objects[self.selected_column].set_style()
            self.gs_atomic_graph.redraw_neighbourhood(self.selected_column)
            self.gs_zeta_graph.interactive_vertex_objects[self.selected_column].set_style()
            self.gs_zeta_graph.redraw_neighbourhood(self.selected_column)
            # Update control window info:
            self.control_window.lbl_column_level.setText('Level: {}'.format(zeta))

    # ----------
    # Self state methods:
    # ----------

    def update_display(self):
        self.update_central_widget()
        self.update_control_window()
        self.sys_message('Ready.')

    def update_central_widget(self):
        if self.project_instance is not None:
            utils.im_out_static(self.project_instance.im_mat.astype(np.float64), 'Images\Outputs\Buffers\\raw_image.png')
            utils.im_out_static(self.project_instance.search_mat.astype(np.float64), 'Images\Outputs\Buffers\search_image.png')
            utils.im_out_static(self.project_instance.fft_im_mat.astype(np.float64), 'Images\Outputs\Buffers\FFT.png')
            void = False
        else:
            void = True
        self.update_raw_image(void=void)
        self.update_column_positions(void=void)
        self.update_overlay(void=void)
        self.update_graph()
        self.update_zeta_graph()

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
        # Do not change scale factor, many bugs activate if scale is not 1!
        self.gs_atomic_graph = GUI_elements.AtomicGraph(ui_obj=self, scale_factor=2)
        self.gv_atomic_graph.setScene(self.gs_atomic_graph)

    def update_zeta_graph(self):
        # Do not change scale factor, many bugs activate if scale is not 1!
        self.gs_zeta_graph = GUI_elements.AtomicGraph(ui_obj=self, scale_factor=2, mode='zeta')
        self.gv_zeta_graph.setScene(self.gs_zeta_graph)

    def update_sub_graph(self):
        self.gs_atomic_sub_graph = GUI_elements.AtomicSubGraph(ui_obj=self)
        self.gv_atomic_sub_graph.setScene(self.gs_atomic_sub_graph)

    def update_anti_graph(self):
        self.gs_anti_graph = GUI_elements.AtomicSubGraph(ui_obj=self)
        self.gv_anti_graph.setScene(self.gs_anti_graph)

    def update_info_graph(self):
        self.gs_info_graph = GUI_elements.InfoGraph(ui_obj=self, scale_factor=2)
        self.gv_info_graph.setScene(self.gs_info_graph)

    def update_search_matrix(self, void=False):
        if void:
            graphic_ = self.no_graphic
        else:
            graphic_ = QtGui.QPixmap('Images\Outputs\Buffers\search_image.png')
        scene = GUI_elements.StaticImage(ui_obj=self, background=graphic_)
        self.gv_search_matrix.setScene(scene)

    def update_fft(self, void=False):
        if void:
            graphic_ = self.no_graphic
        else:
            graphic_ = QtGui.QPixmap('Images\Outputs\Buffers\FFT.png')
        scene = GUI_elements.StaticImage(ui_obj=self, background=graphic_)
        self.gv_fft.setScene(scene)

    def update_control_window(self):
        self.control_window.update_display()

    def column_selected(self, i):
        if self.control_window.chb_move.isChecked():
            pass
        else:
            self.previous_selected_column, self.selected_column = self.selected_column, i
            self.control_window.select_column()
            j = self.previous_selected_column
            if not j == -1:
                self.gs_atomic_positions.interactive_position_objects[j].set_style()
                self.gs_overlay_composition.interactive_overlay_objects[j].set_style()
                self.gs_atomic_graph.interactive_vertex_objects[j].set_style()
                self.gs_zeta_graph.interactive_vertex_objects[j].set_style()
            if not i == -1:
                self.gs_atomic_positions.interactive_position_objects[i].set_style()
                self.gs_overlay_composition.interactive_overlay_objects[i].set_style()
                self.gs_atomic_graph.interactive_vertex_objects[i].set_style()
                self.gs_zeta_graph.interactive_vertex_objects[i].set_style()
            if self.perturb_mode:
                if len(self.selection_history) == 2:
                    if self.tabs.currentIndex() == 3:
                        self.gs_atomic_graph.perturb_edge(self.selection_history[0], self.selection_history[1], self.selected_column)
                    elif self.tabs.currentIndex() == 4:
                        self.gs_zeta_graph.perturb_edge(self.selection_history[0], self.selection_history[1], self.selected_column)
                    self.selection_history = []
                else:
                    self.selection_history.append(self.selected_column)
            if self.control_window.chb_enable_ruler.isChecked():
                projected_distance = self.project_instance.graph.get_projected_separation(self.selected_column, self.previous_selected_column)
                spatial_distance = self.project_instance.graph.get_separation(self.selected_column, self.previous_selected_column)
                expected_hard_sphere_distance = self.project_instance.graph.get_hard_sphere_separation(self.selected_column, self.previous_selected_column)
                string = 'Distance between vertex {} and {}\n' \
                         '    Projected distance: {} pm\n' \
                         '    Spatial distance: {} pm\n' \
                         '    Expected hard-sphere distance: {} pm\n' \
                         '    Deviation from hard sphere: {} pm\n'.format(self.previous_selected_column,
                                                                          self.selected_column, projected_distance,
                                                                          spatial_distance,
                                                                          expected_hard_sphere_distance,
                                                                          expected_hard_sphere_distance - spatial_distance)
                logger.info(string)

    def sys_message(self, msg):
        self.statusBar().showMessage(msg)
        QtWidgets.QApplication.processEvents()

    # ----------
    # Keyboard press methods methods:
    # ----------

    def keyPressEvent(self, event):
        """Handles key-presses when central widget has focus. Used to switch between tabs"""
        self.key_press_trigger(event.key())

    def key_press_trigger(self, key):
        """Process key-press events from graphic elements"""
        if self.project_instance is not None:
            if key == QtCore.Qt.Key_1:
                if not self.selected_column == -1:
                    self.project_instance.graph.vertices[self.selected_column].is_set_by_user = True
                    self.set_species('Si_1')
            elif key == QtCore.Qt.Key_2:
                if not self.selected_column == -1:
                    self.project_instance.graph.vertices[self.selected_column].is_set_by_user = True
                    self.set_species('Cu_1')
            elif key == QtCore.Qt.Key_3:
                if not self.selected_column == -1:
                    self.project_instance.graph.vertices[self.selected_column].is_set_by_user = True
                    self.set_species('Al_1')
            elif key == QtCore.Qt.Key_4:
                if not self.selected_column == -1:
                    self.project_instance.graph.vertices[self.selected_column].is_set_by_user = True
                    self.set_species('Mg_1')
            elif key == QtCore.Qt.Key_Plus:
                if not self.selected_column == -1:
                    self.set_level(self.project_instance.graph.vertices[self.selected_column].anti_zeta())
            elif key == QtCore.Qt.Key_W:
                if self.control_window.chb_move.isChecked():
                    self.gs_atomic_positions.interactive_position_objects[self.selected_column].moveBy(0.0, -1.0)
                else:
                    self.gv_list[self.tabs.currentIndex()].translate_w(10)
            elif key == QtCore.Qt.Key_S:
                if self.control_window.chb_move.isChecked():
                    self.gs_atomic_positions.interactive_position_objects[self.selected_column].moveBy(0.0, 1.0)
                else:
                    self.gv_list[self.tabs.currentIndex()].translate_s(10)
            elif key == QtCore.Qt.Key_A:
                if self.control_window.chb_move.isChecked():
                    self.gs_atomic_positions.interactive_position_objects[self.selected_column].moveBy(-1.0, 0.0)
                else:
                    self.gv_list[self.tabs.currentIndex()].translate_a(10)
            elif key == QtCore.Qt.Key_D:
                if self.control_window.chb_move.isChecked():
                    self.gs_atomic_positions.interactive_position_objects[self.selected_column].moveBy(1.0, 0.0)
                else:
                    self.gv_list[self.tabs.currentIndex()].translate_d(10)
            elif key == QtCore.Qt.Key_E:
                self.btn_align_views_trigger()
            elif key == QtCore.Qt.Key_Z:
                if self.tabs.currentIndex() == 0:
                    self.tabs.setCurrentIndex(-1)
                else:
                    self.tabs.setCurrentIndex(self.tabs.currentIndex() - 1)
            elif key == QtCore.Qt.Key_X:
                if self.tabs.currentIndex() == len(self.gv_list) - 1:
                    self.tabs.setCurrentIndex(0)
                else:
                    self.tabs.setCurrentIndex(self.tabs.currentIndex() + 1)
            elif key == QtCore.Qt.Key_F1:
                if not self.selected_column == -1:
                    self.project_instance.graph.vertices[self.selected_column].flag_1 = not self.project_instance.graph.vertices[self.selected_column].flag_1
                    logger.info('vertex {}, flag 1 set to {}'.format(self.selected_column, self.project_instance.graph.vertices[self.selected_column].flag_1))
            elif key == QtCore.Qt.Key_F2:
                if not self.selected_column == -1:
                    self.project_instance.graph.vertices[self.selected_column].flag_2 = not self.project_instance.graph.vertices[self.selected_column].flag_2
                    logger.info('vertex {}, flag 2 set to {}'.format(self.selected_column, self.project_instance.graph.vertices[self.selected_column].flag_2))
            elif key == QtCore.Qt.Key_F3:
                if not self.selected_column == -1:
                    self.project_instance.graph.vertices[self.selected_column].flag_3 = not self.project_instance.graph.vertices[self.selected_column].flag_3
                    logger.info('vertex {}, flag 3 set to {}'.format(self.selected_column, self.project_instance.graph.vertices[self.selected_column].flag_3))
            elif key == QtCore.Qt.Key_F4:
                if not self.selected_column == -1:
                    self.project_instance.graph.vertices[self.selected_column].flag_4 = not self.project_instance.graph.vertices[self.selected_column].flag_4
                    logger.info('vertex {}, flag 4 set to {}'.format(self.selected_column, self.project_instance.graph.vertices[self.selected_column].flag_4))
            elif key == QtCore.Qt.Key_Space:
                if self.tabs.currentIndex() == 3:
                    self.control_window.chb_perturb_mode.toggle()
                else:
                    pass
            elif key == QtCore.Qt.Key_R:
                if not self.selected_column == -1:
                    self.btn_show_stats_trigger()
            elif key == QtCore.Qt.Key_P:
                if self.control_window.chb_move.isChecked():
                    self.chb_enable_move_trigger(False)
                else:
                    self.chb_enable_move_trigger(True)
            elif key == QtCore.Qt.Key_Enter:
                if self.control_window.chb_move.isChecked():
                    self.btn_set_position_trigger()

    # ----------
    # Menu triggers:
    # ----------

    def menu_new_trigger(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Select dm3', '')
        if filename[0]:
            self.sys_message('Working...')
            self.project_instance = core.Project(filename[0])
            self.savefile = None
            self.control_instance = None
            self.update_display()
            self.btn_configure_classes_trigger()
        else:
            self.sys_message('Ready')

    def menu_open_trigger(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '')
        if filename[0]:
            logger.info('Opening file {}'.format(filename[0]))
            self.sys_message('Working...')
            self.project_instance = core.Project.load(filename[0])
            if self.project_instance is not None:
                if self.control_window.debug_box.visible:
                    self.project_instance.debug_mode = True
                    self.terminal_window.handler.set_mode(True)
                else:
                    self.project_instance.debug_mode = False
                    self.terminal_window.handler.set_mode(False)
                self.control_instance = None
                self.savefile = filename[0]
                self.update_display()
                self.sys_message('Ready')
            else:
                logger.info('File was not loaded. Something must have gone wrong!')
                self.sys_message('Ready')
        else:
            self.sys_message('Ready')

    def menu_save_trigger(self):
        if self.savefile is None:
            filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', '')
        else:
            filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', self.savefile)

        if filename[0]:
            self.sys_message('Working...')
            self.project_instance.save(filename[0])
            self.savefile = filename[0]
            self.update_display()
            self.sys_message('Ready.')
        else:
            self.sys_message('Ready.')

    def menu_close_trigger(self):
        self.sys_message('Working...')
        self.btn_cancel_move_trigger()
        self.column_selected(-1)
        self.project_instance = None
        self.control_instance = None
        self.update_display()
        self.sys_message('Ready.')

    def menu_exit_trigger(self):
        self.close()

    def menu_view_image_title_trigger(self):
        if self.project_instance is not None:
            logger.info(self.project_instance.filename_full)

    def menu_show_stats_trigger(self):
        if self.project_instance is not None:
            self.sys_message('Working...')
            self.project_instance.stats_summary()
            self.sys_message('Ready.')

    def menu_update_display(self):
        self.sys_message('Working...')
        self.update_display()
        self.sys_message('Ready.')

    def menu_toggle_project_control_trigger(self, state):
        if state:
            self.control_window.project_box.set_visible()
        else:
            self.control_window.project_box.set_hidden()

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

    def menu_toggle_sub_graphs_control_trigger(self, state):
        if state:
            self.control_window.sub_graphs_box.set_visible()
        else:
            self.control_window.sub_graphs_box.set_hidden()

    def menu_toggle_anti_graph_control_trigger(self, state):
        if state:
            self.control_window.anti_graph_box.set_visible()
        else:
            self.control_window.anti_graph_box.set_hidden()

    def menu_toggle_info_graph_control_trigger(self, state):
        if state:
            self.control_window.info_graph_box.set_visible()
        else:
            self.control_window.info_graph_box.set_hidden()

    def menu_toggle_analysis_control_trigger(self, state):
        if state:
            self.control_window.analysis_box.set_visible()
        else:
            self.control_window.analysis_box.set_hidden()

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
        self.btn_export_overlay_image_trigger()

    def menu_make_plots_trigger(self):
        self.btn_make_plot_trigger()

    def menu_export_raw_image_trigger(self):
        if self.project_instance is not None:
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
        if self.project_instance is not None:
            filename = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", '', "PNG (*.png);;BMP Files (*.bmp);;JPEG (*.JPEG)")
            if filename[0]:
                self.update_column_positions()
                rect_f = self.gs_atomic_positions.sceneRect()
                img = QtGui.QImage(rect_f.size().toSize(), QtGui.QImage.Format_ARGB32)
                img.fill(QtCore.Qt.white)
                p = QtGui.QPainter(img)
                self.gs_atomic_positions.render(p, target=QtCore.QRectF(img.rect()), source=rect_f)
                p.end()
                saved = img.save(filename[0])
                if saved:
                    logger.info('Successfully exported column positions image to file!')
                else:
                    logger.error('Could not export image!')

    def menu_export_overlay_image_trigger(self):
        if self.project_instance is not None:
            filename = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", '', "PNG (*.png);;BMP Files (*.bmp);;JPEG (*.JPEG)")
            if filename[0]:
                self.update_overlay()
                rect_f = self.gs_overlay_composition.sceneRect()
                img = QtGui.QImage(rect_f.size().toSize(), QtGui.QImage.Format_ARGB32)
                img.fill(QtCore.Qt.white)
                p = QtGui.QPainter(img)
                self.gs_overlay_composition.render(p, target=QtCore.QRectF(img.rect()), source=rect_f)
                p.end()
                saved = img.save(filename[0])
                if saved:
                    logger.info('Successfully exported overlay image to file!')
                else:
                    logger.error('Could not export image!')

    def menu_export_atomic_graph_trigger(self):
        if self.project_instance is not None:
            filename = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", '', "PNG (*.png);;BMP Files (*.bmp);;JPEG (*.JPEG)")
            if filename[0]:
                self.update_graph()
                rect_f = self.gs_atomic_graph.sceneRect()
                img = QtGui.QImage(rect_f.size().toSize(), QtGui.QImage.Format_ARGB32)
                img.fill(QtCore.Qt.white)
                p = QtGui.QPainter(img)
                self.gs_atomic_graph.render(p, target=QtCore.QRectF(img.rect()), source=rect_f)
                p.end()
                saved = img.save(filename[0])
                if saved:
                    logger.info('Successfully exported graph image to file!')
                else:
                    logger.error('Could not export image!')

    def menu_toggle_debug_mode_trigger(self, state):
        if state:
            self.control_window.debug_box.set_visible()
        else:
            self.control_window.debug_box.set_hidden()

    def menu_add_mark_trigger(self):
        logger.info('Column positions: {}'.format(self.gv_atomic_positions.size()))
        logger.info('Graph: {}'.format(self.gv_atomic_positions.size()))
        logger.info('-------------')

    def menu_clear_flags_trigger(self):
        if self.project_instance is not None:
            self.project_instance.graph.reset_all_flags()

    def menu_set_control_file_trigger(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open control file', '')
        if filename[0]:
            self.control_instance = core.Project.load(filename[0])
            if self.control_instance is not None:
                self.project_instance.debug_mode = False
            else:
                logger.info('Control-file was not loaded. Something must have gone wrong!')

    def menu_run_benchmark_trigger(self):
        pass

    def menu_display_deviations_trigger(self):
        if self.project_instance is not None and self.control_instance is not None and self.project_instance.num_columns > 0:
            if not len(self.project_instance.graph.vertices) == len(self.control_instance.graph.vertices):
                msg = 'Could not compare instances!'
                deviation_indices = []
            else:
                deviations = 0
                symmetry_deviations = 0
                deviation_indices = []
                for vertex, control_vertex in zip(self.project_instance.graph.vertices, self.control_instance.graph.vertices):
                    if vertex.species_index == control_vertex.species_index:
                        pass
                    elif vertex.species_index == 0 and control_vertex.species_index == 1:
                        deviations += 1
                        deviation_indices.append(vertex.i)
                    elif vertex.species_index == 1 and control_vertex.species_index == 0:
                        deviations += 1
                        deviation_indices.append(vertex.i)
                    else:
                        deviations += 1
                        symmetry_deviations += 1
                        deviation_indices.append(vertex.i)

                msg = 'Control comparison:----------\n    Deviations: {}\n    Symmetry deviations: {}'.format(deviations, symmetry_deviations)
            message = QtWidgets.QMessageBox()
            message.setText(msg)
            message.exec_()
            for index in deviation_indices:
                msg += '\n        {}'.format(index)
            logger.info(msg)

    def menu_test_consistency_trigger(self):
        pass

    def menu_invert_precipitate_columns_trigger(self):
        if self.project_instance is not None:
            self.project_instance.graph.invert_levels()
            self.update_central_widget()
            self.control_window.select_column()

    def menu_ad_hoc_trigger(self):
        if self.project_instance is not None:
            self.project_instance.species_dict = core.Project.default_dict
            self.project_instance.graph.species_dict = core.Project.default_dict
            default_model = data_module.VertexDataManager.load('default_model')
            default_model.species_dict = core.Project.default_dict
            default_model.save()

    def menu_toggle_tooltips_trigger(self, state):
        self.control_window.mode_tooltip(state)
        self.terminal_window.mode_tooltip(state)
        self.config.set('tooltips', 'tooltips', str(state))
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)

    def menu_set_theme_trigger(self):
        items = ('dark', 'classic')
        item, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Select theme", "Theme:", items, 0, False)
        if ok_pressed and item:
            self.config.set('theme', 'theme', item)
            with open('config.ini', 'w') as configfile:
                self.config.write(configfile)
            message = QtWidgets.QMessageBox()
            message.setText('Save your work and restart the program for the changes to take effect!')
            message.exec_()

    def menu_there_is_no_help_trigger(self):
        message = QtWidgets.QMessageBox()
        message.setText('Not implemented yet')
        message.exec_()

    # ----------
    # Set button triggers:
    # ----------

    def btn_set_model_trigger(self):
        if self.project_instance is not None:
            model_filename = QtWidgets.QFileDialog.getOpenFileName(self, "Load model", '', "")
            if model_filename[0]:
                self.project_instance.graph.active_model = model_filename[0]
            self.control_window.lbl_model.setText('Model: {}'.format(self.project_instance.graph.active_model))

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
                self.project_instance.set_alloy_mat()
                self.control_window.lbl_alloy.setText('Alloy: {}'.format(self.project_instance.alloy_string[self.project_instance.alloy]))

    def btn_set_start_trigger(self):
        if self.project_instance is not None and not self.selected_column == -1:
            self.project_instance.starting_index = self.selected_column
            self.control_window.lbl_starting_index.setText('Default starting index: {}'.format(self.project_instance.starting_index))

    def btn_find_column_trigger(self):
        if self.project_instance is not None:
            index, ok_pressed = QtWidgets.QInputDialog.getInt(self, "Set", "Find column by index:", 0, 0, 100000, 1)
            if ok_pressed:
                if index < self.project_instance.num_columns:
                    self.gs_atomic_positions.interactive_position_objects[index].mouseReleaseEvent(QtWidgets.QGraphicsEllipseItem.mouseReleaseEvent)

    def btn_set_advanced_species_trigger(self):
        if self.project_instance is not None and not self.selected_column == -1:
            items = []
            for item in self.project_instance.species_dict:
                items.append(item)
            item, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Set", "Species", items, 0, False)
            if ok_pressed and item:
                self.set_species(item)
                self.project_instance.graph.vertices[self.selected_column].is_set_by_user = True

    def btn_set_level_trigger(self):
        """Btn-trigger: Run 'set level' dialog."""
        if self.project_instance is not None and not self.selected_column == -1:
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

    def btn_set_map_type_trigger(self):
        if self.project_instance is not None:
            items = ['zeta', 'district', 'separation']
            item, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Set", "Vertex automapping method", items, 0, False)
            if ok_pressed and item:
                self.map_type = item
                self.control_window.lbl_map_type.setText('Map type: {}'.format(item))

    def btn_set_sub_graph_type_trigger(self):
        if self.project_instance is not None:
            items = ('Column-centered', 'Edge-centered', 'Mesh-centered')
            item, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Set", "Sub-graph-type:", items, 0, False)
            if ok_pressed and item:
                if item == 'Column-centered':
                    self.control_window.lbl_sub_graph_type.setText('Sub-graph type: Column-centered')
                elif item == 'Edge-centered':
                    self.control_window.lbl_sub_graph_type.setText('Sub-graph type: Edge-centered')
                elif item == 'Mesh-centered':
                    self.control_window.lbl_sub_graph_type.setText('Sub-graph type: Mesh-centered')

    def btn_set_sub_graph_order_trigger(self):
        if self.project_instance is not None:
            items = ('1st', '2nd', '3rd')
            item, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Set", "Sub-graph-order:", items, 0, False)
            if ok_pressed and item:
                if item == '1st':
                    self.control_window.lbl_sub_graph_order.setText('Sub-graph order: 1st')
                elif item == '2nd':
                    self.control_window.lbl_sub_graph_order.setText('Sub-graph order: 2nd')
                elif item == '3rd':
                    self.control_window.lbl_sub_graph_order.setText('Sub-graph order: 3rd')

    # ----------
    # Other button triggers:
    # ----------

    def btn_new_project_trigger(self):
        self.menu_new_trigger()

    def btn_open_project_trigger(self):
        self.menu_open_trigger()

    def btn_save_project_trigger(self):
        if self.project_instance is not None:
            self.menu_save_trigger()

    def btn_close_project_trigger(self):
        if self.project_instance is not None:
            self.menu_close_trigger()

    def btn_configure_classes_trigger(self):
        GUI_elements.CustomizeClassification(ui_obj=self)

    def btn_cancel_move_trigger(self):
        self.control_window.mode_move(False)
        self.update_central_widget()

    def btn_set_position_trigger(self):
        if self.project_instance is not None:
            self.sys_message('Working...')
            (x, y) = self.gs_atomic_positions.interactive_position_objects[self.selected_column].get_vertex_pos()
            vertex = self.project_instance.graph.vertices[self.selected_column]
            vertex.im_coor_x = x
            vertex.im_coor_y = y
            vertex.spatial_coor_x = x * self.project_instance.scale
            vertex.spatial_coor_y = y * self.project_instance.scale
            vertex.avg_gamma, vertex.peak_gamma = utils.circular_average(
                self.project_instance.im_mat,
                int(x),
                int(y),
                self.project_instance.r
            )
            self.control_window.mode_move(False)
            self.update_central_widget()
            self.sys_message('Ready.')

    def btn_show_stats_trigger(self):
        if self.project_instance is not None:
            self.project_instance.summarize_stats()
            self.project_instance.report(supress_log=False)

    def btn_view_image_title_trigger(self):
        self.menu_view_image_title_trigger()

    def btn_align_views_trigger(self):
        tab = self.tabs.currentIndex()
        coor = self.gv_list[tab].mapToScene(self.gv_list[tab].viewport().rect().center())
        transform = self.gv_list[tab].transform()
        for i, gv in enumerate(self.gv_list):
            if i in [0, 1, 2, 3, 7, 8]:
                gv.resetTransform()
                gv.setTransform(transform)
                gv.centerOn(coor)

    def btn_export_overlay_image_trigger(self):
        GUI_elements.DataExportWizard(ui_obj=self)

    def btn_continue_detection_trigger(self):
        if self.project_instance is not None:
            items = ('Threshold', 'Search size', 'other')
            item, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Set", "Search type", items, 0, False)
            if ok_pressed and item:
                self.sys_message('Working, column detection might take several minutes...')
                self.project_instance.redraw_search_mat()
                if item == 'Search size':
                    self.project_instance.column_detection('s', plot=self.control_window.chb_plot_results.isChecked())
                elif item == 'Threshold':
                    self.project_instance.column_detection('t', plot=self.control_window.chb_plot_results.isChecked())
                else:
                    self.project_instance.column_detection('o', plot=self.control_window.chb_plot_results.isChecked())
                self.update_display()
                self.sys_message('Ready.')

    def btn_restart_detection_trigger(self):
        if self.project_instance is not None:
            self.project_instance.reset_graph()
            self.btn_continue_detection_trigger()

    def btn_redraw_search_mat_trigger(self):
        if self.project_instance is not None:
            self.sys_message('Working...')
            self.project_instance.redraw_search_mat()
            self.update_search_matrix()
            self.sys_message('Ready.')

    def btn_continue_analysis_trigger(self):
        if self.project_instance is not None:

            strings = [
                '0 - Full column characterization algorithm',
                '1 - Basic mappings...',
                '2 - ...The rest',
                '3 - Spatial mapping',
                '4 - Identify edge columns',
                '5 - Basic zeta analysis',
                '6 - Advanced zeta analysis',
                '7 - Apply alpha model',
                '8 - Particle detection',
                '9 - Calculate normalized gamma',
                '10 - Evaluate sub-species',
                '11 - Apply composite model',
                '12 - Reset probability vectors',
                '13 - Reset user-set columns',
                '14 - Search for intersections',
                '15 - Not in use',
                '16 - Map vertex connectivity',
                '17 - Map vertex in-connectivity',
                '18 - Untangling algorithm',
                '19 - Not in use',
                '20 - Not in use',
                '21 - Not in use',
                '22 - Not in use',
                '23 - Plot gamma'
            ]

            string, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Set", "Search step", strings, 0, False)
            if ok_pressed and strings:
                self.statusBar().showMessage('Analyzing... This may take a long time...')
                sys.setrecursionlimit(10000)
                choice = -1
                for k in range(0, len(strings)):
                    if string == strings[k]:
                        choice = k
                if not choice == -1:
                    if self.selected_column == -1:
                        if self.project_instance.starting_index is not None:
                            starting_column = self.project_instance.starting_index
                        else:
                            starting_column = 0
                    else:
                        starting_column = self.selected_column

                    if self.control_window.chb_show_graphic_updates.isChecked():
                        self.project_instance.column_characterization(starting_column, choice, ui_obj=self)
                    else:
                        self.project_instance.column_characterization(starting_column, choice)
                    self.update_display()
                else:
                    logger.error('Invalid selection. Was not able to start column detection.')

    def btn_restart_analysis_trigger(self):
        if self.project_instance is not None:
            self.project_instance.reset_vertex_properties()
            self.btn_continue_analysis_trigger()

    def btn_invert_levels_trigger(self):
        self.menu_invert_precipitate_columns_trigger()

    def btn_set_variant_trigger(self):
        if self.project_instance is not None and not self.selected_column == -1:
            items = ('1', '2', '3', '4')
            item, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Set variant", "Variant: ", items, 0, False)
            if ok_pressed and item:
                self.sys_message('Working...')
                if item == '1':
                    self.project_instance.graph.vertices[self.selected_column].internal_flag_1 = True
                    self.project_instance.graph.vertices[self.selected_column].internal_flag_2 = False
                    self.project_instance.graph.vertices[self.selected_column].internal_flag_3 = False
                    self.project_instance.graph.vertices[self.selected_column].internal_flag_4 = True
                elif item == '2':
                    self.project_instance.graph.vertices[self.selected_column].internal_flag_1 = False
                    self.project_instance.graph.vertices[self.selected_column].internal_flag_2 = True
                    self.project_instance.graph.vertices[self.selected_column].internal_flag_3 = False
                    self.project_instance.graph.vertices[self.selected_column].internal_flag_4 = True
                elif item == '3':
                    self.project_instance.graph.vertices[self.selected_column].internal_flag_1 = False
                    self.project_instance.graph.vertices[self.selected_column].internal_flag_2 = False
                    self.project_instance.graph.vertices[self.selected_column].internal_flag_3 = True
                    self.project_instance.graph.vertices[self.selected_column].internal_flag_4 = True
                elif item == '4':
                    self.project_instance.graph.vertices[self.selected_column].internal_flag_1 = False
                    self.project_instance.graph.vertices[self.selected_column].internal_flag_2 = False
                    self.project_instance.graph.vertices[self.selected_column].internal_flag_3 = False
                    self.project_instance.graph.vertices[self.selected_column].internal_flag_4 = True
                else:
                    pass
                self.sys_message('Ready.')

    def btn_delete_trigger(self):
        if self.project_instance is not None and not self.selected_column == -1:
            ok_pressed = QtWidgets.QMessageBox.question(self, 'Confirm', 'Are you sure you wish to delete this column?', QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if ok_pressed == QtWidgets.QMessageBox.Yes:
                self.sys_message('Working...')
                self.project_instance.graph.remove_vertex(self.selected_column)
                self.project_instance.num_columns -= 1
                self.update_central_widget()
                logger.info('Vertex moved. remember to redraw the search matrix before continuing column detection!')
                self.sys_message('Ready.')

    def btn_print_details_trigger(self):
        if self.project_instance is not None and not self.selected_column == -1:
            self.project_instance.vertex_report(self.selected_column)

    def btn_snap_trigger(self, column_index=None):
        if self.project_instance is not None and not self.selected_column == -1:
            if column_index is None or not column_index:
                i = self.selected_column
            else:
                i = column_index
            coor = self.project_instance.graph.vertices[i].real_coor()
            transform = QtGui.QTransform()
            transform.scale(7.5, 7.5)
            self.gv_raw_image.setTransform(transform)
            self.gv_raw_image.centerOn(coor[0], coor[1])
            self.gv_atomic_positions.setTransform(transform)
            self.gv_atomic_positions.centerOn(coor[0], coor[1])
            self.gv_overlay_composition.setTransform(transform)
            self.gv_overlay_composition.centerOn(coor[0], coor[1])
            transform.scale(0.5, 0.5)
            self.gv_atomic_graph.setTransform(transform)
            self.gv_atomic_graph.centerOn(2 * coor[0], 2 * coor[1])

    def btn_gen_sub_graph(self):
        if self.project_instance is not None:
            if self.project_instance.num_columns > 0:
                if len(self.project_instance.graph.vertices[0].district) > 0:
                    if not self.selected_column == -1:
                        self.project_instance.graph.build_local_maps()
                        sub_graph = self.project_instance.graph.get_column_centered_subgraph(self.selected_column)
                        self.gs_atomic_sub_graph = GUI_elements.AtomicSubGraph(ui_obj=self, sub_graph=sub_graph, scale_factor=4)
                        self.gv_atomic_sub_graph.setScene(self.gs_atomic_sub_graph)

    def btn_refresh_graph_trigger(self):
        self.sys_message('Working...')
        self.project_instance.graph.refresh_graph()
        self.sys_message('Ready.')

    def btn_refresh_mesh_trigger(self):
        if self.project_instance is not None and \
                self.project_instance.num_columns > 0 and \
                len(self.project_instance.graph.vertices[0].neighbour_indices) > 0:
            logger.info('Mapping meshes...')
            self.sys_message('Working...')
            if self.project_instance.starting_index is not None:
                self.project_instance.graph.map_meshes(self.project_instance.starting_index)
            elif not self.selected_column == -1:
                self.project_instance.graph.map_meshes(self.selected_column)
            else:
                self.project_instance.graph.map_meshes(np.floor(self.project_instance.num_columns / 2))
            logger.info('Meshes mapped. Updating graphics')
            self.gs_atomic_graph.re_draw_mesh_details()
            self.gs_info_graph.re_draw_mesh_details()
            self.sys_message('Ready.')

    def btn_deselect_trigger(self):
        self.column_selected(-1)

    def btn_new_column_trigger(self):
        if self.project_instance is not None:
            self.sys_message('Working...')
            new_vertex = core.graph_2.Vertex(
                self.project_instance.graph.order,
                self.project_instance.im_width / 2 + 0.1,
                self.project_instance.im_height / 2 + 0.1,
                self.project_instance.r,
                self.project_instance.scale,
                parent_graph=self.project_instance.graph
            )
            self.project_instance.graph.add_vertex(new_vertex)
            self.project_instance.num_columns += 1
            index = new_vertex.i
            self.update_central_widget()
            self.column_selected(index)
            self.sys_message('Setting move mode to on...')
            self.control_window.mode_move(True)

    def btn_set_style_trigger(self):
        GUI_elements.CustomizeOverlay(ui_obj=self)

    def btn_show_all_trigger(self):
        pass

    def btn_hide_all_trigger(self):
        pass

    def btn_test_trigger(self):
        pass

    def btn_test_data_manager_trigger(self):
        test_module.test_statistical_basics()

    def btn_show_model_prediction_trigger(self):
        if not self.selected_column == -1:
            vertex = self.project_instance.graph.vertices[self.selected_column]
            model = data_module.VertexDataManager.load(self.project_instance.graph.active_model)
            alpha_dict_ = {
                'alpha_max': vertex.alpha_max,
                'alpha_min': vertex.alpha_min
            }
            dict_ = {
                'alpha_max': vertex.alpha_max,
                'alpha_min': vertex.alpha_min,
                'theta_max': vertex.theta_max,
                'theta_min': vertex.theta_min,
                'theta_angle_mean': vertex.theta_angle_mean,
                'normalized_peak_gamma': vertex.normalized_peak_gamma,
                'normalized_avg_gamma': vertex.normalized_avg_gamma,
            }
            alpha_prediction = model.calc_prediction(alpha_dict_)
            full_prediction = model.calc_prediction(dict_)

            string = 'Model prediction for vertex {}:\n'.format(vertex.i)
            string += '    Attributes:\n'
            string += '        Alpha max: {}\n'.format(vertex.alpha_max)
            string += '        Alpha min: {}\n'.format(vertex.alpha_min)
            string += '        Theta max: {}\n'.format(vertex.theta_max)
            string += '        Theta min: {}\n'.format(vertex.theta_min)
            string += '        Theta mean: {}\n'.format(vertex.theta_angle_mean)
            string += '        Normalized peak gamma: {}\n'.format(vertex.normalized_peak_gamma)
            string += '        Normalized avg gamma: {}\n'.format(vertex.normalized_avg_gamma)
            string += '    Alpha prediction:\n'
            for key, value in alpha_prediction.items():
                string += '        {}: {}\n'.format(key, value)
            string += '    Full prediction:\n'
            for key, value in full_prediction.items():
                string += '        {}: {}\n'.format(key, value)
            logger.info(string)

    def btn_crash_trigger(self):
        raise IndexError

    def btn_make_plot_trigger(self):
        message = QtWidgets.QMessageBox()
        message.setText('Not available in this version!')
        message.exec_()
        # GUI_elements.PlotWizard(ui_obj=self)

    def btn_print_distances_trigger(self):
        si_radii = core.Project.si_radii
        cu_radii = core.Project.cu_radii
        zn_radii = core.Project.zn_radii
        al_radii = core.Project.al_radii
        ag_radii = core.Project.ag_radii
        mg_radii = core.Project.mg_radii

        si_si = 2 * si_radii
        si_cu = si_radii + cu_radii
        si_al = si_radii + al_radii
        si_mg = si_radii + mg_radii

        cu_cu = 2 * cu_radii
        cu_si = cu_radii + si_radii
        cu_al = cu_radii + al_radii
        cu_mg = cu_radii + mg_radii

        al_al = al_radii + al_radii
        al_si = al_radii + si_radii
        al_cu = al_radii + cu_radii
        al_mg = al_radii + mg_radii

        mg_mg = mg_radii + mg_radii
        mg_si = mg_radii + si_radii
        mg_cu = mg_radii + cu_radii
        mg_al = mg_radii + al_radii

        string = 'Inter-atomic hard-sphere spatial distances in pico-metres:----------\n' \
                 '    Si <-> Si: \t{} \n' \
                 '    Si <-> Cu: \t{} \n' \
                 '    Si <-> Al: \t{} \n' \
                 '    Si <-> Mg: \t{} \n\n' \
                 '    Cu <-> Cu: \t{} \n' \
                 '    Cu <-> Si: \t{} \n' \
                 '    Cu <-> Al: \t{} \n' \
                 '    Cu <-> Mg: \t{} \n\n' \
                 '    Al <-> Al: \t{} \n' \
                 '    Al <-> Si: \t{} \n' \
                 '    Al <-> Cu: \t{} \n' \
                 '    Al <-> Mg: \t{} \n\n' \
                 '    Mg <-> Mg: \t{} \n' \
                 '    Mg <-> Si: \t{} \n' \
                 '    Mg <-> Cu: \t{} \n' \
                 '    Mg <-> Al: \t{} '.format(si_si, si_cu, si_al, si_mg,
                                               cu_cu, cu_si, cu_al, cu_mg,
                                               al_al, al_si, al_cu, al_mg,
                                               mg_mg, mg_si, mg_cu, mg_al)

        logger.info(string)

    def btn_build_anti_graph_trigger(self):
        if self.project_instance is not None:
            if self.project_instance.num_columns > 0:
                if len(self.project_instance.graph.vertices[0].district) > 0:
                    self.sys_message('Working...')
                    anti_graph = self.project_instance.graph.get_anti_graph()
                    self.gs_anti_graph = GUI_elements.AntiGraph(ui_obj=self, scale_factor=2, graph=anti_graph)
                    self.gv_anti_graph.setScene(self.gs_anti_graph)
                    logger.info('Got anti-graph!')
                    self.sys_message('Ready')

    def btn_build_info_graph_trigger(self):
        if self.project_instance is not None:
            if self.project_instance.num_columns > 0:
                if len(self.project_instance.graph.vertices[0].district) > 0:
                    self.sys_message('Working...')
                    self.gs_info_graph = GUI_elements.InfoGraph(ui_obj=self, scale_factor=2)
                    self.gv_info_graph.setScene(self.gs_info_graph)
                    self.sys_message('Ready')

    def btn_pca_trigger(self):
        message = QtWidgets.QMessageBox()
        message.setText('Not available in this version..')
        message.exec_()
        # GUI_elements.PcaWizard(ui_obj=self)

    def btn_calc_models_trigger(self):
        GUI_elements.CalcModels(ui_obj=self)

    def btn_plot_models_trigger(self):
        if self.project_instance is not None:
            GUI_elements.PlotModels(ui_obj=self, model=self.project_instance.graph.active_model)
        else:
            GUI_elements.PlotModels(ui_obj=self, model='default_model')

    def btn_save_log_trigger(self):
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save log-file', '')
        if filename[0]:
            self.sys_message('Working...')
            logger.info('Saving log file...')
            string = self.terminal_window.handler.widget.toPlainText()
            with open(filename[0], 'w') as f:
                for line in iter(string.splitlines()):
                    f.write('{}\n'.format(line))
            f.close()
            self.sys_message('Ready.')
            logger.info('Saved log to {}'.format(filename[0]))

    def btn_clear_log_trigger(self):
        self.terminal_window.handler.widget.clear()

    def btn_eval_trigger(self, string):
        try:
            eval(string)
        except:
            logger.info('The input\n    {}\n    threw an exception!'.format(string))
        else:
            logger.info('User input: {}'.format(string))

    # ----------
    # Checkbox triggers:
    # ----------

    def chb_lock_views_trigger(self, state):
        if state:
            self.btn_align_views_trigger()
            self.lock_views = True
        else:
            self.lock_views = False

    def chb_plot_results_trigger(self, state):
        pass

    def chb_automatic_threshold_trigger(self, state):
        pass

    def chb_toggle_positions_trigger(self, state):
        if self.project_instance is not None:
            if self.project_instance.num_columns > 0:
                for position_graphic in self.gs_atomic_positions.interactive_position_objects:
                    if state:
                        position_graphic.show()
                    else:
                        position_graphic.hide()

    def chb_show_graphic_updates_trigger(self, state):
        pass

    def chb_precipitate_column_trigger(self, state):
        if self.project_instance is not None and not self.selected_column == -1:
            self.project_instance.graph.vertices[self.selected_column].is_in_precipitate = state

    def chb_show_trigger(self, state):
        if self.project_instance is not None and not self.selected_column == -1:
            self.project_instance.graph.vertices[self.selected_column].show_in_overlay = state
            self.gs_atomic_positions.interactive_position_objects[self.selected_column].set_style()
            self.gs_overlay_composition.interactive_overlay_objects[self.selected_column].set_style()

    def chb_enable_move_trigger(self, state):
        self.sys_message('Working...')
        self.control_window.mode_move(state)
        self.sys_message('Ready.')

    def chb_set_perturb_mode_trigger(self, state):
        self.perturb_mode = state
        if not self.perturb_mode:
            logger.info('Permute mode is off!')
            self.sys_message('Working...')
            self.selection_history = []
            self.sys_message('Ready.')
        else:
            logger.info('Permute mode is on!')

    def chb_graph_detail_trigger(self):
        if self.project_instance is not None:
            self.sys_message('Working...')
            self.gs_atomic_graph.re_draw_edges()
            self.sys_message('Ready.')

    def chb_toggle_mesh_trigger(self):
        if self.project_instance is not None:
            self.sys_message('Working...')
            self.gs_atomic_graph.re_draw_mesh_details()
            self.gs_info_graph.re_draw_mesh_details()
            self.sys_message('Ready.')

    def chb_show_level_0_trigger(self, state):
        if self.project_instance is not None:
            if self.gs_anti_graph.graph is not None:
                self.gs_anti_graph.toggle_level_0(state)

    def chb_show_level_1_trigger(self, state):
        if self.project_instance is not None:
            if self.gs_anti_graph.graph is not None:
                self.gs_anti_graph.toggle_level_1(state)

    def chb_raw_image_trigger(self, state):
        if self.project_instance is not None:
            self.sys_message('Working...')
            if state:
                self.gs_overlay_composition.pixmap.show()
            else:
                self.gs_overlay_composition.pixmap.hide()
            self.sys_message('Ready.')

    def chb_black_background_trigger(self, state):
        if state:
            self.gs_overlay_composition.setBackgroundBrush(GUI_settings.brush_black)
        else:
            if GUI_settings.theme == 'dark':
                self.gs_overlay_composition.setBackgroundBrush(GUI_settings.background_brush)
            else:
                self.gs_overlay_composition.setBackgroundBrush(GUI_settings.brush_white)

    def chb_toggle_si_trigger(self, state):
        if self.project_instance is not None and self.project_instance.num_columns > 0:
            self.sys_message('Working...')
            for vertex in self.project_instance.graph.vertices:
                if vertex.atomic_species =='Si':
                    vertex.show_in_overlay = state
            for graphic_item in self.gs_overlay_composition.interactive_overlay_objects:
                graphic_item.set_style()
            self.sys_message('Ready.')

    def chb_toggle_cu_trigger(self, state):
        if self.project_instance is not None and self.project_instance.num_columns > 0:
            self.sys_message('Working...')
            for vertex in self.project_instance.graph.vertices:
                if vertex.atomic_species =='Cu':
                    vertex.show_in_overlay = state
            for graphic_item in self.gs_overlay_composition.interactive_overlay_objects:
                graphic_item.set_style()
            self.sys_message('Ready.')

    def chb_toggle_al_trigger(self, state):
        if self.project_instance is not None and self.project_instance.num_columns > 0:
            self.sys_message('Working...')
            for vertex in self.project_instance.graph.vertices:
                if vertex.atomic_species == 'Al':
                    vertex.show_in_overlay = state
            for graphic_item in self.gs_overlay_composition.interactive_overlay_objects:
                graphic_item.set_style()
            self.sys_message('Ready.')

    def chb_toggle_ag_trigger(self, state):
        if self.project_instance is not None and self.project_instance.num_columns > 0:
            self.sys_message('Working...')
            for vertex in self.project_instance.graph.vertices:
                if vertex.atomic_species =='Ag':
                    vertex.show_in_overlay = state
            for graphic_item in self.gs_overlay_composition.interactive_overlay_objects:
                graphic_item.set_style()
            self.sys_message('Ready.')

    def chb_toggle_mg_trigger(self, state):
        if self.project_instance is not None and self.project_instance.num_columns > 0:
            self.sys_message('Working...')
            for vertex in self.project_instance.graph.vertices:
                if vertex.atomic_species == 'Mg':
                    vertex.show_in_overlay = state
            for graphic_item in self.gs_overlay_composition.interactive_overlay_objects:
                graphic_item.set_style()
            self.sys_message('Ready.')

    def chb_toggle_un_trigger(self, state):
        if self.project_instance is not None and self.project_instance.num_columns > 0:
            self.sys_message('Working...')
            for vertex in self.project_instance.graph.vertices:
                if vertex.atomic_species == 'Un':
                    vertex.show_in_overlay = state
            for graphic_item in self.gs_overlay_composition.interactive_overlay_objects:
                graphic_item.set_style()
            self.sys_message('Ready.')

    def chb_edge_trigger(self, state):
        if self.project_instance is not None and self.project_instance.num_columns > 0:
            self.sys_message('Working...')
            for vertex in self.project_instance.graph.vertices:
                if vertex.is_edge_column:
                    vertex.show_in_overlay = state
            for graphic_item in self.gs_overlay_composition.interactive_overlay_objects:
                graphic_item.set_style()
            self.sys_message('Ready.')

    def chb_particle_trigger(self, state):
        if self.project_instance is not None and self.project_instance.num_columns > 0:
            self.sys_message('Working...')
            for vertex in self.project_instance.graph.vertices:
                if vertex.is_in_precipitate:
                    vertex.show_in_overlay = state
            for graphic_item in self.gs_overlay_composition.interactive_overlay_objects:
                graphic_item.set_style()
            self.sys_message('Ready.')

    def chb_toggle_all_trigger(self, state):
        if self.project_instance is not None and self.project_instance.num_columns > 0:
            self.sys_message('Working...')
            for vertex in self.project_instance.graph.vertices:
                vertex.show_in_overlay = state
            for graphic_item in self.gs_overlay_composition.interactive_overlay_objects:
                graphic_item.set_style()
            self.sys_message('Ready.')

    def chb_matrix_trigger(self, state):
        if self.project_instance is not None and self.project_instance.num_columns > 0:
            self.sys_message('Working...')
            for vertex in self.project_instance.graph.vertices:
                if not vertex.is_in_precipitate:
                    vertex.show_in_overlay = state
            for graphic_item in self.gs_overlay_composition.interactive_overlay_objects:
                graphic_item.set_style()
            self.sys_message('Ready.')

    def chb_legend_trigger(self, state):
        if self.project_instance is not None:
            if state:
                self.gs_overlay_composition.legend.show()
            else:
                self.gs_overlay_composition.legend.hide()

    def chb_scale_bar_trigger(self, state):
        if self.project_instance is not None:
            if state:
                self.gs_raw_image.scale_bar.show()
                self.gs_overlay_composition.scale_bar.show()
            else:
                self.gs_raw_image.scale_bar.hide()
                self.gs_overlay_composition.scale_bar.hide()

    def chb_placeholder_trigger(self):
        pass

    def chb_0_plane_trigger(self, state):
        if self.project_instance is not None and self.project_instance.num_columns > 0:
            self.sys_message('Working...')
            for vertex in self.project_instance.graph.vertices:
                if vertex.zeta == 0:
                    vertex.show_in_overlay = state
            for graphic_item in self.gs_overlay_composition.interactive_overlay_objects:
                graphic_item.set_style()
            self.sys_message('Ready.')

    def chb_1_plane_trigger(self, state):
        if self.project_instance is not None and self.project_instance.num_columns > 0:
            self.sys_message('Working...')
            for vertex in self.project_instance.graph.vertices:
                if vertex.zeta == 1:
                    vertex.show_in_overlay = state
            for graphic_item in self.gs_overlay_composition.interactive_overlay_objects:
                graphic_item.set_style()
            self.sys_message('Ready.')


