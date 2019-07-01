# By Haakon Tvedt @ NTNU
"""Module container for high-level custom GUI-elements"""

from PyQt5 import QtWidgets, QtGui, QtCore
import logging
import graph_op
import GUI_custom_components
import GUI_settings


# ----------
# Custom QGraphicsScene and QGraphicsView classes:
# ----------


class RawImage(QtWidgets.QGraphicsScene):

    def __init__(self, *args, ui_obj=None, background=None):
        """Initialize a custom QtWidgets.QGraphicsScene object for **raw image**."""

        super().__init__(*args)

        self.ui_obj = ui_obj
        self.background_image = background
        if self.background_image is not None:
            self.addPixmap(self.background_image)


class AtomicPositions(QtWidgets.QGraphicsScene):

    def __init__(self, *args, ui_obj=None, background=None):
        """Initialize a custom QtWidgets.QGraphicsScene object for **atomic positions**."""

        super().__init__(*args)

        self.ui_obj = ui_obj
        self.interactive_position_objects = []
        self.background_image = background
        if self.background_image is not None:
            self.addPixmap(self.background_image)

    def re_draw(self):
        """Redraw contents."""
        self.interactive_position_objects = []
        if self.ui_obj.project_instance is not None:
            for vertex in self.ui_obj.project_instance.graph.vertices:
                self.interactive_position_objects.append(GUI_custom_components.InteractivePosColumn(self.ui_obj, vertex.i, 2 * vertex.r))
                self.addItem(self.interactive_position_objects[-1])


class OverlayComposition(QtWidgets.QGraphicsScene):

    def __init__(self, *args, ui_obj=None, background=None):
        """Initialize a custom QtWidgets.QGraphicsScene object for **overlay composition**."""

        super().__init__(*args)

        self.ui_obj = ui_obj
        self.interactive_overlay_objects = []
        self.background_image = background
        if self.background_image is not None:
            self.addPixmap(self.background_image)

    def re_draw(self):
        """Redraw contents."""
        self.interactive_overlay_objects = []
        if self.ui_obj.project_instance is not None:
            for vertex in self.ui_obj.project_instance.graph.vertices:
                self.interactive_overlay_objects.append(GUI_custom_components.InteractiveOverlayColumn(self.ui_obj, vertex.i, vertex.r))
                self.addItem(self.interactive_overlay_objects[-1])


class AtomicGraph(QtWidgets.QGraphicsScene):

    def __init__(self, *args, ui_obj=None, background=None, scale_factor=1):
        """Initialize a custom QtWidgets.QGraphicsScene object for **atomic graphs**."""

        super().__init__(*args)

        self.ui_obj = ui_obj
        self.scale_factor = scale_factor
        self.interactive_vertex_objects = []
        self.edges = []
        self.background_image = background
        if self.background_image is not None:
            self.addPixmap(self.background_image)
        self.re_draw()

    def re_draw(self):
        """Redraw contents."""
        if self.ui_obj.project_instance is not None:
            self.re_draw_edges(self.ui_obj.project_instance.r)
            self.re_draw_vertices()

    def re_draw_vertices(self):
        """Redraws all column elements."""
        self.interactive_vertex_objects = []
        for vertex in self.ui_obj.project_instance.graph.vertices:
            self.interactive_vertex_objects.append(GUI_custom_components.InteractiveGraphColumn(self.ui_obj, vertex.i, vertex.r, self.scale_factor))
            self.addItem(self.interactive_vertex_objects[-1])

    def re_draw_edges(self, r):
        """Redraws all edge elements."""
        self.ui_obj.project_instance.graph.redraw_edges()
        self.edges = []
        for edge in self.ui_obj.project_instance.graph.edges:
            consistent = edge.is_reciprocated
            dislocation = not edge.is_legal_levels
            p1 = edge.vertex_a.real_coor()
            p2 = edge.vertex_b.real_coor()
            self.edges.append(GUI_custom_components.Arrow(p1, p2, r, self.scale_factor, consistent, dislocation))
            self.addItem(self.edges[-1].arrow[0])
            self.addItem(self.edges[-1].arrow[1])


class AtomicSubGraph(QtWidgets.QGraphicsScene):

    def __init__(self, *args, ui_obj=None, background=None, sub_graph=None):
        """Initialize a custom QtWidgets.QGraphicsScene object for **atomic sub-graphs**."""

        super().__init__(*args)

        self.ui_obj = ui_obj
        self.interactive_vertex_objects = []
        self.edges = []
        self.background_image = background
        if self.background_image is not None:
            self.addPixmap(self.background_image)
        self.sub_graph = sub_graph
        if sub_graph is not None:
            self.re_draw()

    def re_draw(self):
        """Redraw contents."""
        if self.ui_obj.project_instance is not None:
            self.re_draw_vertices()
            self.re_draw_edges()

    def re_draw_vertices(self):
        """Redraws all column elements."""
        for vertex in self.sub_graph.vertices:
            self.interactive_vertex_objects.append(GUI_custom_components.InteractiveGraphColumn(self.ui_obj, vertex.i))
            self.addItem(self.interactive_vertex_objects[-1])

    def re_draw_edges(self):
        """Redraws all edge elements."""
        for edge in self.sub_graph.edges:
            consistent = edge.is_reciprocated
            dislocation = not edge.is_legal_levels
            p1 = edge.vertex_a.real_coor()
            p2 = edge.vertex_b.real_coor()
            self.edges.append(GUI_custom_components.Arrow(p1, p2, self.r, self.scale_factor, consistent, dislocation))
            self.addItem(self.edges[-1].arrow[0])
            self.addItem(self.edges[-1].arrow[1])


class ZoomGraphicsView(QtWidgets.QGraphicsView):
    """An adaptation of QtWidgets.QGraphicsView that supports zooming"""

    def __init__(self, parent=None, ui_obj=None, trigger_func=None):
        super(ZoomGraphicsView, self).__init__(parent)
        self.ui_obj = ui_obj
        self.trigger_func = trigger_func

    def wheelEvent(self, event):

        modifier = QtWidgets.QApplication.keyboardModifiers()
        if modifier == QtCore.Qt.ShiftModifier:

            # Zoom Factor
            zoom_in_factor = 1.25
            zoom_out_factor = 1 / zoom_in_factor

            # Set Anchors
            self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
            self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)

            # Save the scene pos
            oldPos = self.mapToScene(event.pos())

            # Zoom
            if event.angleDelta().y() > 0:
                zoomFactor = zoom_in_factor
            else:
                zoomFactor = zoom_out_factor
            self.scale(zoomFactor, zoomFactor)

            # Get the new position
            newPos = self.mapToScene(event.pos())

            # Move scene to old position
            delta = newPos - oldPos
            self.translate(delta.x(), delta.y())

        else:

            super(ZoomGraphicsView, self).wheelEvent(event)

    def keyPressEvent(self, event):
        if self.trigger_func is not None:
            self.trigger_func(event.key())


# ----------
# Custom Main window tools:
# ----------


class TerminalTextEdit(QtWidgets.QPlainTextEdit):

    def __init__(self, *args):
        super().__init__(*args)

        self.setReadOnly(True)
        self.setWordWrapMode(QtGui.QTextOption.NoWrap)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)


class TerminalHandler(logging.Handler):

    def __init__(self):
        super().__init__()

        self.widget = TerminalTextEdit()

        self.setLevel(logging.INFO)
        self.setFormatter(logging.Formatter('%(name)s:%(levelname)s:%(funcName)s:%(message)s'))

    def set_mode(self, debug=False):

        if debug:
            self.setLevel(logging.DEBUG)
            self.setFormatter(logging.Formatter('%(name)s: %(levelname)s: %(funcName)s: %(message)s'))
        else:
            self.setLevel(logging.INFO)
            self.setFormatter(logging.Formatter('%(name)s: %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)


class Terminal(QtWidgets.QWidget):

    def __init__(self, *args, obj=None):
        super().__init__(*args)

        self.ui_obj = obj

        self.handler = TerminalHandler()

        self.btn_save_log = GUI_custom_components.SmallButton('Save log', self, trigger_func=self.ui_obj.btn_save_log_trigger)
        self.btn_clear_log = GUI_custom_components.SmallButton('Clear log', self, trigger_func=self.ui_obj.btn_clear_log_trigger)

        self.terminal_btns_layout = QtWidgets.QHBoxLayout()
        self.terminal_btns_layout.addWidget(self.btn_save_log)
        self.terminal_btns_layout.addWidget(self.btn_clear_log)
        self.terminal_btns_layout.addStretch()

        self.terminal_display_layout = QtWidgets.QVBoxLayout()
        self.terminal_display_layout.addLayout(self.terminal_btns_layout)
        self.terminal_display_layout.addWidget(self.handler.widget)

        self.setLayout(self.terminal_display_layout)


class ControlWindow(QtWidgets.QWidget):

    def __init__(self, *args, obj=None):
        super().__init__(*args)

        self.ui_obj = obj

        # Prob. Vector graphic:
        self.height = 130
        self.width = 200

        self.probGraphicView = QtWidgets.QGraphicsView()

        self.probGraphicView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.probGraphicView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.probGraphicView.setMinimumHeight(self.height)
        self.probGraphicView.setMaximumHeight(self.height)
        self.probGraphicView.setMinimumWidth(self.width)
        self.probGraphicView.setMaximumWidth(self.width)

        self.probGraphicLayout = QtWidgets.QHBoxLayout()
        self.probGraphicLayout.addWidget(self.probGraphicView)
        self.probGraphicLayout.addStretch()

        # Labels
        self.lbl_num_detected_columns = QtWidgets.QLabel('Number of detected columns: ')
        self.lbl_image_width = QtWidgets.QLabel('Image width (pixels): ')
        self.lbl_image_height = QtWidgets.QLabel('Image height (pixels): ')

        self.lbl_starting_index = QtWidgets.QLabel('Default starting index: ')
        self.lbl_std_1 = QtWidgets.QLabel('Standard deviation 1: ')
        self.lbl_std_2 = QtWidgets.QLabel('Standard deviation 2: ')
        self.lbl_std_3 = QtWidgets.QLabel('Standard deviation 3: ')
        self.lbl_std_4 = QtWidgets.QLabel('Standard deviation 4: ')
        self.lbl_std_5 = QtWidgets.QLabel('Standard deviation 5: ')
        self.lbl_std_8 = QtWidgets.QLabel('Standard deviation 8: ')
        self.lbl_cert_threshold = QtWidgets.QLabel('Certainty threshold: ')

        self.lbl_atomic_radii = QtWidgets.QLabel('Approx atomic radii (pixels): ')
        self.lbl_overhead_radii = QtWidgets.QLabel('Overhead (pixels): ')
        self.lbl_detection_threshold = QtWidgets.QLabel('Detection threshold value: ')
        self.lbl_search_matrix_peak = QtWidgets.QLabel('Search matrix peak: ')
        self.lbl_search_size = QtWidgets.QLabel('Search size: ')
        self.lbl_scale = QtWidgets.QLabel('Scale (pm / pixel): ')

        self.lbl_alloy = QtWidgets.QLabel('Alloy: ')

        self.lbl_column_index = QtWidgets.QLabel('Column index: ')
        self.lbl_column_x_pos = QtWidgets.QLabel('x: ')
        self.lbl_column_y_pos = QtWidgets.QLabel('y: ')
        self.lbl_column_peak_gamma = QtWidgets.QLabel('Peak gamma: ')
        self.lbl_column_avg_gamma = QtWidgets.QLabel('Avg gamma: ')
        self.lbl_column_species = QtWidgets.QLabel('Atomic species: ')
        self.lbl_column_level = QtWidgets.QLabel('Level: ')
        self.lbl_confidence = QtWidgets.QLabel('Confidence: ')
        self.lbl_symmetry_confidence = QtWidgets.QLabel('Symmetry confidence: ')
        self.lbl_level_confidence = QtWidgets.QLabel('Level confidence: ')
        self.lbl_prob_vector = QtWidgets.QLabel('Probability vector: ')
        self.lbl_neighbours = QtWidgets.QLabel('Nearest neighbours: ')
        self.lbl_central_variance = QtWidgets.QLabel('Central angle variance: ')
        self.lbl_alpha_max = QtWidgets.QLabel('Alpha max: ')
        self.lbl_alpha_min = QtWidgets.QLabel('Alpha min: ')

        self.lbl_chi = QtWidgets.QLabel('Chi: ')
        self.lbl_avg_variance = QtWidgets.QLabel('Average central variance: ')
        self.lbl_avg_level_confidence = QtWidgets.QLabel('Average level confidence: ')
        self.lbl_avg_symmetry_confidence = QtWidgets.QLabel('Average symmetry confidence: ')
        self.lbl_avg_species_confidence = QtWidgets.QLabel('Average species confidence: ')

        # Checkboxes

        self.chb_precipitate_column = QtWidgets.QCheckBox('Precipitate column')
        self.chb_show = QtWidgets.QCheckBox('Show in overlay')
        self.chb_move = QtWidgets.QCheckBox('Enable move')

        self.chb_graph = QtWidgets.QCheckBox('Show inconsistent connections')

        self.chb_raw_image = QtWidgets.QCheckBox('Raw image')
        self.chb_black_background = QtWidgets.QCheckBox('Black background')
        self.chb_structures = QtWidgets.QCheckBox('Structures')
        self.chb_boarders = QtWidgets.QCheckBox('Boarders')
        self.chb_si_columns = QtWidgets.QCheckBox('Si columns')
        self.chb_si_network = QtWidgets.QCheckBox('Si network')
        self.chb_mg_columns = QtWidgets.QCheckBox('Mg columns')
        self.chb_mg_network = QtWidgets.QCheckBox('Mg network')
        self.chb_al_columns = QtWidgets.QCheckBox('Al columns')
        self.chb_al_network = QtWidgets.QCheckBox('Al network')
        self.chb_cu_columns = QtWidgets.QCheckBox('Cu columns')
        self.chb_cu_network = QtWidgets.QCheckBox('Cu network')
        self.chb_ag_columns = QtWidgets.QCheckBox('Ag columns')
        self.chb_ag_network = QtWidgets.QCheckBox('Ag network')
        self.chb_un_columns = QtWidgets.QCheckBox('Un columns')
        self.chb_columns = QtWidgets.QCheckBox('Columns')
        self.chb_al_mesh = QtWidgets.QCheckBox('Al-mesh')
        self.chb_neighbours = QtWidgets.QCheckBox('Neighbour path')
        self.chb_legend = QtWidgets.QCheckBox('Legend')
        self.chb_scalebar = QtWidgets.QCheckBox('Scalebar')

        overlay_layout_left = QtWidgets.QVBoxLayout()
        overlay_layout_left.addWidget(self.chb_raw_image)
        overlay_layout_left.addWidget(self.chb_structures)
        overlay_layout_left.addWidget(self.chb_si_columns)
        overlay_layout_left.addWidget(self.chb_cu_columns)
        overlay_layout_left.addWidget(self.chb_al_columns)
        overlay_layout_left.addWidget(self.chb_ag_columns)
        overlay_layout_left.addWidget(self.chb_mg_columns)
        overlay_layout_left.addWidget(self.chb_un_columns)
        overlay_layout_left.addWidget(self.chb_al_mesh)
        overlay_layout_left.addWidget(self.chb_legend)

        overlay_layout_right = QtWidgets.QVBoxLayout()
        overlay_layout_right.addWidget(self.chb_black_background)
        overlay_layout_right.addWidget(self.chb_boarders)
        overlay_layout_right.addWidget(self.chb_si_network)
        overlay_layout_right.addWidget(self.chb_cu_network)
        overlay_layout_right.addWidget(self.chb_al_network)
        overlay_layout_right.addWidget(self.chb_ag_network)
        overlay_layout_right.addWidget(self.chb_mg_network)
        overlay_layout_right.addWidget(self.chb_columns)
        overlay_layout_right.addWidget(self.chb_neighbours)
        overlay_layout_right.addWidget(self.chb_scalebar)

        overlay_layout = QtWidgets.QHBoxLayout()
        overlay_layout.addLayout(overlay_layout_left)
        overlay_layout.addLayout(overlay_layout_right)
        overlay_layout.addStretch()

        self.chb_precipitate_column.setChecked(False)
        self.chb_show.setChecked(False)
        self.chb_move.setChecked(False)

        self.chb_graph.setChecked(True)

        self.chb_raw_image.setChecked(True)
        self.chb_black_background.setChecked(False)
        self.chb_structures.setChecked(True)
        self.chb_boarders.setChecked(False)
        self.chb_si_columns.setChecked(True)
        self.chb_si_network.setChecked(False)
        self.chb_cu_columns.setChecked(True)
        self.chb_cu_network.setChecked(False)
        self.chb_al_columns.setChecked(True)
        self.chb_al_network.setChecked(False)
        self.chb_ag_columns.setChecked(True)
        self.chb_ag_network.setChecked(False)
        self.chb_mg_columns.setChecked(True)
        self.chb_mg_network.setChecked(False)
        self.chb_un_columns.setChecked(True)
        self.chb_columns.setChecked(True)
        self.chb_al_mesh.setChecked(True)
        self.chb_neighbours.setChecked(False)
        self.chb_legend.setChecked(True)
        self.chb_scalebar.setChecked(False)

        self.chb_precipitate_column.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_show.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_move.toggled.connect(self.ui_obj.chb_placeholder_trigger)

        self.chb_graph.toggled.connect(self.ui_obj.chb_placeholder_trigger)

        self.chb_raw_image.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_black_background.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_structures.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_boarders.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_si_columns.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_si_network.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_cu_columns.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_cu_network.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_al_columns.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_al_network.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_ag_columns.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_ag_network.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_mg_columns.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_mg_network.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_un_columns.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_columns.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_al_mesh.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_neighbours.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_legend.toggled.connect(self.ui_obj.chb_placeholder_trigger)
        self.chb_scalebar.toggled.connect(self.ui_obj.chb_placeholder_trigger)

        # The Set values buttons
        self.btn_set_threshold_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_threshold_trigger, label=self.lbl_detection_threshold)
        self.btn_set_search_size_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_search_size_trigger, label=self.lbl_search_size)
        self.btn_set_scale_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_scale_trigger, label=self.lbl_scale)
        self.btn_set_alloy_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_alloy_trigger, label=self.lbl_alloy)
        self.btn_set_start_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_start_trigger, label=self.lbl_starting_index)
        self.btn_set_std_1_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_std_1_trigger, label=self.lbl_std_1)
        self.btn_set_std_2_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_std_2_trigger, label=self.lbl_std_2)
        self.btn_set_std_3_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_std_3_trigger, label=self.lbl_std_3)
        self.btn_set_std_4_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_std_4_trigger, label=self.lbl_std_4)
        self.btn_set_std_5_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_std_5_trigger, label=self.lbl_std_5)
        self.btn_set_std_8_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_std_8_trigger, label=self.lbl_std_8)
        self.btn_set_cert_threshold_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_cert_threshold_trigger, label=self.lbl_cert_threshold)
        self.btn_find_column_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_find_column_trigger, label=self.lbl_column_index)
        self.btn_set_species_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_species_trigger, label=self.lbl_column_species)
        self.btn_set_level_layout = GUI_custom_components.SetButtonLayout(obj=self, trigger_func=self.ui_obj.btn_set_level_trigger, label=self.lbl_column_level)

        # Move buttons
        self.btn_cancel_move = GUI_custom_components.SmallButton('Cancel', self, trigger_func=self.ui_obj.btn_cancel_move_trigger)
        self.btn_cancel_move.setDisabled(True)
        self.btn_set_move = GUI_custom_components.SmallButton('Accept', self, trigger_func=self.ui_obj.btn_set_position_trigger)
        self.btn_set_move.setDisabled(True)

        # other buttons
        self.btn_show_stats = GUI_custom_components.SmallButton('Stats', self, trigger_func=self.ui_obj.btn_show_stats_trigger)
        self.btn_show_source = GUI_custom_components.SmallButton('Source', self, trigger_func=self.ui_obj.btn_view_image_title_trigger)
        self.btn_export = GUI_custom_components.SmallButton('Export', self, trigger_func=self.ui_obj.btn_export_overlay_image_trigger)
        self.btn_start_alg_1 = GUI_custom_components.SmallButton('Start', self, trigger_func=self.ui_obj.btn_continue_detection_trigger)
        self.btn_reset_alg_1 = GUI_custom_components.SmallButton('Reset', self, trigger_func=self.ui_obj.btn_restart_detection_trigger)
        self.btn_start_alg_2 = GUI_custom_components.SmallButton('Start', self, trigger_func=self.ui_obj.btn_continue_analysis_trigger)
        self.btn_reset_alg_2 = GUI_custom_components.SmallButton('Reset', self, trigger_func=self.ui_obj.btn_restart_analysis_trigger)
        self.btn_invert_lvl_alg_2 = GUI_custom_components.SmallButton('Invert lvl', self, trigger_func=self.ui_obj.btn_invert_levels_trigger)
        self.btn_delete = GUI_custom_components.SmallButton('Delete', self, trigger_func=self.ui_obj.btn_delete_trigger)
        self.btn_sub = GUI_custom_components.SmallButton('Sub-graph', self, trigger_func=self.ui_obj.btn_gen_sub_graph)
        self.btn_deselect = GUI_custom_components.SmallButton('Deselect', self, trigger_func=self.ui_obj.btn_deselect_trigger)
        self.btn_new = GUI_custom_components.SmallButton('New', self, trigger_func=self.ui_obj.btn_new_column_trigger)
        self.btn_set_style = GUI_custom_components.MediumButton('Set overlay style', self, trigger_func=self.ui_obj.btn_set_style_trigger)
        self.btn_set_indices = GUI_custom_components.MediumButton('Set neighbours', self, trigger_func=self.ui_obj.btn_set_indices_trigger)
        self.btn_set_indices_2 = GUI_custom_components.MediumButton('Set neighbours manually', self, trigger_func=self.ui_obj.btn_set_indices_2_trigger)
        self.btn_set_perturb_mode = GUI_custom_components.MediumButton('Perturb mode', self, trigger_func=self.ui_obj.btn_set_perturb_mode_trigger)
        self.btn_plot_variance = GUI_custom_components.MediumButton('Plot variance', self, trigger_func=self.ui_obj.btn_plot_variance_trigger)
        self.btn_plot_angles = GUI_custom_components.MediumButton('Plot angles', self, trigger_func=self.ui_obj.btn_plot_angles_trigger)

        # Button layouts
        btn_move_control_layout = QtWidgets.QHBoxLayout()
        btn_move_control_layout.addWidget(self.chb_move)
        btn_move_control_layout.addWidget(self.btn_cancel_move)
        btn_move_control_layout.addWidget(self.btn_set_move)
        btn_move_control_layout.addStretch()

        btn_debug_btns_layout = QtWidgets.QHBoxLayout()
        btn_debug_btns_layout.addWidget(self.btn_set_indices)
        btn_debug_btns_layout.addWidget(self.btn_set_indices_2)
        btn_debug_btns_layout.addStretch()

        btn_image_btns_layout = QtWidgets.QHBoxLayout()
        btn_image_btns_layout.addWidget(self.btn_show_stats)
        btn_image_btns_layout.addWidget(self.btn_show_source)
        btn_image_btns_layout.addWidget(self.btn_export)
        btn_image_btns_layout.addStretch()

        btn_alg_1_btns_layout = QtWidgets.QHBoxLayout()
        btn_alg_1_btns_layout.addWidget(self.btn_start_alg_1)
        btn_alg_1_btns_layout.addWidget(self.btn_reset_alg_1)
        btn_alg_1_btns_layout.addStretch()

        btn_alg_2_btns_layout = QtWidgets.QHBoxLayout()
        btn_alg_2_btns_layout.addWidget(self.btn_start_alg_2)
        btn_alg_2_btns_layout.addWidget(self.btn_reset_alg_2)
        btn_alg_2_btns_layout.addWidget(self.btn_invert_lvl_alg_2)
        btn_alg_2_btns_layout.addStretch()

        btn_column_btns_layout = QtWidgets.QHBoxLayout()
        btn_column_btns_layout.addWidget(self.btn_new)
        btn_column_btns_layout.addWidget(self.btn_deselect)
        btn_column_btns_layout.addWidget(self.btn_delete)
        btn_column_btns_layout.addWidget(self.btn_sub)
        btn_column_btns_layout.addStretch()

        btn_overlay_btns_layout = QtWidgets.QHBoxLayout()
        btn_overlay_btns_layout.addWidget(self.btn_set_style)
        btn_overlay_btns_layout.addStretch()

        btn_graph_btns_layout = QtWidgets.QHBoxLayout()
        btn_graph_btns_layout.addWidget(self.btn_set_perturb_mode)
        btn_graph_btns_layout.addWidget(self.btn_plot_variance)
        btn_graph_btns_layout.addWidget(self.btn_plot_angles)
        btn_graph_btns_layout.addStretch()

        # Group boxes
        self.image_box_layout = QtWidgets.QVBoxLayout()
        self.image_box_layout.addLayout(btn_image_btns_layout)
        self.image_box_layout.addWidget(self.lbl_image_width)
        self.image_box_layout.addWidget(self.lbl_image_height)
        self.image_box_layout.addWidget(self.lbl_num_detected_columns)
        self.image_box = GUI_custom_components.GroupBox('Image')
        self.image_box.setLayout(self.image_box_layout)

        self.debug_box_layout = QtWidgets.QVBoxLayout()
        self.debug_box_layout.addLayout(btn_debug_btns_layout)
        self.debug_box_layout.addLayout(self.btn_set_start_layout)
        self.debug_box_layout.addLayout(self.btn_set_std_1_layout)
        self.debug_box_layout.addLayout(self.btn_set_std_2_layout)
        self.debug_box_layout.addLayout(self.btn_set_std_3_layout)
        self.debug_box_layout.addLayout(self.btn_set_std_4_layout)
        self.debug_box_layout.addLayout(self.btn_set_std_5_layout)
        self.debug_box_layout.addLayout(self.btn_set_std_8_layout)
        self.debug_box_layout.addLayout(self.btn_set_cert_threshold_layout)
        self.debug_box = GUI_custom_components.GroupBox('Advanced debug mode')
        self.debug_box.setLayout(self.debug_box_layout)

        self.alg_1_box_layout = QtWidgets.QVBoxLayout()
        self.alg_1_box_layout.addLayout(btn_alg_1_btns_layout)
        self.alg_1_box_layout.addWidget(self.lbl_search_matrix_peak)
        self.alg_1_box_layout.addWidget(self.lbl_atomic_radii)
        self.alg_1_box_layout.addWidget(self.lbl_overhead_radii)
        self.alg_1_box_layout.addLayout(self.btn_set_scale_layout)
        self.alg_1_box_layout.addLayout(self.btn_set_threshold_layout)
        self.alg_1_box_layout.addLayout(self.btn_set_search_size_layout)
        self.alg_1_box = GUI_custom_components.GroupBox('Column detection')
        self.alg_1_box.setLayout(self.alg_1_box_layout)

        self.alg_2_box_layout = QtWidgets.QVBoxLayout()
        self.alg_2_box_layout.addLayout(btn_alg_2_btns_layout)
        self.alg_2_box_layout.addLayout(self.btn_set_scale_layout)
        self.alg_2_box_layout.addLayout(self.btn_set_alloy_layout)
        self.alg_2_box = GUI_custom_components.GroupBox('Column characterization')
        self.alg_2_box.setLayout(self.alg_2_box_layout)

        self.column_box_layout = QtWidgets.QVBoxLayout()
        self.column_box_layout.addLayout(btn_column_btns_layout)
        self.column_box_layout.addLayout(self.btn_find_column_layout)
        self.column_box_layout.addLayout(self.btn_set_species_layout)
        self.column_box_layout.addLayout(self.btn_set_level_layout)
        self.column_box_layout.addWidget(self.lbl_column_x_pos)
        self.column_box_layout.addWidget(self.lbl_column_y_pos)
        self.column_box_layout.addWidget(self.lbl_column_peak_gamma)
        self.column_box_layout.addWidget(self.lbl_column_avg_gamma)
        self.column_box_layout.addWidget(self.lbl_confidence)
        self.column_box_layout.addWidget(self.lbl_symmetry_confidence)
        self.column_box_layout.addWidget(self.lbl_level_confidence)
        self.column_box_layout.addWidget(self.lbl_prob_vector)
        self.column_box_layout.addLayout(self.probGraphicLayout)
        self.column_box_layout.addWidget(self.chb_precipitate_column)
        self.column_box_layout.addWidget(self.chb_show)
        self.column_box_layout.addLayout(btn_move_control_layout)
        self.column_box_layout.addWidget(self.lbl_neighbours)
        self.column_box_layout.addWidget(self.lbl_central_variance)
        self.column_box = GUI_custom_components.GroupBox('Selected column')
        self.column_box.setLayout(self.column_box_layout)

        self.graph_box_layout = QtWidgets.QVBoxLayout()
        self.graph_box_layout.addLayout(btn_graph_btns_layout)
        self.graph_box_layout.addWidget(self.chb_graph)
        self.graph_box_layout.addWidget(self.lbl_chi)
        self.graph_box_layout.addWidget(self.lbl_avg_species_confidence)
        self.graph_box_layout.addWidget(self.lbl_avg_symmetry_confidence)
        self.graph_box_layout.addWidget(self.lbl_avg_level_confidence)
        self.graph_box_layout.addWidget(self.lbl_avg_variance)
        self.graph_box = GUI_custom_components.GroupBox('Atomic graph')
        self.graph_box.setLayout(self.graph_box_layout)

        self.overlay_box_layout = QtWidgets.QVBoxLayout()
        self.overlay_box_layout.addLayout(btn_overlay_btns_layout)
        self.overlay_box_layout.addLayout(overlay_layout)
        self.overlay_box = GUI_custom_components.GroupBox('Overlay settings')
        self.overlay_box.setLayout(self.overlay_box_layout)

        # Top level layout
        self.info_display_layout = QtWidgets.QVBoxLayout()
        self.info_display_layout.addWidget(self.image_box)
        self.info_display_layout.addWidget(self.image_box.shadow_box)
        self.info_display_layout.addWidget(self.debug_box)
        self.info_display_layout.addWidget(self.debug_box.shadow_box)
        self.info_display_layout.addWidget(self.alg_1_box)
        self.info_display_layout.addWidget(self.alg_1_box.shadow_box)
        self.info_display_layout.addWidget(self.alg_2_box)
        self.info_display_layout.addWidget(self.alg_2_box.shadow_box)
        self.info_display_layout.addWidget(self.column_box)
        self.info_display_layout.addWidget(self.column_box.shadow_box)
        self.info_display_layout.addWidget(self.graph_box)
        self.info_display_layout.addWidget(self.graph_box.shadow_box)
        self.info_display_layout.addWidget(self.overlay_box)
        self.info_display_layout.addWidget(self.overlay_box.shadow_box)
        self.info_display_layout.addStretch()

        self.setLayout(self.info_display_layout)

    def mode_move(self, on):

        if on:

            if self.chb_move.isChecked():
                if self.ui_obj.project_loaded and not self.ui_obj.selected_column == -1:

                    self.ui_obj.pos_objects[self.ui_obj.selected_column].setFlag(QtWidgets.QGraphicsItem.ItemIsMovable)

                    self.btn_show_stats.setDisabled(True)
                    self.btn_show_source.setDisabled(True)
                    self.btn_export.setDisabled(True)

                    self.btn_start_alg_1.setDisabled(True)
                    self.btn_reset_alg_1.setDisabled(True)

                    self.btn_start_alg_2.setDisabled(True)
                    self.btn_reset_alg_2.setDisabled(True)
                    self.btn_invert_lvl_alg_2.setDisabled(True)

                    self.btn_new.setDisabled(True)
                    self.btn_deselect.setDisabled(True)
                    self.btn_delete.setDisabled(True)
                    self.btn_set_species.setDisabled(True)
                    self.btn_set_level.setDisabled(True)
                    self.btn_find_column.setDisabled(True)
                    self.btn_set_move.setDisabled(False)
                    self.btn_cancel_move.setDisabled(False)
                    self.chb_show.setDisabled(True)
                    self.chb_precipitate_column.setDisabled(True)

                    self.btn_set_threshold.setDisabled(True)
                    self.btn_set_search_size.setDisabled(True)
                    self.btn_set_scale.setDisabled(True)

                    self.btn_set_alloy.setDisabled(True)
                    self.btn_set_std_1.setDisabled(True)
                    self.btn_set_std_2.setDisabled(True)
                    self.btn_set_std_3.setDisabled(True)
                    self.btn_set_std_4.setDisabled(True)
                    self.btn_set_std_5.setDisabled(True)
                    self.btn_set_std_8.setDisabled(True)
                    self.btn_set_cert_threshold.setDisabled(True)

                    self.btn_set_style.setDisabled(True)

                    self.chb_raw_image.setDisabled(True)
                    self.chb_black_background.setDisabled(True)
                    self.chb_structures.setDisabled(True)
                    self.chb_boarders.setDisabled(True)
                    self.chb_si_columns.setDisabled(True)
                    self.chb_si_network.setDisabled(True)
                    self.chb_cu_columns.setDisabled(True)
                    self.chb_cu_network.setDisabled(True)
                    self.chb_al_columns.setDisabled(True)
                    self.chb_al_network.setDisabled(True)
                    self.chb_ag_columns.setDisabled(True)
                    self.chb_ag_network.setDisabled(True)
                    self.chb_mg_columns.setDisabled(True)
                    self.chb_mg_network.setDisabled(True)
                    self.chb_un_columns.setDisabled(True)
                    self.chb_columns.setDisabled(True)
                    self.chb_al_mesh.setDisabled(True)
                    self.chb_neighbours.setDisabled(True)
                    self.chb_legend.setDisabled(True)
                    self.chb_scalebar.setDisabled(True)

                    for i in range(0, self.ui_obj.project_instance.num_columns):
                        self.ui_obj.pos_objects[i].setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
                        self.ui_obj.overlay_objects[i].setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)

                    self.ui_obj.statusBar().showMessage('Ready to move...')
            else:
                if self.ui_obj.project_loaded and not self.ui_obj.selected_column == -1:
                    self.ui_obj.cancel_move_trigger()

        else:

            self.chb_move.blockSignals(True)
            self.chb_move.setChecked(False)
            self.chb_move.blockSignals(False)

            self.btn_show_stats.setDisabled(False)
            self.btn_show_source.setDisabled(False)
            self.btn_export.setDisabled(False)

            self.btn_start_alg_1.setDisabled(False)
            self.btn_reset_alg_1.setDisabled(False)

            self.btn_start_alg_2.setDisabled(False)
            self.btn_reset_alg_2.setDisabled(False)
            self.btn_invert_lvl_alg_2.setDisabled(False)

            self.btn_set_threshold.setDisabled(False)
            self.btn_set_search_size.setDisabled(False)
            self.btn_set_scale.setDisabled(False)

            self.btn_set_alloy.setDisabled(False)
            self.btn_set_std_1.setDisabled(False)
            self.btn_set_std_2.setDisabled(False)
            self.btn_set_std_3.setDisabled(False)
            self.btn_set_std_4.setDisabled(False)
            self.btn_set_std_5.setDisabled(False)
            self.btn_set_std_8.setDisabled(False)
            self.btn_set_cert_threshold.setDisabled(False)

            self.btn_new.setDisabled(False)
            self.btn_deselect.setDisabled(False)
            self.btn_delete.setDisabled(False)
            self.btn_set_species.setDisabled(False)
            self.btn_set_level.setDisabled(False)
            self.btn_find_column.setDisabled(False)
            self.chb_show.setDisabled(False)
            self.chb_precipitate_column.setDisabled(False)

            self.btn_set_style.setDisabled(False)

            self.chb_raw_image.setDisabled(False)
            self.chb_black_background.setDisabled(False)
            self.chb_structures.setDisabled(False)
            self.chb_boarders.setDisabled(False)
            self.chb_si_columns.setDisabled(False)
            self.chb_si_network.setDisabled(False)
            self.chb_cu_columns.setDisabled(False)
            self.chb_cu_network.setDisabled(False)
            self.chb_al_columns.setDisabled(False)
            self.chb_al_network.setDisabled(False)
            self.chb_ag_columns.setDisabled(False)
            self.chb_ag_network.setDisabled(False)
            self.chb_mg_columns.setDisabled(False)
            self.chb_mg_network.setDisabled(False)
            self.chb_un_columns.setDisabled(False)
            self.chb_columns.setDisabled(False)
            self.chb_al_mesh.setDisabled(False)
            self.chb_neighbours.setDisabled(False)
            self.chb_legend.setDisabled(False)
            self.chb_scalebar.setDisabled(False)

            self.btn_cancel_move.setChecked(False)
            self.btn_set_move.setChecked(False)
            self.btn_cancel_move.setDisabled(True)
            self.btn_set_move.setDisabled(True)

            self.ui_obj.update_central_widget()

    def mode_debug(self, on):

        pass

    def draw_histogram(self):

        box_width = 15
        box_seperation = 10
        box_displacement = 25

        if not self.ui_obj.selected_column == -1:

            si_box_height = int(100 * self.ui_obj.project_instance.graph.vertices[self.ui_obj.selected_column].prob_vector[0])
            cu_box_height = int(100 * self.ui_obj.project_instance.graph.vertices[self.ui_obj.selected_column].prob_vector[1])
            zn_box_height = int(100 * self.ui_obj.project_instance.graph.vertices[self.ui_obj.selected_column].prob_vector[2])
            al_box_height = int(100 * self.ui_obj.project_instance.graph.vertices[self.ui_obj.selected_column].prob_vector[3])
            ag_box_height = int(100 * self.ui_obj.project_instance.graph.vertices[self.ui_obj.selected_column].prob_vector[4])
            mg_box_height = int(100 * self.ui_obj.project_instance.graph.vertices[self.ui_obj.selected_column].prob_vector[5])
            un_box_height = int(100 * self.ui_obj.project_instance.graph.vertices[self.ui_obj.selected_column].prob_vector[6])

        else:

            si_box_height = 0
            cu_box_height = 0
            zn_box_height = 0
            al_box_height = 0
            ag_box_height = 0
            mg_box_height = 0
            un_box_height = 0

        box = QtWidgets.QGraphicsRectItem(0, -10, self.width - 10, self.height - 10)
        box.setPen(GUI_settings.pen_boarder)
        box.hide()

        x_box = box_seperation + 0 * box_displacement
        box_height = si_box_height
        si_box = QtWidgets.QGraphicsRectItem(0, 0, box_width, box_height)
        si_box.setBrush(GUI_settings.brush_si)
        si_box.setX(x_box)
        si_box.setY(100 - box_height)
        si_text = QtWidgets.QGraphicsSimpleTextItem()
        si_text.setText('Si')
        si_text.setFont(GUI_settings.font_tiny)
        si_text.setX(x_box + 3)
        si_text.setY(100 + 4)
        si_number = QtWidgets.QGraphicsSimpleTextItem()
        si_number.setText(str(box_height / 100))
        si_number.setFont(GUI_settings.font_tiny)
        si_number.setX(x_box - 1)
        si_number.setY(100 - box_height - 10)

        x_box = box_seperation + 1 * box_displacement
        box_height = cu_box_height
        cu_box = QtWidgets.QGraphicsRectItem(0, 0, box_width, box_height)
        cu_box.setBrush(GUI_settings.brush_cu)
        cu_box.setX(x_box)
        cu_box.setY(100 - box_height)
        cu_text = QtWidgets.QGraphicsSimpleTextItem()
        cu_text.setText('Cu')
        cu_text.setFont(GUI_settings.font_tiny)
        cu_text.setX(x_box + 2)
        cu_text.setY(100 + 4)
        cu_number = QtWidgets.QGraphicsSimpleTextItem()
        cu_number.setText(str(box_height / 100))
        cu_number.setFont(GUI_settings.font_tiny)
        cu_number.setX(x_box - 1)
        cu_number.setY(100 - box_height - 10)

        x_box = box_seperation + 2 * box_displacement
        box_height = zn_box_height
        zn_box = QtWidgets.QGraphicsRectItem(0, 0, box_width, box_height)
        zn_box.setBrush(GUI_settings.brush_zn)
        zn_box.setX(x_box)
        zn_box.setY(100 - box_height)
        zn_text = QtWidgets.QGraphicsSimpleTextItem()
        zn_text.setText('Zn')
        zn_text.setFont(GUI_settings.font_tiny)
        zn_text.setX(x_box + 2)
        zn_text.setY(100 + 4)
        zn_number = QtWidgets.QGraphicsSimpleTextItem()
        zn_number.setText(str(box_height / 100))
        zn_number.setFont(GUI_settings.font_tiny)
        zn_number.setX(x_box - 1)
        zn_number.setY(100 - box_height - 10)

        x_box = box_seperation + 3 * box_displacement
        box_height = al_box_height
        al_box = QtWidgets.QGraphicsRectItem(0, 0, box_width, box_height)
        al_box.setBrush(GUI_settings.brush_al)
        al_box.setX(x_box)
        al_box.setY(100 - box_height)
        al_text = QtWidgets.QGraphicsSimpleTextItem()
        al_text.setText('Al')
        al_text.setFont(GUI_settings.font_tiny)
        al_text.setX(x_box + 2)
        al_text.setY(100 + 4)
        al_number = QtWidgets.QGraphicsSimpleTextItem()
        al_number.setText(str(box_height / 100))
        al_number.setFont(GUI_settings.font_tiny)
        al_number.setX(x_box - 1)
        al_number.setY(100 - box_height - 10)

        x_box = box_seperation + 4 * box_displacement
        box_height = ag_box_height
        ag_box = QtWidgets.QGraphicsRectItem(0, 0, box_width, box_height)
        ag_box.setBrush(GUI_settings.brush_ag)
        ag_box.setX(x_box)
        ag_box.setY(100 - box_height)
        ag_text = QtWidgets.QGraphicsSimpleTextItem()
        ag_text.setText('Ag')
        ag_text.setFont(GUI_settings.font_tiny)
        ag_text.setX(x_box + 2)
        ag_text.setY(100 + 4)
        ag_number = QtWidgets.QGraphicsSimpleTextItem()
        ag_number.setText(str(box_height / 100))
        ag_number.setFont(GUI_settings.font_tiny)
        ag_number.setX(x_box - 1)
        ag_number.setY(100 - box_height - 10)

        x_box = box_seperation + 5 * box_displacement
        box_height = mg_box_height
        mg_box = QtWidgets.QGraphicsRectItem(0, 0, box_width, box_height)
        mg_box.setBrush(GUI_settings.brush_mg)
        mg_box.setX(x_box)
        mg_box.setY(100 - box_height)
        mg_text = QtWidgets.QGraphicsSimpleTextItem()
        mg_text.setText('Mg')
        mg_text.setFont(GUI_settings.font_tiny)
        mg_text.setX(x_box + 2)
        mg_text.setY(100 + 4)
        mg_number = QtWidgets.QGraphicsSimpleTextItem()
        mg_number.setText(str(box_height / 100))
        mg_number.setFont(GUI_settings.font_tiny)
        mg_number.setX(x_box - 1)
        mg_number.setY(100 - box_height - 10)

        x_box = box_seperation + 6 * box_displacement
        box_height = un_box_height
        un_box = QtWidgets.QGraphicsRectItem(0, 0, box_width, box_height)
        un_box.setBrush(GUI_settings.brush_un)
        un_box.setX(x_box)
        un_box.setY(100 - box_height)
        un_text = QtWidgets.QGraphicsSimpleTextItem()
        un_text.setText('Un')
        un_text.setFont(GUI_settings.font_tiny)
        un_text.setX(x_box + 2)
        un_text.setY(100 + 4)
        un_number = QtWidgets.QGraphicsSimpleTextItem()
        un_number.setText(str(box_height / 100))
        un_number.setFont(GUI_settings.font_tiny)
        un_number.setX(x_box - 1)
        un_number.setY(100 - box_height - 10)

        probGraphicScene = QtWidgets.QGraphicsScene()

        probGraphicScene.addItem(box)

        probGraphicScene.addItem(si_box)
        probGraphicScene.addItem(cu_box)
        probGraphicScene.addItem(zn_box)
        probGraphicScene.addItem(al_box)
        probGraphicScene.addItem(ag_box)
        probGraphicScene.addItem(mg_box)
        probGraphicScene.addItem(un_box)

        probGraphicScene.addItem(si_text)
        probGraphicScene.addItem(cu_text)
        probGraphicScene.addItem(zn_text)
        probGraphicScene.addItem(al_text)
        probGraphicScene.addItem(ag_text)
        probGraphicScene.addItem(mg_text)
        probGraphicScene.addItem(un_text)

        probGraphicScene.addItem(si_number)
        probGraphicScene.addItem(cu_number)
        probGraphicScene.addItem(zn_number)
        probGraphicScene.addItem(al_number)
        probGraphicScene.addItem(ag_number)
        probGraphicScene.addItem(mg_number)
        probGraphicScene.addItem(un_number)

        self.probGraphicView.setScene(probGraphicScene)

    def select_column(self):

        i = self.ui_obj.selected_column

        if i == -1:
            self.deselect_column()
        else:

            vertex = self.ui_obj.project_instance.graph.vertices[i]

            self.lbl_column_index.setText('Column index: {}'.format(i))
            self.lbl_column_x_pos.setText('x: {}'.format(vertex.im_coor_x))
            self.lbl_column_y_pos.setText('y: {}'.format(vertex.im_coor_y))
            self.lbl_column_peak_gamma.setText('Peak gamma: {}'.format(vertex.peak_gamma))
            self.lbl_column_avg_gamma.setText('Avg gamma: {}'.format(vertex.avg_gamma))
            self.lbl_column_species.setText('Atomic species: {}'.format(vertex.species()))
            self.lbl_column_level.setText('Level: {}'.format(vertex.level))
            self.lbl_confidence.setText('Confidence: {}'.format(vertex.confidence))
            self.lbl_symmetry_confidence.setText('Symmetry confidence: {}'.format(vertex.symmetry_confidence))
            self.lbl_level_confidence.setText('Level confidence: {}'.format(vertex.level_confidence))
            self.lbl_prob_vector.setText('Probability vector: {}'.format(vertex.prob_vector))
            self.lbl_neighbours.setText('Nearest neighbours: {}'.format(vertex.neighbour_indices))
            *_, variance = self.ui_obj.project_instance.graph.calc_central_angle_variance(self.ui_obj.selected_column)
            alpha_max, alpha_min, *_ = graph_op.base_angle_score(self.ui_obj.project_instance.graph, i, apply=False)
            self.lbl_central_variance.setText('Central angle variance: {}'.format(variance))
            self.lbl_alpha_max = QtWidgets.QLabel('Alpha max: {}'.format(alpha_max))
            self.lbl_alpha_min = QtWidgets.QLabel('Alpha min: {}'.format(alpha_min))

            self.btn_new.setDisabled(False)
            self.btn_deselect.setDisabled(False)
            self.btn_delete.setDisabled(False)
            self.btn_set_species_layout.itemAt(0).widget().setDisabled(False)
            self.btn_set_level_layout.itemAt(0).widget().setDisabled(False)
            self.btn_find_column_layout.itemAt(0).widget().setDisabled(False)
            self.chb_precipitate_column.setDisabled(False)
            self.chb_show.setDisabled(False)
            self.chb_move.setDisabled(False)

            self.chb_show.blockSignals(True)
            self.chb_show.setChecked(self.ui_obj.project_instance.graph.vertices[i].show_in_overlay)
            self.chb_show.blockSignals(False)
            self.chb_precipitate_column.blockSignals(True)
            self.chb_precipitate_column.setChecked(self.ui_obj.project_instance.graph.vertices[i].is_in_precipitate)
            self.chb_precipitate_column.blockSignals(False)
            self.chb_move.blockSignals(True)
            self.chb_move.setChecked(False)
            self.chb_move.blockSignals(False)

            self.btn_set_move.setDisabled(True)
            self.btn_cancel_move.setDisabled(True)

            self.draw_histogram()

    def deselect_column(self):

        self.draw_histogram()

        self.lbl_column_index.setText('Column index: ')
        self.lbl_column_x_pos.setText('x: ')
        self.lbl_column_y_pos.setText('y: ')
        self.lbl_column_peak_gamma.setText('Peak gamma: ')
        self.lbl_column_avg_gamma.setText('Avg gamma: ')
        self.lbl_column_species.setText('Atomic species: ')
        self.lbl_column_level.setText('Level: ')
        self.lbl_confidence.setText('Confidence: ')
        self.lbl_symmetry_confidence.setText('Symmetry confidence: ')
        self.lbl_level_confidence.setText('Level confidence: ')
        self.lbl_prob_vector.setText('Probability vector: ')
        self.lbl_neighbours.setText('Nearest neighbours: ')
        self.lbl_central_variance.setText('Central angle variance: ')
        self.lbl_alpha_max = QtWidgets.QLabel('Alpha max: ')
        self.lbl_alpha_min = QtWidgets.QLabel('Alpha min: ')

    def update_display(self):

        if self.ui_obj.project_instance is not None:

            self.lbl_num_detected_columns.setText('Number of detected columns: {}'.format(self.ui_obj.project_instance.num_columns))
            self.lbl_image_width.setText('Image width (pixels): {}'.format(self.ui_obj.project_instance.im_width))
            self.lbl_image_height.setText('Image height (pixels): {}'.format(self.ui_obj.project_instance.im_height))

            self.lbl_starting_index.setText('Default starting index: {}'.format(self.ui_obj.project_instance.starting_index))
            self.lbl_std_1.setText('Standard deviation 1: ')
            self.lbl_std_2.setText('Standard deviation 2: ')
            self.lbl_std_3.setText('Standard deviation 3: ')
            self.lbl_std_4.setText('Standard deviation 4: ')
            self.lbl_std_5.setText('Standard deviation 5: ')
            self.lbl_std_8.setText('Standard deviation 8: ')
            self.lbl_cert_threshold.setText('Certainty threshold: {}'.format(self.ui_obj.project_instance.certainty_threshold))

            self.lbl_atomic_radii.setText('Approx atomic radii (pixels): {}'.format(self.ui_obj.project_instance.r))
            self.lbl_overhead_radii.setText('Overhead (pixels): {}'.format(self.ui_obj.project_instance.overhead))
            self.lbl_detection_threshold.setText('Detection threshold value: {}'.format(self.ui_obj.project_instance.threshold))
            self.lbl_search_matrix_peak.setText('Search matrix peak: {}'.format(self.ui_obj.project_instance.search_mat.max()))
            self.lbl_search_size.setText('Search size: {}'.format(self.ui_obj.project_instance.search_size))
            self.lbl_scale.setText('Scale (pm / pixel): {}'.format(self.ui_obj.project_instance.scale))

            self.lbl_alloy.setText(self.ui_obj.project_instance.alloy_string())

            if self.ui_obj.selected_column == -1:
                self.deselect_column()
            else:
                self.select_column()

            self.lbl_chi.setText('Chi: {}'.format(self.ui_obj.project_instance.graph.chi))
            self.lbl_avg_species_confidence.setText('Average species confidence: {}'.format(self.ui_obj.project_instance.graph.avg_species_confidence))
            self.lbl_avg_symmetry_confidence.setText('Average symmetry confidence: {}'.format(self.ui_obj.project_instance.graph.avg_symmetry_confidence))
            self.lbl_avg_level_confidence.setText('Average level confidence: {}'.format(self.ui_obj.project_instance.graph.avg_level_confidence))
            self.lbl_avg_variance.setText('Average angle variance: {}'.format(self.ui_obj.project_instance.graph.avg_central_variance))

        else:

            self.empty_display()

    def empty_display(self):

        self.deselect_column()

        self.lbl_num_detected_columns.setText('Number of detected columns: ')
        self.lbl_image_width.setText('Image width (pixels): ')
        self.lbl_image_height.setText('Image height (pixels): ')

        self.lbl_atomic_radii.setText('Approx atomic radii (pixels): ')
        self.lbl_overhead_radii.setText('Overhead (pixels): ')
        self.lbl_detection_threshold.setText('Detection threshold value: ')
        self.lbl_search_matrix_peak.setText('Search matrix peak: ')
        self.lbl_search_size.setText('Search size: ')
        self.lbl_scale.setText('Scale (pm / pixel): ')

        self.lbl_alloy.setText('Alloy: ')

        self.lbl_starting_index.setText('Default starting index: ')
        self.lbl_std_1.setText('Standard deviation 1: ')
        self.lbl_std_2.setText('Standard deviation 2: ')
        self.lbl_std_3.setText('Standard deviation 3: ')
        self.lbl_std_4.setText('Standard deviation 4: ')
        self.lbl_std_5.setText('Standard deviation 5: ')
        self.lbl_std_8.setText('Standard deviation 8: ')
        self.lbl_cert_threshold.setText('Certainty threshold: ')

        self.lbl_chi.setText('Chi: ')
        self.lbl_avg_species_confidence.setText('Average species confidence: ')
        self.lbl_avg_symmetry_confidence.setText('Average symmetry confidence: ')
        self.lbl_avg_level_confidence.setText('Average level confidence: ')
        self.lbl_avg_variance.setText('Average angle variance: ')

        self.chb_raw_image.setChecked(True)
        self.chb_structures.setChecked(True)
        self.chb_si_network.setChecked(False)
        self.chb_mg_network.setChecked(False)
        self.chb_al_network.setChecked(False)
        self.chb_boarders.setChecked(False)
        self.chb_columns.setChecked(True)
        self.chb_legend.setChecked(True)
        self.chb_scalebar.setChecked(False)


class MenuBar:

    def __init__(self, bar_obj, ui_obj):

        self.bar_obj = bar_obj
        self.ui_obj = ui_obj

        # Main headers
        file = self.bar_obj.addMenu('File')
        edit = self.bar_obj.addMenu('Edit')
        view = self.bar_obj.addMenu('View')
        process = self.bar_obj.addMenu('Process')
        column_detection = process.addMenu('Column detection')
        column_analysis = process.addMenu('Column analysis')
        export = self.bar_obj.addMenu('Export')
        image = export.addMenu('Image')
        debug = self.bar_obj.addMenu('Debug')
        help = self.bar_obj.addMenu('Help')

        # Create actions for menus
        # - file
        new_action = QtWidgets.QAction('New', self.ui_obj)
        open_action = QtWidgets.QAction('Open', self.ui_obj)
        save_action = QtWidgets.QAction('Save', self.ui_obj)
        close_action = QtWidgets.QAction('Close', self.ui_obj)
        exit_action = QtWidgets.QAction('Exit', self.ui_obj)
        # - edit
        # - view
        view_image_title_action = QtWidgets.QAction('View path of original image', self.ui_obj)
        show_stats_action = QtWidgets.QAction('Show image statistics', self.ui_obj)
        update_display_action = QtWidgets.QAction('Update display', self.ui_obj)
        toggle_image_control_action = QtWidgets.QAction('Show image controls', self.ui_obj)
        toggle_image_control_action.setCheckable(True)
        toggle_image_control_action.setChecked(True)
        toggle_alg_1_control_action = QtWidgets.QAction('Show column detection controls', self.ui_obj)
        toggle_alg_1_control_action.setCheckable(True)
        toggle_alg_1_control_action.setChecked(True)
        toggle_alg_2_control_action = QtWidgets.QAction('Show column characterization controls', self.ui_obj)
        toggle_alg_2_control_action.setCheckable(True)
        toggle_alg_2_control_action.setChecked(True)
        toggle_column_control_action = QtWidgets.QAction('Show selected column controls', self.ui_obj)
        toggle_column_control_action.setCheckable(True)
        toggle_column_control_action.setChecked(True)
        toggle_overlay_control_action = QtWidgets.QAction('Show overlay controls', self.ui_obj)
        toggle_overlay_control_action.setCheckable(True)
        toggle_overlay_control_action.setChecked(True)
        # - Process
        image_correction_action = QtWidgets.QAction('Image corrections', self.ui_obj)
        image_filter_action = QtWidgets.QAction('Image filters', self.ui_obj)
        image_adjustments_action = QtWidgets.QAction('Image adjustments', self.ui_obj)
        continue_detection_action = QtWidgets.QAction('Continue column detection', self.ui_obj)
        restart_detection_action = QtWidgets.QAction('Restart column detection', self.ui_obj)
        continue_analysis_action = QtWidgets.QAction('Continue column analysis', self.ui_obj)
        restart_analysis_action = QtWidgets.QAction('Restart column analysis', self.ui_obj)
        # - export
        export_data_action = QtWidgets.QAction('Export data', self.ui_obj)
        export_raw_image_action = QtWidgets.QAction('Export raw image', self.ui_obj)
        export_column_position_image_action = QtWidgets.QAction('Export column position image', self.ui_obj)
        export_overlay_image_action = QtWidgets.QAction('Export overlay image', self.ui_obj)
        export_atomic_graph_action = QtWidgets.QAction('Export atomic graph', self.ui_obj)
        # - Debug
        advanced_debug_mode_action = QtWidgets.QAction('Advanced debug mode', self.ui_obj)
        advanced_debug_mode_action.setCheckable(True)
        advanced_debug_mode_action.blockSignals(True)
        advanced_debug_mode_action.setChecked(False)
        advanced_debug_mode_action.blockSignals(False)
        add_mark_action = QtWidgets.QAction('Add mark to terminal', self.ui_obj)
        reset_flags_action = QtWidgets.QAction('Reset all flags', self.ui_obj)
        set_control_file_action = QtWidgets.QAction('Set control instance', self.ui_obj)
        display_deviations_action = QtWidgets.QAction('Display deviation stats', self.ui_obj)
        run_validation_test_action = QtWidgets.QAction('Run algorithm benchmark', self.ui_obj)
        test_consistency_action = QtWidgets.QAction('Reset levels', self.ui_obj)
        invert_precipitate_levels_action = QtWidgets.QAction('Invert precipitate levels', self.ui_obj)
        ad_hoc_action = QtWidgets.QAction('Ad Hoc functionality', self.ui_obj)
        # - help
        there_is_no_help_action = QtWidgets.QAction('HJALP!', self.ui_obj)

        # Add actions to menus
        # - file
        file.addAction(new_action)
        file.addAction(open_action)
        file.addAction(save_action)
        file.addAction(close_action)
        file.addAction(exit_action)
        # - edit
        # - View
        view.addAction(view_image_title_action)
        view.addAction(show_stats_action)
        view.addAction(update_display_action)
        view.addSeparator()
        view.addAction(toggle_image_control_action)
        view.addAction(toggle_alg_1_control_action)
        view.addAction(toggle_alg_2_control_action)
        view.addAction(toggle_column_control_action)
        view.addAction(toggle_overlay_control_action)
        # - Process
        process.addAction(image_correction_action)
        process.addAction(image_filter_action)
        process.addAction(image_adjustments_action)
        column_detection.addAction(continue_detection_action)
        column_detection.addAction(restart_detection_action)
        column_analysis.addAction(continue_analysis_action)
        column_analysis.addAction(restart_analysis_action)
        # - Export
        export.addAction(export_data_action)
        image.addAction(export_raw_image_action)
        image.addAction(export_column_position_image_action)
        image.addAction(export_overlay_image_action)
        image.addAction(export_atomic_graph_action)
        # - Debug
        debug.addAction(advanced_debug_mode_action)
        debug.addAction(add_mark_action)
        debug.addAction(reset_flags_action)
        debug.addAction(set_control_file_action)
        debug.addAction(run_validation_test_action)
        debug.addAction(display_deviations_action)
        debug.addAction(test_consistency_action)
        debug.addAction(invert_precipitate_levels_action)
        debug.addAction(ad_hoc_action)
        # - Help
        help.addAction(there_is_no_help_action)

        # Events
        # - file
        new_action.triggered.connect(self.ui_obj.menu_new_trigger)
        open_action.triggered.connect(self.ui_obj.menu_open_trigger)
        save_action.triggered.connect(self.ui_obj.menu_save_trigger)
        close_action.triggered.connect(self.ui_obj.menu_close_trigger)
        exit_action.triggered.connect(self.ui_obj.menu_exit_trigger)
        # - edit
        # - view
        view_image_title_action.triggered.connect(self.ui_obj.menu_view_image_title_trigger)
        show_stats_action.triggered.connect(self.ui_obj.menu_show_stats_trigger)
        update_display_action.triggered.connect(self.ui_obj.menu_update_display)
        toggle_image_control_action.triggered.connect(self.ui_obj.menu_toggle_image_control_trigger)
        toggle_alg_1_control_action.triggered.connect(self.ui_obj.menu_toggle_alg_1_control_trigger)
        toggle_alg_2_control_action.triggered.connect(self.ui_obj.menu_toggle_alg_2_control_trigger)
        toggle_column_control_action.triggered.connect(self.ui_obj.menu_toggle_column_control_trigger)
        toggle_overlay_control_action.triggered.connect(self.ui_obj.menu_toggle_overlay_control_trigger)
        # - Process
        image_correction_action.triggered.connect(self.ui_obj.menu_image_correction_trigger)
        image_filter_action.triggered.connect(self.ui_obj.menu_image_filter_trigger)
        image_adjustments_action.triggered.connect(self.ui_obj.menu_image_adjustments_trigger)
        continue_detection_action.triggered.connect(self.ui_obj.menu_continue_detection_trigger)
        restart_detection_action.triggered.connect(self.ui_obj.menu_restart_detection_trigger)
        continue_analysis_action.triggered.connect(self.ui_obj.menu_continue_analysis_trigger)
        restart_analysis_action.triggered.connect(self.ui_obj.menu_restart_analysis_trigger)
        # - Export
        export_data_action.triggered.connect(self.ui_obj.menu_export_data_trigger)
        export_raw_image_action.triggered.connect(self.ui_obj.menu_export_raw_image_trigger)
        export_column_position_image_action.triggered.connect(self.ui_obj.menu_export_column_position_image_trigger)
        export_overlay_image_action.triggered.connect(self.ui_obj.menu_export_overlay_image_trigger)
        export_atomic_graph_action.triggered.connect(self.ui_obj.menu_export_atomic_graph_trigger)
        # - debug
        advanced_debug_mode_action.triggered.connect(self.ui_obj.menu_toggle_debug_mode_trigger)
        add_mark_action.triggered.connect(self.ui_obj.menu_add_mark_trigger)
        reset_flags_action.triggered.connect(self.ui_obj.menu_clear_flags_trigger)
        set_control_file_action.triggered.connect(self.ui_obj.menu_set_control_file_trigger)
        run_validation_test_action.triggered.connect(self.ui_obj.menu_run_benchmark_trigger)
        display_deviations_action.triggered.connect(self.ui_obj.menu_display_deviations_trigger)
        test_consistency_action.triggered.connect(self.ui_obj.menu_test_consistency_trigger)
        invert_precipitate_levels_action.triggered.connect(self.ui_obj.menu_invert_precipitate_columns_trigger)
        ad_hoc_action.triggered.connect(self.ui_obj.menu_ad_hoc_trigger)
        # - hjelp
        there_is_no_help_action.triggered.connect(self.ui_obj.menu_there_is_no_help_trigger)


# ----------
# Custom dialogs:
# ----------

