# By Haakon Tvedt @ NTNU
"""Module container for high-level custom GUI-elements"""

from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import core
import GUI_custom_components


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
        for vertex in self.ui_obj.project_instance.graph.vertices:
            self.interactive_position_objects.append(GUI_custom_components.InteractivePosColumn(self.ui_obj, vertex.i))
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
        for vertex in self.ui_obj.project_instance.graph.vertices:
            self.interactive_overlay_objects.append(GUI_custom_components.InteractiveOverlayColumn(self.ui_obj, vertex.i))
            self.addItem(self.interactive_overlay_objects[-1])


class AtomicGraph(QtWidgets.QGraphicsScene):

    def __init__(self, *args, ui_obj=None, background=None, scale_factor=1):
        """Initialize a custom QtWidgets.QGraphicsScene object for **atomic graphs**."""

        super().__init__(*args)

        self.ui_obj = ui_obj
        self.r = self.ui_obj.project_instance.r
        self.scale_factor = scale_factor
        self.interactive_vertex_objects = []
        self.edges = []
        self.background_image = background
        if self.background_image is not None:
            self.addPixmap(self.background_image)
        self.re_draw()

    def re_draw(self):
        """Redraw contents."""
        self.re_draw_vertices()
        self.re_draw_edges()

    def re_draw_vertices(self):
        """Redraws all column elements."""
        for vertex in self.ui_obj.project_instance.graph.vertices:
            self.interactive_vertex_objects.append(GUI_custom_components.InteractiveGraphColumn(self.ui_obj, vertex.i))
            self.addItem(self.interactive_vertex_objects[-1])

    def re_draw_edges(self):
        """Redraws all edge elements."""
        for edge in self.ui_obj.project_instance.graph.edges:
            consistent = edge.is_reciprocated
            dislocation = not edge.is_legal_levels
            p1 = edge.vertex_a.real_coor()
            p2 = edge.vertex_b.real_coor()
            self.edges.append(GUI_custom_components.Arrow(p1, p2, self.r, self.scale_factor, consistent, dislocation))
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

    def __init__(self, parent=None, ui_obj=None):
        super(ZoomGraphicsView, self).__init__(parent)
        self.ui_obj = ui_obj

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
        if self.ui_obj is not None:
            self.ui_obj.key_press_trigger(event.key())


# ----------
# Custom Mainwindow tools:
# ----------


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

