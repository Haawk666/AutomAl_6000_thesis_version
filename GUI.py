# By Haakon Tvedt @ NTNU
"""Module containing the Main Window class. This is the top-level GUI and contains the *business logic* of the
interface."""

from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import numpy as np
import numpy.core._dtype_ctypes  # This is needed because of a bug in pyinstaller
import mat_op
import core
import GUI_elements
import utils
import dev_module
import logging

# Instantiate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Instantiate logging handlers and add formatter
file_handler = logging.FileHandler('log_test.log')
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s:%(levelname)s:%(funcName)s:%(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


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

        # gs = QGraphicsView
        self.gs_raw_image = GUI_elements.RawImage(ui_obj=self, background=self.no_graphic)
        self.gs_atomic_positions = GUI_elements.AtomicPositions(ui_obj=self, background=self.no_graphic)
        self.gs_overlay_composition = GUI_elements.OverlayComposition(ui_obj=self, background=self.no_graphic)
        self.gs_atomic_graph = GUI_elements.AtomicGraph(ui_obj=self, background=self.no_graphic)
        self.gs_atomic_sub_graph = GUI_elements.AtomicSubGraph(ui_obj=self, background=self.no_graphic)
        self.gs_search_matrix = GUI_elements.RawImage(ui_obj=self, background=self.no_graphic)
        self.gs_fft = GUI_elements.RawImage(ui_obj=self, background=self.no_graphic)

        # gv = QGraphicsView
        self.gv_raw_image = GUI_elements.ZoomGraphicsView(self.gs_raw_image)
        self.gv_atomic_positions = GUI_elements.ZoomGraphicsView(self.gs_atomic_positions)
        self.gv_overlay_composition = GUI_elements.ZoomGraphicsView(self.gs_overlay_composition)
        self.gv_atomic_graph = GUI_elements.ZoomGraphicsView(self.gs_atomic_graph)
        self.gv_atomic_sub_graph = GUI_elements.ZoomGraphicsView(self.gs_atomic_sub_graph)
        self.gv_search_matrix = GUI_elements.ZoomGraphicsView(self.gs_search_matrix)
        self.gv_fft = GUI_elements.ZoomGraphicsView(self.gs_fft)

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
        # GUI_elements.MenuBar(self.menuBar(), self)

        # Generate elements
        self.setWindowTitle('AACC - Automatic Atomic Column Characterizer - By Haakon Tvedt @ NTNU. Version {}.{}.{}'.
                            format(self.version[0], self.version[1], self.version[2]))
        self.resize(1500, 900)
        self.move(50, 30)
        self.statusBar().showMessage('Ready')

        # Display
        self.show()

