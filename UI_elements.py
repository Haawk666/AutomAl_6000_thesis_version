from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import Core


class InteractivePosColumn(QtWidgets.QGraphicsEllipseItem):

    def __init__(self, *args):
        super().__init__(*args)

        self.obj = None
        self.x_0 = 0
        self.y_0 = 0
        self.i = 0

    def reference_object(self, obj, i):

        self.obj = obj
        self.i = i

    def mouseReleaseEvent(self, event: 'QtWidgets.QGraphicsEllipseItem.mouseReleaseEvent'):

        self.x_0 = int(self.x() + self.obj.project_instance.r)
        self.y_0 = int(self.y() + self.obj.project_instance.r)

        if not self.obj.control_window.chb_move.isChecked():

            self.obj.column_selected(self.i)

            if self.obj.previous_pos_obj is None:
                pass
            else:
                self.obj.pos_objects[self.obj.previous_pos_obj.i].unselect()

            if self.obj.previous_overlay_obj is None:
                pass
            else:
                self.obj.overlay_objects[self.obj.previous_overlay_obj.i].unselect()

            self.select()
            self.obj.overlay_objects[self.i].select()

            self.obj.previous_pos_obj = self
            self.obj.previous_overlay_obj = self.obj.overlay_objects[self.i]

        else:

            self.obj.control_window.lbl_column_index.setText('Column index: ' + str(self.i))
            self.obj.control_window.lbl_column_x_pos.setText('x: ' + str(self.x_0))
            self.obj.control_window.lbl_column_y_pos.setText('y: ' + str(self.y_0))
            self.obj.control_window.lbl_column_peak_gamma.setText(
                'Peak gamma: ' + str(self.obj.project_instance.columns[self.i].peak_gamma))
            self.obj.control_window.lbl_column_species.setText(
                'Atomic species: ' + self.obj.project_instance.columns[self.i].atomic_species)
            self.obj.control_window.lbl_column_level.setText('Level: ' + str(self.obj.project_instance.columns[self.i].level))
            self.obj.control_window.lbl_confidence.setText('Confidence: ' + str(self.obj.project_instance.columns[self.i].confidence))

        self.obj.selected_column = self.i

    def select(self):

        self.setPen(self.obj.yellow_pen)
        self.setBrush(QtGui.QBrush(QtCore.Qt.transparent))

    def unselect(self):

        if self.obj.project_instance.columns[self.i].show_in_overlay:
            self.setPen(self.obj.red_pen)
        else:
            self.setPen(self.obj.dark_red_pen)

        self.setBrush(QtGui.QBrush(QtCore.Qt.transparent))


class InteractiveOverlayColumn(QtWidgets.QGraphicsEllipseItem):

    def __init__(self, *args):
        super().__init__(*args)

        self.obj = None
        self.x_0 = 0
        self.y_0 = 0
        self.i = 0

    def reference_object(self, obj, i):

        self.obj = obj
        self.i = i

    def mouseReleaseEvent(self, event: 'QtWidgets.QGraphicsEllipseItem.mouseReleaseEvent'):

        self.x_0 = int(self.x() + np.round(self.obj.project_instance.r / 2))
        self.y_0 = int(self.y() + np.round(self.obj.project_instance.r / 2))

        if not self.obj.control_window.chb_move.isChecked():

            if self.obj.previous_pos_obj is None:
                pass
            else:
                self.obj.pos_objects[self.obj.previous_pos_obj.i].unselect()

            if self.obj.previous_overlay_obj is None:
                pass
            else:
                self.obj.overlay_objects[self.obj.previous_overlay_obj.i].unselect()

            self.obj.column_selected(self.i)

            self.select()
            self.obj.pos_objects[self.i].select()

            self.obj.previous_overlay_obj = self
            self.obj.previous_pos_obj = self.obj.pos_objects[self.i]

        else:

            self.obj.control_window.lbl_column_index.setText('Column index: ' + str(self.i))
            self.obj.control_window.lbl_column_x_pos.setText('x: ' + str(self.x_0))
            self.obj.control_window.lbl_column_y_pos.setText('y: ' + str(self.y_0))
            self.obj.control_window.lbl_column_peak_gamma.setText(
                'Peak gamma: ' + str(self.obj.project_instance.columns[self.i].peak_gamma))
            self.obj.control_window.lbl_column_species.setText(
                'Atomic species: ' + self.obj.project_instance.columns[self.i].atomic_species)
            self.obj.control_window.lbl_column_level.setText('Level: ' + str(self.obj.project_instance.columns[self.i].level))
            self.obj.control_window.lbl_confidence.setText('Confidence: ' + str(self.obj.project_instance.columns[self.i].confidence))

        self.obj.selected_column = self.i

    def select(self):
        pass

    def unselect(self):
        pass


class InteractiveGraphVertex(QtWidgets.QGraphicsEllipseItem):

    def __init__(self, *args):
        super().__init__(*args)

        self.obj = None
        self.x_0 = 0
        self.y_0 = 0
        self.i = 0

    def reference_object(self, obj, i):

        self.obj = obj
        self.i = i

    def mouseReleaseEvent(self, event: 'QtWidgets.QGraphicsEllipseItem.mouseReleaseEvent'):

        self.x_0 = int(self.x() + np.round(self.obj.project_instance.r / 2))
        self.y_0 = int(self.y() + np.round(self.obj.project_instance.r / 2))
        self.obj.selected_column = self.i

        if not self.obj.control_window.chb_move.isChecked():

            if self.obj.previous_pos_obj is None:
                pass
            else:
                self.obj.pos_objects[self.obj.previous_pos_obj.i].unselect()

            if self.obj.previous_overlay_obj is None:
                pass
            else:
                self.obj.overlay_objects[self.obj.previous_overlay_obj.i].unselect()

            self.obj.column_selected(self.i)

            self.select()
            self.obj.pos_objects[self.i].select()

            self.obj.previous_overlay_obj = self
            self.obj.previous_pos_obj = self.obj.pos_objects[self.i]

        else:

            self.obj.control_window.lbl_column_index.setText('Column index: ' + str(self.i))
            self.obj.control_window.lbl_column_x_pos.setText('x: ' + str(self.x_0))
            self.obj.control_window.lbl_column_y_pos.setText('y: ' + str(self.y_0))
            self.obj.control_window.lbl_column_peak_gamma.setText(
                'Peak gamma: ' + str(self.obj.project_instance.columns[self.i].peak_gamma))
            self.obj.control_window.lbl_column_species.setText(
                'Atomic species: ' + self.obj.project_instance.columns[self.i].atomic_species)
            self.obj.control_window.lbl_column_level.setText('Level: ' + str(self.obj.project_instance.columns[self.i].level))
            self.obj.control_window.lbl_confidence.setText('Confidence: ' + str(self.obj.project_instance.columns[self.i].confidence))

    def select(self):
        pass

    def unselect(self):
        pass


class SetIndicesDialog(QtWidgets.QDialog):

    def __init__(self, *args):
        super().__init__(*args)

        self.obj = None
        self.i = 0

        self.Combo_1 = QtWidgets.QComboBox()
        self.Combo_2 = QtWidgets.QComboBox()
        self.Combo_3 = QtWidgets.QComboBox()
        self.Combo_4 = QtWidgets.QComboBox()
        self.Combo_5 = QtWidgets.QComboBox()
        self.Combo_6 = QtWidgets.QComboBox()
        self.Combo_7 = QtWidgets.QComboBox()
        self.Combo_8 = QtWidgets.QComboBox()

        self.lbl_old_indices = QtWidgets.QLabel('Old indices: ')
        self.lbl_new_indices = QtWidgets.QLabel('New indices: ')

        self.lbl_combo_1 = QtWidgets.QLabel('combo')
        self.lbl_combo_2 = QtWidgets.QLabel('combo')
        self.lbl_combo_3 = QtWidgets.QLabel('combo')
        self.lbl_combo_4 = QtWidgets.QLabel('combo')
        self.lbl_combo_5 = QtWidgets.QLabel('combo')
        self.lbl_combo_6 = QtWidgets.QLabel('combo')
        self.lbl_combo_7 = QtWidgets.QLabel('combo')
        self.lbl_combo_8 = QtWidgets.QLabel('combo')

        self.btn_ok = QtWidgets.QPushButton('Set')
        self.btn_ok.clicked.connect(self.ok_trigger)
        self.btn_cancel = QtWidgets.QPushButton('Cancel')
        self.btn_cancel.clicked.connect(self.cancel_trigger)

        self.indices = None
        self.new_indices = None

    def reference_object(self, obj, i):

        self.obj = obj
        self.i = i
        self.indices = obj.project_instance.columns[i].neighbour_indices

    def gen_layout(self):

        v_lay_0 = QtWidgets.QVBoxLayout()
        v_lay_1 = QtWidgets.QVBoxLayout()
        v_lay_2 = QtWidgets.QVBoxLayout()
        v_lay_3 = QtWidgets.QVBoxLayout()
        v_lay_4 = QtWidgets.QVBoxLayout()
        v_lay_5 = QtWidgets.QVBoxLayout()
        v_lay_6 = QtWidgets.QVBoxLayout()
        v_lay_7 = QtWidgets.QVBoxLayout()
        v_lay_8 = QtWidgets.QVBoxLayout()

        h_lay = QtWidgets.QHBoxLayout()

        v_lay_0.addWidget(self.lbl_old_indices)
        v_lay_0.addWidget(self.lbl_new_indices)

        v_lay_1.addWidget(self.lbl_combo_1)
        v_lay_1.addWidget(self.Combo_1)

        v_lay_2.addWidget(self.lbl_combo_2)
        v_lay_2.addWidget(self.Combo_2)

        v_lay_3.addWidget(self.lbl_combo_3)
        v_lay_3.addWidget(self.Combo_3)

        v_lay_4.addWidget(self.lbl_combo_4)
        v_lay_4.addWidget(self.Combo_4)

        v_lay_5.addWidget(self.lbl_combo_5)
        v_lay_5.addWidget(self.Combo_5)

        v_lay_6.addWidget(self.lbl_combo_6)
        v_lay_6.addWidget(self.Combo_6)

        v_lay_7.addWidget(self.lbl_combo_7)
        v_lay_7.addWidget(self.Combo_7)

        v_lay_8.addWidget(self.lbl_combo_8)
        v_lay_8.addWidget(self.Combo_8)

        btn_lay = QtWidgets.QHBoxLayout()
        btn_lay.addStretch()
        btn_lay.addWidget(self.btn_ok)
        btn_lay.addWidget(self.btn_cancel)
        btn_lay.addStretch()

        h_lay.addLayout(v_lay_0)
        h_lay.addLayout(v_lay_1)
        h_lay.addLayout(v_lay_2)
        h_lay.addLayout(v_lay_3)
        h_lay.addLayout(v_lay_4)
        h_lay.addLayout(v_lay_5)
        h_lay.addLayout(v_lay_6)
        h_lay.addLayout(v_lay_7)
        h_lay.addLayout(v_lay_8)

        lay = QtWidgets.QVBoxLayout()
        lay.addLayout(h_lay)
        lay.addLayout(btn_lay)

        self.lbl_combo_1.setText(str(self.indices[0]))
        self.lbl_combo_2.setText(str(self.indices[1]))
        self.lbl_combo_3.setText(str(self.indices[2]))
        self.lbl_combo_4.setText(str(self.indices[3]))
        self.lbl_combo_5.setText(str(self.indices[4]))
        self.lbl_combo_6.setText(str(self.indices[5]))
        self.lbl_combo_7.setText(str(self.indices[6]))
        self.lbl_combo_8.setText(str(self.indices[7]))

        for x in range(0, self.indices.shape[0]):

            self.Combo_1.addItem(str(self.indices[x]))
            self.Combo_2.addItem(str(self.indices[x]))
            self.Combo_3.addItem(str(self.indices[x]))
            self.Combo_4.addItem(str(self.indices[x]))
            self.Combo_5.addItem(str(self.indices[x]))
            self.Combo_6.addItem(str(self.indices[x]))
            self.Combo_7.addItem(str(self.indices[x]))
            self.Combo_8.addItem(str(self.indices[x]))

        self.Combo_1.setCurrentIndex(0)
        self.Combo_2.setCurrentIndex(1)
        self.Combo_3.setCurrentIndex(2)
        self.Combo_4.setCurrentIndex(3)
        self.Combo_5.setCurrentIndex(4)
        self.Combo_6.setCurrentIndex(5)
        self.Combo_7.setCurrentIndex(6)
        self.Combo_8.setCurrentIndex(7)

        self.setLayout(lay)

    def ok_trigger(self):

        all_is_good = True

        self.new_indices = np.ndarray([8], dtype=int)

        print(str(self.indices))
        print(str(self.new_indices))

        self.new_indices[0] = self.indices[self.Combo_1.currentIndex()]
        self.new_indices[1] = self.indices[self.Combo_2.currentIndex()]
        self.new_indices[2] = self.indices[self.Combo_3.currentIndex()]
        self.new_indices[3] = self.indices[self.Combo_4.currentIndex()]
        self.new_indices[4] = self.indices[self.Combo_5.currentIndex()]
        self.new_indices[5] = self.indices[self.Combo_6.currentIndex()]
        self.new_indices[6] = self.indices[self.Combo_7.currentIndex()]
        self.new_indices[7] = self.indices[self.Combo_8.currentIndex()]

        print(str(self.indices))
        print(str(self.new_indices))

        print(str(self.Combo_1.currentIndex()))
        print(str(self.Combo_2.currentIndex()))
        print(str(self.Combo_3.currentIndex()))
        print(str(self.Combo_4.currentIndex()))
        print(str(self.Combo_5.currentIndex()))
        print(str(self.Combo_6.currentIndex()))
        print(str(self.Combo_7.currentIndex()))
        print(str(self.Combo_8.currentIndex()))

        for x in range(0, 8):

            for y in range(0, 8):

                if self.new_indices[x] == self.new_indices[y] and not x == y:

                    print(str(x), str(y))
                    print(str(self.new_indices[x]) + ', ' + str(self.new_indices[y]))

                    all_is_good = False

        print(str(all_is_good))

        if all_is_good:

            self.obj.project_instance.columns[self.i].neighbour_indices = self.new_indices

            self.close()

        else:

            msg = QtWidgets.QMessageBox()
            msg.setText('Error!')
            msg.exec_()

    def cancel_trigger(self):

        self.close()


class SetIndicesManuallyDialog(QtWidgets.QDialog):

    def __init__(self, *args):
        super().__init__(*args)

        self.obj = None
        self.i = 0

        self.Combo_1 = QtWidgets.QLineEdit()
        self.Combo_2 = QtWidgets.QLineEdit()
        self.Combo_3 = QtWidgets.QLineEdit()
        self.Combo_4 = QtWidgets.QLineEdit()
        self.Combo_5 = QtWidgets.QLineEdit()
        self.Combo_6 = QtWidgets.QLineEdit()
        self.Combo_7 = QtWidgets.QLineEdit()
        self.Combo_8 = QtWidgets.QLineEdit()

        self.lbl_old_indices = QtWidgets.QLabel('Old indices: ')
        self.lbl_new_indices = QtWidgets.QLabel('New indices: ')

        self.lbl_combo_1 = QtWidgets.QLabel('combo')
        self.lbl_combo_2 = QtWidgets.QLabel('combo')
        self.lbl_combo_3 = QtWidgets.QLabel('combo')
        self.lbl_combo_4 = QtWidgets.QLabel('combo')
        self.lbl_combo_5 = QtWidgets.QLabel('combo')
        self.lbl_combo_6 = QtWidgets.QLabel('combo')
        self.lbl_combo_7 = QtWidgets.QLabel('combo')
        self.lbl_combo_8 = QtWidgets.QLabel('combo')

        self.btn_ok = QtWidgets.QPushButton('Set')
        self.btn_ok.clicked.connect(self.ok_trigger)
        self.btn_cancel = QtWidgets.QPushButton('Cancel')
        self.btn_cancel.clicked.connect(self.cancel_trigger)

        self.indices = None
        self.new_indices = None

    def reference_object(self, obj, i):

        self.obj = obj
        self.i = i
        self.indices = obj.project_instance.columns[i].neighbour_indices

    def gen_layout(self):

        v_lay_0 = QtWidgets.QVBoxLayout()
        v_lay_1 = QtWidgets.QVBoxLayout()
        v_lay_2 = QtWidgets.QVBoxLayout()
        v_lay_3 = QtWidgets.QVBoxLayout()
        v_lay_4 = QtWidgets.QVBoxLayout()
        v_lay_5 = QtWidgets.QVBoxLayout()
        v_lay_6 = QtWidgets.QVBoxLayout()
        v_lay_7 = QtWidgets.QVBoxLayout()
        v_lay_8 = QtWidgets.QVBoxLayout()

        h_lay = QtWidgets.QHBoxLayout()

        v_lay_0.addWidget(self.lbl_old_indices)
        v_lay_0.addWidget(self.lbl_new_indices)

        v_lay_1.addWidget(self.lbl_combo_1)
        v_lay_1.addWidget(self.Combo_1)

        v_lay_2.addWidget(self.lbl_combo_2)
        v_lay_2.addWidget(self.Combo_2)

        v_lay_3.addWidget(self.lbl_combo_3)
        v_lay_3.addWidget(self.Combo_3)

        v_lay_4.addWidget(self.lbl_combo_4)
        v_lay_4.addWidget(self.Combo_4)

        v_lay_5.addWidget(self.lbl_combo_5)
        v_lay_5.addWidget(self.Combo_5)

        v_lay_6.addWidget(self.lbl_combo_6)
        v_lay_6.addWidget(self.Combo_6)

        v_lay_7.addWidget(self.lbl_combo_7)
        v_lay_7.addWidget(self.Combo_7)

        v_lay_8.addWidget(self.lbl_combo_8)
        v_lay_8.addWidget(self.Combo_8)

        btn_lay = QtWidgets.QHBoxLayout()
        btn_lay.addStretch()
        btn_lay.addWidget(self.btn_ok)
        btn_lay.addWidget(self.btn_cancel)
        btn_lay.addStretch()

        h_lay.addLayout(v_lay_0)
        h_lay.addLayout(v_lay_1)
        h_lay.addLayout(v_lay_2)
        h_lay.addLayout(v_lay_3)
        h_lay.addLayout(v_lay_4)
        h_lay.addLayout(v_lay_5)
        h_lay.addLayout(v_lay_6)
        h_lay.addLayout(v_lay_7)
        h_lay.addLayout(v_lay_8)

        lay = QtWidgets.QVBoxLayout()
        lay.addLayout(h_lay)
        lay.addLayout(btn_lay)

        self.lbl_combo_1.setText(str(self.indices[0]))
        self.lbl_combo_2.setText(str(self.indices[1]))
        self.lbl_combo_3.setText(str(self.indices[2]))
        self.lbl_combo_4.setText(str(self.indices[3]))
        self.lbl_combo_5.setText(str(self.indices[4]))
        self.lbl_combo_6.setText(str(self.indices[5]))
        self.lbl_combo_7.setText(str(self.indices[6]))
        self.lbl_combo_8.setText(str(self.indices[7]))

        self.Combo_1.setText(str(self.indices[0]))
        self.Combo_2.setText(str(self.indices[1]))
        self.Combo_3.setText(str(self.indices[2]))
        self.Combo_4.setText(str(self.indices[3]))
        self.Combo_5.setText(str(self.indices[4]))
        self.Combo_6.setText(str(self.indices[5]))
        self.Combo_7.setText(str(self.indices[6]))
        self.Combo_8.setText(str(self.indices[7]))

        self.setLayout(lay)

    def ok_trigger(self):

        self.new_indices = np.ndarray([8], dtype=int)

        self.new_indices[0] = int(self.Combo_1.text())
        self.new_indices[1] = int(self.Combo_2.text())
        self.new_indices[2] = int(self.Combo_3.text())
        self.new_indices[3] = int(self.Combo_4.text())
        self.new_indices[4] = int(self.Combo_5.text())
        self.new_indices[5] = int(self.Combo_6.text())
        self.new_indices[6] = int(self.Combo_7.text())
        self.new_indices[7] = int(self.Combo_8.text())

        self.obj.project_instance.columns[self.i].neighbour_indices = self.new_indices

        self.close()

    def cancel_trigger(self):

        self.close()


class SmallButton(QtWidgets.QPushButton):

    def __init__(self, *args):
        super().__init__(*args)

        self.font_tiny = QtGui.QFont()
        self.font_tiny.setPixelSize(9)

        self.setMaximumHeight(15)
        self.setMaximumWidth(30)
        self.setFont(self.font_tiny)


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

        self.font_tiny = QtGui.QFont()
        self.font_tiny.setPixelSize(9)

        self.lbl_num_detected_columns = QtWidgets.QLabel('Number of detected columns: ')
        self.lbl_image_width = QtWidgets.QLabel('Image width (pixels): ')
        self.lbl_image_height = QtWidgets.QLabel('Image height (pixels): ')

        self.lbl_atomic_radii = QtWidgets.QLabel('Approx atomic radii (pixels): ')
        self.lbl_overhead_radii = QtWidgets.QLabel('Overhead (pixels): ')
        self.lbl_detection_threshold = QtWidgets.QLabel('Detection threshold value: ')
        self.lbl_search_matrix_peak = QtWidgets.QLabel('Search matrix peak: ')
        self.lbl_search_size = QtWidgets.QLabel('Search size: ')
        self.lbl_scale = QtWidgets.QLabel('Scale (pm / pixel): ')

        self.lbl_alloy = QtWidgets.QLabel('Alloy: ')

        self.lbl_std_1 = QtWidgets.QLabel('Standard deviation 1: ')
        self.lbl_std_2 = QtWidgets.QLabel('Standard deviation 2: ')
        self.lbl_std_3 = QtWidgets.QLabel('Standard deviation 3: ')
        self.lbl_std_4 = QtWidgets.QLabel('Standard deviation 4: ')
        self.lbl_std_5 = QtWidgets.QLabel('Standard deviation 5: ')
        self.lbl_std_8 = QtWidgets.QLabel('Standard deviation 8: ')
        self.lbl_cert_threshold = QtWidgets.QLabel('Certainty threshold: ')

        self.lbl_column_index = QtWidgets.QLabel('Column index: ')
        self.lbl_column_x_pos = QtWidgets.QLabel('x: ')
        self.lbl_column_y_pos = QtWidgets.QLabel('y: ')
        self.lbl_column_peak_gamma = QtWidgets.QLabel('Peak gamma: ')
        self.lbl_column_avg_gamma = QtWidgets.QLabel('Avg gamma: ')
        self.lbl_column_species = QtWidgets.QLabel('Atomic species: ')
        self.lbl_column_level = QtWidgets.QLabel('Level: ')
        self.lbl_confidence = QtWidgets.QLabel('Confidence: ')
        self.lbl_prob_vector = QtWidgets.QLabel('Probability vector: ')
        self.lbl_neighbours = QtWidgets.QLabel('Nearest neighbours: ')

        # Checkboxes

        self.chb_precipitate_column = QtWidgets.QCheckBox('Precipitate column')
        self.chb_show = QtWidgets.QCheckBox('Show in overlay')
        self.chb_move = QtWidgets.QCheckBox('Enable move')

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

        self.chb_precipitate_column.toggled.connect(self.ui_obj.toggle_precipitate_trigger)
        self.chb_show.toggled.connect(self.ui_obj.toggle_show_trigger)
        self.chb_move.toggled.connect(self.ui_obj.move_trigger)

        self.chb_raw_image.toggled.connect(self.ui_obj.update_central_widget)
        self.chb_black_background.toggled.connect(self.ui_obj.update_central_widget)
        self.chb_structures.toggled.connect(self.ui_obj.update_central_widget)
        self.chb_boarders.toggled.connect(self.ui_obj.update_central_widget)
        self.chb_si_columns.toggled.connect(self.ui_obj.toggle_si_trigger)
        self.chb_si_network.toggled.connect(self.ui_obj.update_central_widget)
        self.chb_cu_columns.toggled.connect(self.ui_obj.toggle_cu_trigger)
        self.chb_cu_network.toggled.connect(self.ui_obj.update_central_widget)
        self.chb_al_columns.toggled.connect(self.ui_obj.toggle_al_trigger)
        self.chb_al_network.toggled.connect(self.ui_obj.update_central_widget)
        self.chb_ag_columns.toggled.connect(self.ui_obj.toggle_ag_trigger)
        self.chb_ag_network.toggled.connect(self.ui_obj.update_central_widget)
        self.chb_mg_columns.toggled.connect(self.ui_obj.toggle_mg_trigger)
        self.chb_mg_network.toggled.connect(self.ui_obj.update_central_widget)
        self.chb_un_columns.toggled.connect(self.ui_obj.toggle_un_trigger)
        self.chb_columns.toggled.connect(self.ui_obj.toggle_column_trigger)
        self.chb_al_mesh.toggled.connect(self.ui_obj.toggle_al_mesh_trigger)
        self.chb_neighbours.toggled.connect(self.ui_obj.update_central_widget)
        self.chb_legend.toggled.connect(self.ui_obj.update_central_widget)
        self.chb_scalebar.toggled.connect(self.ui_obj.update_central_widget)

        # The Set values buttons

        self.btn_set_threshold = SmallButton('Set', self)
        self.btn_set_threshold.clicked.connect(self.ui_obj.set_threshold_trigger)
        btn_set_threshold_layout = QtWidgets.QHBoxLayout()
        btn_set_threshold_layout.addWidget(self.btn_set_threshold)
        btn_set_threshold_layout.addWidget(self.lbl_detection_threshold)
        btn_set_threshold_layout.addStretch()

        self.btn_set_search_size = SmallButton('Set', self)
        self.btn_set_search_size.clicked.connect(self.ui_obj.set_search_size_trigger)
        btn_set_search_size_layout = QtWidgets.QHBoxLayout()
        btn_set_search_size_layout.addWidget(self.btn_set_search_size)
        btn_set_search_size_layout.addWidget(self.lbl_search_size)
        btn_set_search_size_layout.addStretch()

        self.btn_set_scale = SmallButton('Set', self)
        self.btn_set_scale.clicked.connect(self.ui_obj.set_scale_trigger)
        btn_set_scale_layout = QtWidgets.QHBoxLayout()
        btn_set_scale_layout.addWidget(self.btn_set_scale)
        btn_set_scale_layout.addWidget(self.lbl_scale)
        btn_set_scale_layout.addStretch()

        self.btn_set_alloy = SmallButton('Set', self)
        self.btn_set_alloy.clicked.connect(self.ui_obj.set_alloy_trigger)
        btn_set_alloy_layout = QtWidgets.QHBoxLayout()
        btn_set_alloy_layout.addWidget(self.btn_set_alloy)
        btn_set_alloy_layout.addWidget(self.lbl_alloy)
        btn_set_alloy_layout.addStretch()

        self.btn_set_std_1 = SmallButton('Set', self)
        self.btn_set_std_1.clicked.connect(self.ui_obj.set_std_1_trigger)
        btn_set_std_1_layout = QtWidgets.QHBoxLayout()
        btn_set_std_1_layout.addWidget(self.btn_set_std_1)
        btn_set_std_1_layout.addWidget(self.lbl_std_1)
        btn_set_std_1_layout.addStretch()

        self.btn_set_std_2 = SmallButton('Set', self)
        self.btn_set_std_2.clicked.connect(self.ui_obj.set_std_2_trigger)
        btn_set_std_2_layout = QtWidgets.QHBoxLayout()
        btn_set_std_2_layout.addWidget(self.btn_set_std_2)
        btn_set_std_2_layout.addWidget(self.lbl_std_2)
        btn_set_std_2_layout.addStretch()

        self.btn_set_std_3 = SmallButton('Set', self)
        self.btn_set_std_3.clicked.connect(self.ui_obj.set_std_3_trigger)
        btn_set_std_3_layout = QtWidgets.QHBoxLayout()
        btn_set_std_3_layout.addWidget(self.btn_set_std_3)
        btn_set_std_3_layout.addWidget(self.lbl_std_3)
        btn_set_std_3_layout.addStretch()

        self.btn_set_std_4 = SmallButton('Set', self)
        self.btn_set_std_4.clicked.connect(self.ui_obj.set_std_4_trigger)
        btn_set_std_4_layout = QtWidgets.QHBoxLayout()
        btn_set_std_4_layout.addWidget(self.btn_set_std_4)
        btn_set_std_4_layout.addWidget(self.lbl_std_4)
        btn_set_std_4_layout.addStretch()

        self.btn_set_std_5 = SmallButton('Set', self)
        self.btn_set_std_5.clicked.connect(self.ui_obj.set_std_5_trigger)
        btn_set_std_5_layout = QtWidgets.QHBoxLayout()
        btn_set_std_5_layout.addWidget(self.btn_set_std_5)
        btn_set_std_5_layout.addWidget(self.lbl_std_5)
        btn_set_std_5_layout.addStretch()

        self.btn_set_std_8 = SmallButton('Set', self)
        self.btn_set_std_8.clicked.connect(self.ui_obj.set_std_8_trigger)
        btn_set_std_8_layout = QtWidgets.QHBoxLayout()
        btn_set_std_8_layout.addWidget(self.btn_set_std_8)
        btn_set_std_8_layout.addWidget(self.lbl_std_8)
        btn_set_std_8_layout.addStretch()

        self.btn_set_cert_threshold = SmallButton('Set', self)
        self.btn_set_cert_threshold.clicked.connect(self.ui_obj.set_cert_threshold_trigger)
        btn_set_cert_threshold_layout = QtWidgets.QHBoxLayout()
        btn_set_cert_threshold_layout.addWidget(self.btn_set_cert_threshold)
        btn_set_cert_threshold_layout.addWidget(self.lbl_cert_threshold)
        btn_set_cert_threshold_layout.addStretch()

        self.btn_find_column = SmallButton('Set', self)
        self.btn_find_column.clicked.connect(self.ui_obj.find_column_trigger)
        btn_find_column_layout = QtWidgets.QHBoxLayout()
        btn_find_column_layout.addWidget(self.btn_find_column)
        btn_find_column_layout.addWidget(self.lbl_column_index)
        btn_find_column_layout.addStretch()

        self.btn_set_species = SmallButton('Set', self)
        self.btn_set_species.clicked.connect(self.ui_obj.set_species_trigger)
        btn_set_species_layout = QtWidgets.QHBoxLayout()
        btn_set_species_layout.addWidget(self.btn_set_species)
        btn_set_species_layout.addWidget(self.lbl_column_species)
        btn_set_species_layout.addStretch()

        self.btn_set_level = SmallButton('Set', self)
        self.btn_set_level.clicked.connect(self.ui_obj.set_level_trigger)
        btn_set_level_layout = QtWidgets.QHBoxLayout()
        btn_set_level_layout.addWidget(self.btn_set_level)
        btn_set_level_layout.addWidget(self.lbl_column_level)
        btn_set_level_layout.addStretch()

        # Move buttons

        self.btn_cancel_move = QtWidgets.QPushButton('Cancel', self)
        self.btn_cancel_move.clicked.connect(self.ui_obj.cancel_move_trigger)
        self.btn_cancel_move.setMaximumHeight(15)
        self.btn_cancel_move.setMaximumWidth(50)
        self.btn_cancel_move.setFont(self.font_tiny)
        self.btn_cancel_move.setDisabled(True)

        self.btn_set_move = QtWidgets.QPushButton('Accept', self)
        self.btn_set_move.clicked.connect(self.ui_obj.set_position_trigger)
        self.btn_set_move.setMaximumHeight(15)
        self.btn_set_move.setMaximumWidth(50)
        self.btn_set_move.setFont(self.font_tiny)
        self.btn_set_move.setDisabled(True)

        btn_move_control_layout = QtWidgets.QHBoxLayout()
        btn_move_control_layout.addWidget(self.chb_move)
        btn_move_control_layout.addWidget(self.btn_cancel_move)
        btn_move_control_layout.addWidget(self.btn_set_move)
        btn_move_control_layout.addStretch()

        # other buttons

        self.btn_show_stats = QtWidgets.QPushButton('Stats', self)
        self.btn_show_stats.clicked.connect(self.ui_obj.show_stats_trigger)
        self.btn_show_stats.setMaximumHeight(15)
        self.btn_show_stats.setMaximumWidth(50)
        self.btn_show_stats.setFont(self.font_tiny)

        self.btn_show_source = QtWidgets.QPushButton('Source', self)
        self.btn_show_source.clicked.connect(self.ui_obj.view_image_title_trigger)
        self.btn_show_source.setMaximumHeight(15)
        self.btn_show_source.setMaximumWidth(50)
        self.btn_show_source.setFont(self.font_tiny)

        self.btn_export = QtWidgets.QPushButton('Export', self)
        self.btn_export.clicked.connect(self.ui_obj.export_overlay_image_trigger)
        self.btn_export.setMaximumHeight(15)
        self.btn_export.setMaximumWidth(50)
        self.btn_export.setFont(self.font_tiny)

        self.btn_start_alg_1 = QtWidgets.QPushButton('Start', self)
        self.btn_start_alg_1.clicked.connect(self.ui_obj.continue_detection_trigger)
        self.btn_start_alg_1.setMaximumHeight(15)
        self.btn_start_alg_1.setMaximumWidth(50)
        self.btn_start_alg_1.setFont(self.font_tiny)

        self.btn_reset_alg_1 = QtWidgets.QPushButton('Reset', self)
        self.btn_reset_alg_1.clicked.connect(self.ui_obj.restart_detection_trigger)
        self.btn_reset_alg_1.setMaximumHeight(15)
        self.btn_reset_alg_1.setMaximumWidth(50)
        self.btn_reset_alg_1.setFont(self.font_tiny)

        self.btn_start_alg_2 = QtWidgets.QPushButton('Start', self)
        self.btn_start_alg_2.clicked.connect(self.ui_obj.continue_analysis_trigger)
        self.btn_start_alg_2.setMaximumHeight(15)
        self.btn_start_alg_2.setMaximumWidth(50)
        self.btn_start_alg_2.setFont(self.font_tiny)

        self.btn_reset_alg_2 = QtWidgets.QPushButton('Reset', self)
        self.btn_reset_alg_2.clicked.connect(self.ui_obj.restart_analysis_trigger)
        self.btn_reset_alg_2.setMaximumHeight(15)
        self.btn_reset_alg_2.setMaximumWidth(50)
        self.btn_reset_alg_2.setFont(self.font_tiny)

        self.btn_invert_lvl_alg_2 = QtWidgets.QPushButton('Invert lvl', self)
        self.btn_invert_lvl_alg_2.clicked.connect(self.ui_obj.invert_levels_trigger)
        self.btn_invert_lvl_alg_2.setMaximumHeight(15)
        self.btn_invert_lvl_alg_2.setMaximumWidth(50)
        self.btn_invert_lvl_alg_2.setFont(self.font_tiny)

        self.btn_delete = QtWidgets.QPushButton('Delete', self)
        self.btn_delete.clicked.connect(self.ui_obj.delete_trigger)
        self.btn_delete.setMaximumHeight(15)
        self.btn_delete.setMaximumWidth(50)
        self.btn_delete.setFont(self.font_tiny)

        self.btn_deselect = QtWidgets.QPushButton('Deselect', self)
        self.btn_deselect.clicked.connect(self.ui_obj.deselect_trigger)
        self.btn_deselect.setMaximumHeight(15)
        self.btn_deselect.setMaximumWidth(50)
        self.btn_deselect.setFont(self.font_tiny)

        self.btn_new = QtWidgets.QPushButton('New', self)
        self.btn_new.clicked.connect(self.ui_obj.new_column_trigger)
        self.btn_new.setMaximumHeight(15)
        self.btn_new.setMaximumWidth(50)
        self.btn_new.setFont(self.font_tiny)

        self.btn_set_style = QtWidgets.QPushButton('Set overlay style', self)
        self.btn_set_style.clicked.connect(self.ui_obj.set_style_trigger)
        self.btn_set_style.setMaximumHeight(20)
        self.btn_set_style.setMaximumWidth(200)
        self.btn_set_style.setFont(self.font_tiny)

        self.btn_set_indices = QtWidgets.QPushButton('Set neighbours', self)
        self.btn_set_indices.clicked.connect(self.ui_obj.set_indices_trigger)
        self.btn_set_indices.setMaximumHeight(20)
        self.btn_set_indices.setMaximumWidth(200)
        self.btn_set_indices.setFont(self.font_tiny)

        self.btn_set_indices_2 = QtWidgets.QPushButton('Set neighbours manually', self)
        self.btn_set_indices_2.clicked.connect(self.ui_obj.set_indices_2_trigger)
        self.btn_set_indices_2.setMaximumHeight(20)
        self.btn_set_indices_2.setMaximumWidth(200)
        self.btn_set_indices_2.setFont(self.font_tiny)

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
        btn_column_btns_layout.addStretch()

        btn_overlay_btns_layout = QtWidgets.QHBoxLayout()
        btn_overlay_btns_layout.addWidget(self.btn_set_style)
        btn_overlay_btns_layout.addStretch()

        # Info layout

        self.image_box = QtWidgets.QGroupBox('Image')
        self.image_box.setStyleSheet('QGroupBox { font-weight: bold; } ')
        self.image_box_empty = QtWidgets.QGroupBox('Image')
        self.image_box_empty.setStyleSheet('QGroupBox { font-weight: bold; } ')
        self.image_box_empty.hide()

        self.alg_1_box = QtWidgets.QGroupBox('Column detection')
        self.alg_1_box.setStyleSheet('QGroupBox { font-weight: bold; } ')
        self.alg_1_box_empty = QtWidgets.QGroupBox('Column detection')
        self.alg_1_box_empty.setStyleSheet('QGroupBox { font-weight: bold; } ')
        self.alg_1_box_empty.hide()

        self.alg_2_box = QtWidgets.QGroupBox('Column characterization')
        self.alg_2_box.setStyleSheet('QGroupBox { font-weight: bold; } ')
        self.alg_2_box_empty = QtWidgets.QGroupBox('Column characterization')
        self.alg_2_box_empty.setStyleSheet('QGroupBox { font-weight: bold; } ')
        self.alg_2_box_empty.hide()

        self.column_box = QtWidgets.QGroupBox('Selected column')
        self.column_box.setStyleSheet('QGroupBox { font-weight: bold; } ')
        self.column_box_empty = QtWidgets.QGroupBox('Selected column')
        self.column_box_empty.setStyleSheet('QGroupBox { font-weight: bold; } ')
        self.column_box_empty.hide()

        self.overlay_box = QtWidgets.QGroupBox('Overlay settings')
        self.overlay_box.setStyleSheet('QGroupBox { font-weight: bold; } ')
        self.overlay_box_empty = QtWidgets.QGroupBox('Overlay settings')
        self.overlay_box_empty.setStyleSheet('QGroupBox { font-weight: bold; } ')
        self.overlay_box_empty.hide()

        self.debug_box = QtWidgets.QGroupBox('Advanced debug mode')
        self.debug_box.setStyleSheet('QGroupBox { font-weight: bold; } ')

        self.info_display_layout_1 = QtWidgets.QVBoxLayout()
        self.info_display_layout_1.addLayout(btn_image_btns_layout)
        self.info_display_layout_1.addWidget(self.lbl_image_width)
        self.info_display_layout_1.addWidget(self.lbl_image_height)
        self.info_display_layout_1.addWidget(self.lbl_num_detected_columns)

        self.image_box.setLayout(self.info_display_layout_1)

        self.info_display_layout_2 = QtWidgets.QVBoxLayout()
        self.info_display_layout_2.addLayout(btn_alg_1_btns_layout)
        self.info_display_layout_2.addWidget(self.lbl_search_matrix_peak)
        self.info_display_layout_2.addWidget(self.lbl_atomic_radii)
        self.info_display_layout_2.addWidget(self.lbl_overhead_radii)
        self.info_display_layout_2.addLayout(btn_set_scale_layout)
        self.info_display_layout_2.addLayout(btn_set_threshold_layout)
        self.info_display_layout_2.addLayout(btn_set_search_size_layout)

        self.alg_1_box.setLayout(self.info_display_layout_2)

        self.info_display_layout_3 = QtWidgets.QVBoxLayout()
        self.info_display_layout_3.addLayout(btn_alg_2_btns_layout)
        self.info_display_layout_3.addLayout(btn_set_scale_layout)
        self.info_display_layout_3.addLayout(btn_set_alloy_layout)

        self.alg_2_box.setLayout(self.info_display_layout_3)

        self.info_display_layout_4 = QtWidgets.QVBoxLayout()
        self.info_display_layout_4.addLayout(btn_column_btns_layout)
        self.info_display_layout_4.addLayout(btn_find_column_layout)
        self.info_display_layout_4.addLayout(btn_set_species_layout)
        self.info_display_layout_4.addLayout(btn_set_level_layout)
        self.info_display_layout_4.addWidget(self.lbl_column_x_pos)
        self.info_display_layout_4.addWidget(self.lbl_column_y_pos)
        self.info_display_layout_4.addWidget(self.lbl_column_peak_gamma)
        self.info_display_layout_4.addWidget(self.lbl_column_avg_gamma)
        self.info_display_layout_4.addWidget(self.lbl_confidence)
        self.info_display_layout_4.addWidget(self.lbl_prob_vector)
        self.info_display_layout_4.addLayout(self.probGraphicLayout)
        self.info_display_layout_4.addWidget(self.chb_precipitate_column)
        self.info_display_layout_4.addWidget(self.chb_show)
        self.info_display_layout_4.addLayout(btn_move_control_layout)
        self.info_display_layout_4.addWidget(self.lbl_neighbours)

        self.column_box.setLayout(self.info_display_layout_4)

        self.info_display_layout_5 = QtWidgets.QVBoxLayout()
        self.info_display_layout_5.addLayout(btn_overlay_btns_layout)
        self.info_display_layout_5.addLayout(overlay_layout)

        self.overlay_box.setLayout(self.info_display_layout_5)

        self.info_display_layout_6 = QtWidgets.QVBoxLayout()
        self.info_display_layout_6.addLayout(btn_debug_btns_layout)
        self.info_display_layout_6.addLayout(btn_set_std_1_layout)
        self.info_display_layout_6.addLayout(btn_set_std_2_layout)
        self.info_display_layout_6.addLayout(btn_set_std_3_layout)
        self.info_display_layout_6.addLayout(btn_set_std_4_layout)
        self.info_display_layout_6.addLayout(btn_set_std_5_layout)
        self.info_display_layout_6.addLayout(btn_set_std_8_layout)
        self.info_display_layout_6.addLayout(btn_set_cert_threshold_layout)

        self.debug_box.setLayout(self.info_display_layout_6)

        self.info_display_layout = QtWidgets.QVBoxLayout()
        self.info_display_layout.addWidget(self.image_box)
        self.info_display_layout.addWidget(self.image_box_empty)
        self.info_display_layout.addWidget(self.debug_box)
        self.info_display_layout.addWidget(self.alg_1_box)
        self.info_display_layout.addWidget(self.alg_1_box_empty)
        self.info_display_layout.addWidget(self.alg_2_box)
        self.info_display_layout.addWidget(self.alg_2_box_empty)
        self.info_display_layout.addWidget(self.column_box)
        self.info_display_layout.addWidget(self.column_box_empty)
        self.info_display_layout.addWidget(self.overlay_box)
        self.info_display_layout.addWidget(self.overlay_box_empty)
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

        if self.ui_obj.project_loaded and not self.ui_obj.selected_column == -1:

            box_width = 15
            box_seperation = 10
            box_displacement = 25

            si_box_height = int(100 * self.ui_obj.project_instance.columns[self.ui_obj.selected_column].prob_vector[0])
            cu_box_height = int(100 * self.ui_obj.project_instance.columns[self.ui_obj.selected_column].prob_vector[1])
            zn_box_height = int(100 * self.ui_obj.project_instance.columns[self.ui_obj.selected_column].prob_vector[2])
            al_box_height = int(100 * self.ui_obj.project_instance.columns[self.ui_obj.selected_column].prob_vector[3])
            ag_box_height = int(100 * self.ui_obj.project_instance.columns[self.ui_obj.selected_column].prob_vector[4])
            mg_box_height = int(100 * self.ui_obj.project_instance.columns[self.ui_obj.selected_column].prob_vector[5])
            un_box_height = int(100 * self.ui_obj.project_instance.columns[self.ui_obj.selected_column].prob_vector[6])

        else:

            box_width = 15
            box_seperation = 10
            box_displacement = 25

            si_box_height = 0
            cu_box_height = 0
            zn_box_height = 0
            al_box_height = 0
            ag_box_height = 0
            mg_box_height = 0
            un_box_height = 0

        probGraphicScene = QtWidgets.QGraphicsScene()

        box = QtWidgets.QGraphicsRectItem(0, -10, self.width - 10, self.height - 10)
        box.setPen(self.ui_obj.black_pen)
        box.hide()

        x_box = box_seperation + 0 * box_displacement
        box_height = si_box_height
        si_box = QtWidgets.QGraphicsRectItem(0, 0, box_width, box_height)
        si_box.setBrush(self.ui_obj.brush_si)
        si_box.setX(x_box)
        si_box.setY(100 - box_height)
        si_text = QtWidgets.QGraphicsSimpleTextItem()
        si_text.setText('Si')
        si_text.setFont(self.font_tiny)
        si_text.setX(x_box + 3)
        si_text.setY(100 + 4)
        si_number = QtWidgets.QGraphicsSimpleTextItem()
        si_number.setText(str(box_height / 100))
        si_number.setFont(self.font_tiny)
        si_number.setX(x_box - 1)
        si_number.setY(100 - box_height - 10)

        x_box = box_seperation + 1 * box_displacement
        box_height = cu_box_height
        cu_box = QtWidgets.QGraphicsRectItem(0, 0, box_width, box_height)
        cu_box.setBrush(self.ui_obj.brush_cu)
        cu_box.setX(x_box)
        cu_box.setY(100 - box_height)
        cu_text = QtWidgets.QGraphicsSimpleTextItem()
        cu_text.setText('Cu')
        cu_text.setFont(self.font_tiny)
        cu_text.setX(x_box + 2)
        cu_text.setY(100 + 4)
        cu_number = QtWidgets.QGraphicsSimpleTextItem()
        cu_number.setText(str(box_height / 100))
        cu_number.setFont(self.font_tiny)
        cu_number.setX(x_box - 1)
        cu_number.setY(100 - box_height - 10)

        x_box = box_seperation + 2 * box_displacement
        box_height = zn_box_height
        zn_box = QtWidgets.QGraphicsRectItem(0, 0, box_width, box_height)
        zn_box.setBrush(self.ui_obj.brush_zn)
        zn_box.setX(x_box)
        zn_box.setY(100 - box_height)
        zn_text = QtWidgets.QGraphicsSimpleTextItem()
        zn_text.setText('Zn')
        zn_text.setFont(self.font_tiny)
        zn_text.setX(x_box + 2)
        zn_text.setY(100 + 4)
        zn_number = QtWidgets.QGraphicsSimpleTextItem()
        zn_number.setText(str(box_height / 100))
        zn_number.setFont(self.font_tiny)
        zn_number.setX(x_box - 1)
        zn_number.setY(100 - box_height - 10)

        x_box = box_seperation + 3 * box_displacement
        box_height = al_box_height
        al_box = QtWidgets.QGraphicsRectItem(0, 0, box_width, box_height)
        al_box.setBrush(self.ui_obj.brush_al)
        al_box.setX(x_box)
        al_box.setY(100 - box_height)
        al_text = QtWidgets.QGraphicsSimpleTextItem()
        al_text.setText('Al')
        al_text.setFont(self.font_tiny)
        al_text.setX(x_box + 2)
        al_text.setY(100 + 4)
        al_number = QtWidgets.QGraphicsSimpleTextItem()
        al_number.setText(str(box_height / 100))
        al_number.setFont(self.font_tiny)
        al_number.setX(x_box - 1)
        al_number.setY(100 - box_height - 10)

        x_box = box_seperation + 4 * box_displacement
        box_height = ag_box_height
        ag_box = QtWidgets.QGraphicsRectItem(0, 0, box_width, box_height)
        ag_box.setBrush(self.ui_obj.brush_ag)
        ag_box.setX(x_box)
        ag_box.setY(100 - box_height)
        ag_text = QtWidgets.QGraphicsSimpleTextItem()
        ag_text.setText('Ag')
        ag_text.setFont(self.font_tiny)
        ag_text.setX(x_box + 2)
        ag_text.setY(100 + 4)
        ag_number = QtWidgets.QGraphicsSimpleTextItem()
        ag_number.setText(str(box_height / 100))
        ag_number.setFont(self.font_tiny)
        ag_number.setX(x_box - 1)
        ag_number.setY(100 - box_height - 10)

        x_box = box_seperation + 5 * box_displacement
        box_height = mg_box_height
        mg_box = QtWidgets.QGraphicsRectItem(0, 0, box_width, box_height)
        mg_box.setBrush(self.ui_obj.brush_mg)
        mg_box.setX(x_box)
        mg_box.setY(100 - box_height)
        mg_text = QtWidgets.QGraphicsSimpleTextItem()
        mg_text.setText('Mg')
        mg_text.setFont(self.font_tiny)
        mg_text.setX(x_box + 2)
        mg_text.setY(100 + 4)
        mg_number = QtWidgets.QGraphicsSimpleTextItem()
        mg_number.setText(str(box_height / 100))
        mg_number.setFont(self.font_tiny)
        mg_number.setX(x_box - 1)
        mg_number.setY(100 - box_height - 10)

        x_box = box_seperation + 6 * box_displacement
        box_height = un_box_height
        un_box = QtWidgets.QGraphicsRectItem(0, 0, box_width, box_height)
        un_box.setBrush(self.ui_obj.brush_un)
        un_box.setX(x_box)
        un_box.setY(100 - box_height)
        un_text = QtWidgets.QGraphicsSimpleTextItem()
        un_text.setText('Un')
        un_text.setFont(self.font_tiny)
        un_text.setX(x_box + 2)
        un_text.setY(100 + 4)
        un_number = QtWidgets.QGraphicsSimpleTextItem()
        un_number.setText(str(box_height / 100))
        un_number.setFont(self.font_tiny)
        un_number.setX(x_box - 1)
        un_number.setY(100 - box_height - 10)

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

    def empty_display(self):

        self.ui_obj.graphic = QtGui.QPixmap('Images\\no_image.png')
        self.ui_obj.graphicScene_1 = QtWidgets.QGraphicsScene()
        self.ui_obj.graphicScene_1.addPixmap(self.ui_obj.graphic)
        self.ui_obj.graphicScene_2 = QtWidgets.QGraphicsScene()
        self.ui_obj.graphicScene_2.addPixmap(self.ui_obj.graphic)
        self.ui_obj.graphicScene_3 = QtWidgets.QGraphicsScene()
        self.ui_obj.graphicScene_3.addPixmap(self.ui_obj.graphic)
        self.ui_obj.graphicScene_4 = QtWidgets.QGraphicsScene()
        self.ui_obj.graphicScene_4.addPixmap(self.ui_obj.graphic)
        self.ui_obj.graphicScene_5 = QtWidgets.QGraphicsScene()
        self.ui_obj.graphicScene_5.addPixmap(self.ui_obj.graphic)
        self.ui_obj.graphicScene_6 = QtWidgets.QGraphicsScene()
        self.ui_obj.graphicScene_6.addPixmap(self.ui_obj.graphic)
        self.ui_obj.graphicScene_7 = QtWidgets.QGraphicsScene()
        self.ui_obj.graphicScene_7.addPixmap(self.ui_obj.graphic)
        self.ui_obj.graphicsView_1.setScene(self.ui_obj.graphicScene_1)
        self.ui_obj.graphicsView_2.setScene(self.ui_obj.graphicScene_2)
        self.ui_obj.graphicsView_3.setScene(self.ui_obj.graphicScene_3)
        self.ui_obj.graphicsView_4.setScene(self.ui_obj.graphicScene_4)
        self.ui_obj.graphicsView_5.setScene(self.ui_obj.graphicScene_5)
        self.ui_obj.graphicsView_6.setScene(self.ui_obj.graphicScene_6)
        self.ui_obj.graphicsView_7.setScene(self.ui_obj.graphicScene_7)

        self.ui_obj.project_instance = Core.SuchSoftware('empty')
        self.ui_obj.project_loaded = False
        self.ui_obj.savefile = None
        self.ui_obj.selected_column = -1
        self.ui_obj.previous_pos_obj = None
        self.ui_obj.previous_overlay_obj = None
        dummy_instance = InteractivePosColumn(0, 0, 0, 0)
        self.ui_obj.pos_objects = np.ndarray([1], dtype=type(dummy_instance))
        dummy_instance = InteractiveOverlayColumn(0, 0, 0, 0)
        self.ui_obj.overlay_objects = np.ndarray([1], dtype=type(dummy_instance))
        self.ui_obj.neighbour_line_objects = np.ndarray([1], dtype=type(QtWidgets.QGraphicsLineItem()))
        self.ui_obj.neighbour_line_objects[0] = QtWidgets.QGraphicsLineItem(1.0, 2.0, 3.0, 4.0)

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
        self.lbl_std_1.setText('Standard deviation 1: ')
        self.lbl_std_2.setText('Standard deviation 2: ')
        self.lbl_std_3.setText('Standard deviation 3: ')
        self.lbl_std_4.setText('Standard deviation 4: ')
        self.lbl_std_5.setText('Standard deviation 5: ')
        self.lbl_std_8.setText('Standard deviation 8: ')
        self.lbl_cert_threshold.setText('Certainty threshold: ')

        self.ui_obj.deselect_trigger()

        self.chb_raw_image.setChecked(True)
        self.chb_structures.setChecked(True)
        self.chb_si_network.setChecked(False)
        self.chb_mg_network.setChecked(False)
        self.chb_al_network.setChecked(False)
        self.chb_boarders.setChecked(False)
        self.chb_columns.setChecked(True)
        self.chb_legend.setChecked(True)
        self.chb_scalebar.setChecked(False)

        self.ui_obj.statusBar().showMessage('Ready')


class MenuBar:

    def __init__(self, bar_obj, ui_obj):

        self.bar_obj = bar_obj
        self.ui_obj = ui_obj

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
        test_consistency_action = QtWidgets.QAction('Reset levels', self.ui_obj)
        invert_precipitate_levels_action = QtWidgets.QAction('Invert precipitate levels', self.ui_obj)
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
        debug.addAction(display_deviations_action)
        debug.addAction(test_consistency_action)
        debug.addAction(invert_precipitate_levels_action)
        # - Help
        help.addAction(there_is_no_help_action)

        # Events
        # - file
        new_action.triggered.connect(self.ui_obj.new_trigger)
        open_action.triggered.connect(self.ui_obj.open_trigger)
        save_action.triggered.connect(self.ui_obj.save_trigger)
        close_action.triggered.connect(self.ui_obj.close_trigger)
        exit_action.triggered.connect(self.ui_obj.exit_trigger)
        # - edit
        # - view
        view_image_title_action.triggered.connect(self.ui_obj.view_image_title_trigger)
        show_stats_action.triggered.connect(self.ui_obj.show_stats_trigger)
        update_display_action.triggered.connect(self.ui_obj.update_display)
        toggle_image_control_action.triggered.connect(self.ui_obj.toggle_image_control_trigger)
        toggle_alg_1_control_action.triggered.connect(self.ui_obj.toggle_alg_1_control_trigger)
        toggle_alg_2_control_action.triggered.connect(self.ui_obj.toggle_alg_2_control_trigger)
        toggle_column_control_action.triggered.connect(self.ui_obj.toggle_column_control_trigger)
        toggle_overlay_control_action.triggered.connect(self.ui_obj.toggle_overlay_control_trigger)
        # - Process
        image_correction_action.triggered.connect(self.ui_obj.image_correction_trigger)
        image_filter_action.triggered.connect(self.ui_obj.image_filter_trigger)
        image_adjustments_action.triggered.connect(self.ui_obj.image_adjustments_trigger)
        continue_detection_action.triggered.connect(self.ui_obj.continue_detection_trigger)
        restart_detection_action.triggered.connect(self.ui_obj.restart_detection_trigger)
        continue_analysis_action.triggered.connect(self.ui_obj.continue_analysis_trigger)
        restart_analysis_action.triggered.connect(self.ui_obj.restart_analysis_trigger)
        # - Export
        export_data_action.triggered.connect(self.ui_obj.export_data_trigger)
        export_raw_image_action.triggered.connect(self.ui_obj.export_raw_image_trigger)
        export_column_position_image_action.triggered.connect(self.ui_obj.export_column_position_image_trigger)
        export_overlay_image_action.triggered.connect(self.ui_obj.export_overlay_image_trigger)
        export_atomic_graph_action.triggered.connect(self.ui_obj.export_atomic_graph_trigger)
        # - debug
        advanced_debug_mode_action.triggered.connect(self.ui_obj.toggle_debug_mode_trigger)
        add_mark_action.triggered.connect(self.ui_obj.add_mark_trigger)
        reset_flags_action.triggered.connect(self.ui_obj.clear_flags_trigger)
        set_control_file_action.triggered.connect(self.ui_obj.set_control_file_trigger)
        display_deviations_action.triggered.connect(self.ui_obj.display_deviations_trigger)
        test_consistency_action.triggered.connect(self.ui_obj.test_consistency_trigger)
        invert_precipitate_levels_action.triggered.connect(self.ui_obj.invert_precipitate_columns_trigger)
        # - hjelp
        there_is_no_help_action.triggered.connect(self.ui_obj.there_is_no_help_trigger)

