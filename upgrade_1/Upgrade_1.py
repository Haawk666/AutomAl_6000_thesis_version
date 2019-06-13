from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import Core
import core_
import graph


class Ui(QtWidgets.QMainWindow):

    def __init__(self, filename=None):
        super().__init__()

        self.old_obj = None
        self.intermediate_obj = None
        self.new_obj = None
        self.filename = filename

        self.setWindowTitle(
            'Convert files to new version')
        self.resize(500, 100)
        self.move(50, 30)

        self.btn_open = QtWidgets.QPushButton('Choose file', self)
        self.btn_open.clicked.connect(self.open_trigger)

        self.show()

    def open_trigger(self):
        self.filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '')
        if self.filename[0]:
            self.old_obj = Core.SuchSoftware.load(self.filename[0])
            self.new_obj = core_.SuchSoftware('empty')
            self.convert_to_0_0_0()
            self.new_obj.save(self.filename[0])
            print('Converted file successfully! {}'.format(self.filename[0]))
        else:
            print('error')

    def convert_to_0_0_0(self):
        # All the attributes that are unchanged:
        self.new_obj.filename_full = self.old_obj.filename_full
        self.new_obj.im_mat = self.old_obj.im_mat
        self.new_obj.scale = self.old_obj.scale
        self.new_obj.im_height = self.old_obj.N
        self.new_obj.im_width = self.old_obj.M
        self.new_obj.search_mat = self.old_obj.search_mat
        self.new_obj.column_centre_mat = self.old_obj.column_centre_mat
        self.new_obj.column_circumference_mat = self.old_obj.column_circumference_mat
        self.new_obj.fft_im_mat = self.old_obj.fft_im_mat
        self.new_obj.num_columns = self.old_obj.num_columns
        self.new_obj.alloy = self.old_obj.alloy
        self.new_obj.alloy_mat = self.old_obj.alloy_mat
        self.new_obj.certainty_threshold = self.old_obj.certainty_threshold
        self.new_obj.r = self.old_obj.r

        # New varibles:

        self.new_obj.graph = graph.AtomicGraph()
        for i in range(0, self.new_obj.num_columns):
            column = self.old_obj.columns[i]
            vertex = graph.Vertex(i, column.x, column.y, self.new_obj.r, column.peak_gamma, column.avg_gamma,
                                  self.new_obj.alloy_mat, num_selections=self.new_obj.num_selections,
                                  species_strings=self.new_obj.species_strings,
                                  certainty_threshold=self.new_obj.certainty_threshold)
            vertex.prob_vector = column.prob_vector
            if column.neighbour_indices is None:
                vertex.neighbour_indices = []
            else:
                vertex.neighbour_indices = column.neighbour_indices
            vertex.h_index = column.h_index
            vertex.level = column.level
            vertex.confidence = column.confidence
            vertex.atomic_species = column.atomic_species

            self.new_obj.graph.add_vertex(vertex)

        self.new_obj.graph.redraw_edges()

    def convert_to_module(self):
        pass


app = QtWidgets.QApplication(sys.argv)
program = Ui()
sys.exit(app.exec_())

