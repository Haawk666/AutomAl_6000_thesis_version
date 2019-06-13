from PyQt5 import QtWidgets
import sys
import core_
import core
import graph


class Ui(QtWidgets.QMainWindow):

    def __init__(self, filename=None):
        super().__init__()

        self.old_obj = None
        self.intermediate_obj = None
        self.new_obj = None
        self.filename = None

        self.setWindowTitle(
            'Convert files to new version')
        self.resize(500, 100)
        self.move(50, 30)

        self.btn_open = QtWidgets.QPushButton('Choose file', self)
        self.btn_open.clicked.connect(self.open_trigger)

        if filename is not None:
            self.load()
        else:
            self.show()

    def open_trigger(self):
        self.filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '')
        if self.filename[0]:
            self.load()
            print('Converted file successfully! {}'.format(self.filename[0]))
        else:
            print('error')

    def load(self):
        self.old_obj = core_.SuchSoftware.load(self.filename[0])
        self.new_obj = core.SuchSoftware('empty')
        self.convert_to_module()
        self.new_obj.save(self.filename[0])

    def convert_to_module(self):
        # All the attributes that are unchanged:
        self.new_obj.filename_full = self.old_obj.filename_full
        self.new_obj.im_mat = self.old_obj.im_mat
        self.new_obj.scale = self.old_obj.scale
        self.new_obj.im_height = self.old_obj.im_height
        self.new_obj.im_width = self.old_obj.im_width
        self.new_obj.search_mat = self.old_obj.search_mat
        self.new_obj.column_centre_mat = self.old_obj.column_centre_mat
        self.new_obj.column_circumference_mat = self.old_obj.column_circumference_mat
        self.new_obj.fft_im_mat = self.old_obj.fft_im_mat
        self.new_obj.num_columns = self.old_obj.num_columns
        self.new_obj.alloy = self.old_obj.alloy
        self.new_obj.alloy_mat = self.old_obj.alloy_mat
        self.new_obj.certainty_threshold = self.old_obj.certainty_threshold
        self.new_obj.r = self.old_obj.r
        self.new_obj.graph = self.old_obj.graph


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    program = Ui()
    sys.exit(app.exec_())

