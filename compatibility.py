import old_core_reference as Core
import core
import graph


class Upgrade:

    def __init__(self, filename, GUI_version, core_version, report):

        self.filename = filename
        self.GUI_version = GUI_version
        self.core_version = core_version
        self.old_instance = None
        self.new_instance = None
        self.file_version = None
        self.report = report

    def load(self):

        try:
            self.old_instance = Core.SuchSoftware.load(self.filename)
        except:
            print('Compatibility script failed!')
        else:
            self.upgrade()

        print('Ready to send')
        return self.new_instance

    def upgrade(self):

        self.new_instance = core.SuchSoftware('empty')

        # All the attributes that are unchanged:
        self.new_instance.filename_full = self.old_instance.filename_full
        self.new_instance.im_mat = self.old_instance.im_mat
        self.new_instance.scale = self.old_instance.scale
        self.new_instance.im_height = self.old_instance.N
        self.new_instance.im_width = self.old_instance.M
        self.new_instance.search_mat = self.old_instance.search_mat
        self.new_instance.column_centre_mat = self.old_instance.column_centre_mat
        self.new_instance.column_circumference_mat = self.old_instance.column_circumference_mat
        self.new_instance.fft_im_mat = self.old_instance.fft_im_mat
        self.new_instance.num_columns = self.old_instance.num_columns
        self.new_instance.alloy = self.old_instance.alloy
        self.new_instance.alloy_mat = self.old_instance.alloy_mat
        self.new_instance.certainty_threshold = self.old_instance.certainty_threshold
        self.new_instance.r = self.old_instance.r


        # New varibles:


        self.new_instance.graph = graph.AtomicGraph()
        for i in range(0, self.new_instance.num_columns):
            column = self.old_instance.columns[i]
            vertex = graph.Vertex(i, column.x, column.y, self.new_instance.r, column.peak_gamma, column.avg_gamma,
                                  self.new_instance.alloy_mat, num_selections=self.new_instance.num_selections,
                                  species_strings=self.new_instance.species_strings,
                                  certainty_threshold=self.new_instance.certainty_threshold)
            vertex.prob_vector = column.prob_vector
            vertex.neighbour_indices = column.neighbour_indices

            self.new_instance.graph.add_vertex(vertex)
