import numpy as np
from copy import deepcopy
import pickle
import dm3_lib as dm3
import Utilities
import mat_op
import graph


class SuchSoftware:
    """Structure data from a HAADF-STEM .dm3 image and perform analysis."""

    # Class constants

    # Number of elements in the probability vectors
    selections = 7

    # Al lattice constant in picometers
    al_lattice_const = 404.95

    # Atomic "hard sphere" radii in pm:
    si_radii = 117.5
    cu_radii = 127.81
    zn_radii = 133.25
    al_radii = 143
    ag_radii = 144.5
    mg_radii = 160
    un_radii = 200

    # Indexable list
    radii = (si_radii, cu_radii, zn_radii, al_radii, ag_radii, mg_radii, un_radii)

    # Relative mean peak intensities for the different implemented alloys:
    intensities_0 = [0.44, 0.88, 0.00, 0.40, 0.00, 0.33, 0.00]
    intensities_1 = [0.70, 0.00, 0.00, 0.67, 0.00, 0.49, 0.00]

    # Indexable list
    intensities = [intensities_0, intensities_1]

    # Constructor
    def __init__(self, filename_full):

        # Import image data. The dm3 type object is discarded after image matrix is stored because holding the dm3
        # in memory will clog the system buffer and mess up the pickle process
        self.filename_full = filename_full

        self.im_mat = None
        self.scale = None
        self.r = 0
        self.over_r = 0
        self.im_height = None
        self.im_width = None

        if not filename_full == 'empty':
            self.load_image()

        # Data matrices: These hold much of the information gathered by the different algorithms
        self.search_mat = self.im_mat
        self.column_centre_mat = np.zeros((self.N, self.M, 2), dtype=type(self.im_mat))
        self.column_circumference_mat = np.zeros((self.N, self.M), dtype=type(self.im_mat))
        self.fft_im_mat = mat_op.gen_fft(self.im_mat)

        # Alloy info: This vector is used to multiply away elements in the AtomicColumn.prob_vector that are not in
        # the alloy being studied. Currently supported alloys are:
        #
        # 0 = Al-Si-Mg-(Cu)
        # 1 = Al-Si-Mg
        self.alloy = 0
        self.alloy_mat = np.ndarray([SuchSoftware.selections], dtype=int)
        self.set_alloy_mat()

        # Counting and statistical variables
        self.num_columns = 0
        self.num_precipitate_columns = 0

        self.num_al = 0
        self.num_mg = 0
        self.num_si = 0
        self.num_cu = 0
        self.num_ag = 0
        self.num_zn = 0
        self.num_un = 0

        self.num_precipitate_si = 0
        self.num_precipitate_cu = 0
        self.num_precipitate_zn = 0
        self.num_precipitate_al = 0
        self.num_precipitate_ag = 0
        self.num_precipitate_mg = 0
        self.num_precipitate_un = 0

        self.boarder_size = 0
        self.num_eyes = 0
        self.num_flowers = 0

        self.avg_peak_gamma = 0.0
        self.avg_avg_gamma = 0.0

        self.avg_si_peak_gamma = 0.0
        self.avg_cu_peak_gamma = 0.0
        self.avg_zn_peak_gamma = 0.0
        self.avg_al_peak_gamma = 0.0
        self.avg_ag_peak_gamma = 0.0
        self.avg_mg_peak_gamma = 0.0
        self.avg_un_peak_gamma = 0.0

        self.avg_si_avg_gamma = 0.0
        self.avg_cu_avg_gamma = 0.0
        self.avg_zn_avg_gamma = 0.0
        self.avg_al_avg_gamma = 0.0
        self.avg_ag_avg_gamma = 0.0
        self.avg_mg_avg_gamma = 0.0
        self.avg_un_avg_gamma = 0.0

        self.number_percentage_si = 0.0
        self.number_percentage_cu = 0.0
        self.number_percentage_zn = 0.0
        self.number_percentage_al = 0.0
        self.number_percentage_ag = 0.0
        self.number_percentage_mg = 0.0
        self.number_percentage_un = 0.0

        self.precipitate_number_percentage_si = 0.0
        self.precipitate_number_percentage_cu = 0.0
        self.precipitate_number_percentage_zn = 0.0
        self.precipitate_number_percentage_al = 0.0
        self.precipitate_number_percentage_ag = 0.0
        self.precipitate_number_percentage_mg = 0.0
        self.precipitate_number_percentage_un = 0.0

        self.display_stats_string = 'Empty'
        self.export_data_string = ' '

        # Parameter variables. these are parameters of the algorithms. See the documentation.
        self.threshold = 0.2586
        self.search_size = 1
        self.certainty_threshold = 0.8

        self.dist_1_std = 20.0
        self.dist_2_std = 18.0
        self.dist_3_std = 0.4
        self.dist_4_std = 0.34
        self.dist_5_std = 0.48
        self.dist_8_std = 1.2

        # The graph object:
        self.graph = graph.AtomicGraph()

    # Support functions (private, not intended to be called by external objects)
    def set_alloy_mat(self):

        if self.alloy == 0:

            for x in range(0, SuchSoftware.num_poss):
                if x == 2 or x == 4 or x == 6:
                    self.alloy_mat[x] = 0
                else:
                    self.alloy_mat[x] = 1

        elif self.alloy == 1:

            for x in range(0, SuchSoftware.num_poss):
                if x == 2 or x == 4 or x == 1 or x == 6:
                    self.alloy_mat[x] = 0
                else:
                    self.alloy_mat[x] = 1

        else:

            print('Not a supported alloy number!')

    def load_image(self):
        dm3f = dm3.DM3(self.filename_full)
        self.im_mat = dm3f.imagedata
        (self.scale, junk) = dm3f.pxsize
        self.scale = 1000 * self.scale
        self.im_mat = mat_op.normalize_static(self.im_mat)
        (self.im_height, self.im_width) = self.im_mat.shape
        self.r = int(100 / self.scale)
        self.over_r = int(6 * (self.r / 10))

    # Importing and exporting instances
    def save(self, filename_full):
        with open(filename_full, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename_full):
        with open(filename_full, 'rb') as f:
            obj = pickle.load(f)
            return obj

    # Algorithms
    def column_finder(self, search_type='s'):
        pass

    def column_analyser(self, i, search_type='s'):
        pass


