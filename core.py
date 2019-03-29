# This file contains the SuchSoftware class that is the algorithms.
import numpy as np
import dm3_lib as dm3
import mat_op


class SuchSoftware:

    # Number of elements in the probability vectors
    num_selections = 7

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
    atomic_radii = (si_radii, cu_radii, zn_radii, al_radii, ag_radii, mg_radii, un_radii)

    # Relative mean peak intensities for the different implemented alloys:
    intensities_0 = [0.44, 0.88, 0.00, 0.40, 0.00, 0.33, 0.00]
    intensities_1 = [0.70, 0.00, 0.00, 0.67, 0.00, 0.49, 0.00]

    # Indexable list
    intensities = [intensities_0, intensities_1]

    # Constructor
    def __init__(self, filename_full):

        self.filename_full = filename_full
        self.im_mat = None
        self.scale = 0
        self.im_height = 0
        self.im_width = 0




        # Data matrices: These hold much of the information gathered by the different algorithms
        self.search_mat = self.im_mat
        self.column_centre_mat = np.zeros((self.N, self.M, 2), dtype=type(self.im_mat))
        self.column_circumference_mat = np.zeros((self.N, self.M), dtype=type(self.im_mat))
        self.fft_im_mat = mat_op.gen_fft(self.im_mat)
        self.precipitate_boarder = np.ndarray([1], dtype=int)
        self.structures_eye = np.ndarray([9], dtype=int)
        self.the_eyes = np.ndarray([1], dtype=type(self.structures_eye))
        self.structures_flower = np.ndarray([13], dtype=int)
        self.the_flowers = np.ndarray([1], dtype=type(self.structures_flower))

        # Alloy info: This vector is used to multiply away elements in the AtomicColumn.prob_vector that are not in
        # the alloy being studied. Currently supported alloys are:
        # self.alloy = alloy
        # 0 = Al-Si-Mg-(Cu)
        # 1 = Al-Si-Mg
        self.alloy = 0
        self.alloy_mat = np.ndarray([SuchSoftware.num_selections], dtype=int)
        self.set_alloy_mat()

        # Counting and statistical variables
        self.num_columns = 0
        self.num_precipitate_columns = 0

        self.num_inconsistencies = 0
        self.num_popular = 0
        self.num_unpopular = 0
        self.chi = 0.0

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
        self.r = int(100 / self.scale)
        self.certainty_threshold = 0.8
        self.overhead = int(6 * (self.r / 10))

        self.dist_1_std = 20.0
        self.dist_2_std = 18.0
        self.dist_3_std = 0.4
        self.dist_4_std = 0.34
        self.dist_5_std = 0.48
        self.dist_8_std = 1.2

    def load_image(self):

        dm3f = dm3.DM3(self.filename_full)
        self.im_mat = dm3f.imagedata
        (self.scale, junk) = dm3f.pxsize
        self.scale = 1000 * self.scale
        self.im_mat = mat_op.normalize_static(self.im_mat)
        (self.im_height, self.im_width) = self.im_mat.shape

    def set_alloy_mat(self, alloy=0):

