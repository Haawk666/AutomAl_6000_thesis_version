# Core.py
# -------------------------------------------
# This file contains the main functionality and data structures. Data is stored in the classes AtomicColumn and
# SuchSoftware. The algorithms are methods of the SuchSoftware class.

import numpy as np
from copy import deepcopy
import pickle
import dm3_lib as dm3
import Utilities
import mat_op


class AtomicColumn:

    # Constructor
    def __init__(self, i, x, y, r, peak_gamma, avg_gamma, num_poss):

        self.i = i
        self.x = x
        self.y = y
        self.r = r
        self.peak_gamma = peak_gamma
        self.avg_gamma = avg_gamma

        self.level = 0
        self.atomic_species = 'Un'
        self.h_index = 6
        self.confidence = 0.0
        self.is_in_precipitate = False
        self.set_by_user = False
        self.flag_1 = False
        self.flag_2 = False
        self.flag_3 = False
        self.flag_4 = False
        self.is_unpopular = False
        self.is_popular = False
        self.is_edge_column = False
        self.show_in_overlay = True
        self.prob_vector = np.ndarray([num_poss], dtype=np.float64)
        self.neighbour_indices = None

        # The prob_vector is ordered to represent the elements in order of their radius:

        # Si
        # Cu
        # Zm
        # Al
        # Ag
        # Mg
        # Un

    def n(self):

        n = 3

        if self.h_index == 0 or self.h_index == 1:
            n = 3
        elif self.h_index == 3:
            n = 4
        elif self.h_index == 5:
            n = 5

        return n


class SuchSoftware:
    """Structure data from a HAADF-STEM .dm3 image and perform analysis."""

    # Class constants

    # Number of elements in the probability vectors
    num_poss = 7

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

        # Codeword 'empty' is used to instantiate placeholder object
        if filename_full == 'empty':
            pass
        else:

            # Import image data. The dm3 type object is discarded after image matrix is stored because holding the dm3
            # in memory will clog the system buffer and mess up the pickle process
            self.filename_full = filename_full
            dm3f = dm3.DM3(self.filename_full)
            self.im_mat = dm3f.imagedata
            (self.scale, junk) = dm3f.pxsize
            self.scale = 1000 * self.scale
            self.im_mat = mat_op.normalize_static(self.im_mat)
            (self.N, self.M) = self.im_mat.shape

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
            self.alloy_mat = np.ndarray([SuchSoftware.num_poss], dtype=type(1))
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

            # Initialize list of atomic column classes. self.columns is the array that will hold all the column
            # information as it is found by the algorithms. The dummy instance is created to give the correct dtype
            # of the array, all though numpy still gives a warning when non-standard types are given to the array later
            # in the code. It still works though:)
            dummy_instance = AtomicColumn(0, 1, 1, 1, 1, 1, SuchSoftware.num_poss)
            self.columns = np.ndarray([1], dtype=type(dummy_instance))

    def column_finder(self, search_type='s'):
        """Find local peaks in self.im_mat and store them as AtomicColumn types in self.columns.

        keyword arguments:
        search_type = string that determines the stopping mechanism of the algorithm (default 's')

        Argument search_type is a string, and valid inputs are 's' and 't'. 's' is default.
        The method assumes that the object is not in a 'empty'-loaded state.
        search_type='s' will search until it finds self.search_size number of columns.
        search_type='t' will search until self.search_mat.max() is less than self.threshold.
        """

        if search_type == 's':

            counter = self.num_columns
            self.column_circumference_mat = mat_op.gen_framed_mat(self.column_circumference_mat, self.r + self.overhead)
            self.search_mat = mat_op.gen_framed_mat(self.search_mat, self.r + self.overhead)
            self.im_mat = mat_op.gen_framed_mat(self.im_mat, self.r + self.overhead)

            for x in range(self.num_columns, self.search_size):

                pos = np.unravel_index(self.search_mat.argmax(), (self.N + 2 * (self.r + self.overhead), self.M + 2 * (self.r + self.overhead)))
                column_portrait, x_fit, y_fit = Utilities.cm_fit(self.im_mat, pos[1], pos[0], self.r)

                x_fit_real_coor = x_fit - self.r - self.overhead
                y_fit_real_coor = y_fit - self.r - self.overhead

                self.search_mat = mat_op.delete_pixels(self.search_mat, x_fit, y_fit, self.r + self.overhead)

                if counter == 0:
                    self.columns[0] = AtomicColumn(counter, x_fit_real_coor, y_fit_real_coor, self.r, np.max(column_portrait), 0, SuchSoftware.num_poss)
                else:
                    dummy = AtomicColumn(counter, x_fit_real_coor, y_fit_real_coor, self.r, np.max(column_portrait), 0, SuchSoftware.num_poss)
                    self.columns = np.append(self.columns, dummy)

                    self.reset_prop_vector(x)

                self.column_centre_mat[y_fit_real_coor, x_fit_real_coor, 0] = 1
                self.column_centre_mat[y_fit_real_coor, x_fit_real_coor, 1] = counter
                self.column_circumference_mat = mat_op.draw_circle(self.column_circumference_mat, x_fit, y_fit, self.r)

                print(str(counter) + ': (' + str(x_fit_real_coor) + ', ' + str(y_fit_real_coor) + ') | (' + str(pos[1]) + ', ' + str(pos[0]) + ')')
                self.num_columns = self.num_columns + 1
                self.num_un = self.num_un + 1
                counter = counter + 1

            self.column_circumference_mat = mat_op.gen_de_framed_mat(self.column_circumference_mat, self.r + self.overhead)
            self.search_mat = mat_op.gen_de_framed_mat(self.search_mat, self.r + self.overhead)
            self.im_mat = mat_op.gen_de_framed_mat(self.im_mat, self.r + self.overhead)
            self.calc_avg_gamma()
            self.summarize_stats()

        elif search_type == 't':

            current_max = self.search_mat.max()
            counter = self.num_columns
            self.column_circumference_mat = mat_op.gen_framed_mat(self.column_circumference_mat, self.r + self.overhead)
            self.search_mat = mat_op.gen_framed_mat(self.search_mat, self.r + self.overhead)
            self.im_mat = mat_op.gen_framed_mat(self.im_mat, self.r + self.overhead)

            while current_max > self.threshold:

                pos = np.unravel_index(self.search_mat.argmax(),
                                       (self.N + 2 * (self.r + self.overhead), self.M + 2 * (self.r + self.overhead)))
                column_portrait, x_fit, y_fit = Utilities.cm_fit(self.im_mat, pos[1], pos[0], self.r)

                x_fit_real_coor = x_fit - self.r - self.overhead
                y_fit_real_coor = y_fit - self.r - self.overhead

                self.search_mat = mat_op.delete_pixels(self.search_mat, x_fit, y_fit, self.r + self.overhead)

                if counter == 0:
                    self.columns[0] = AtomicColumn(counter, x_fit_real_coor, y_fit_real_coor, self.r, np.max(column_portrait), 0, SuchSoftware.num_poss)
                else:
                    dummy = AtomicColumn(counter, x_fit_real_coor, y_fit_real_coor, self.r, np.max(column_portrait), 0, SuchSoftware.num_poss)
                    self.columns = np.append(self.columns, dummy)

                self.reset_prop_vector(counter)

                self.column_centre_mat[y_fit_real_coor, x_fit_real_coor, 0] = 1
                self.column_centre_mat[y_fit_real_coor, x_fit_real_coor, 1] = counter
                self.column_circumference_mat = mat_op.draw_circle(self.column_circumference_mat, x_fit, y_fit, self.r)

                current_max = self.search_mat.max()
                print(str(counter) + ': (' + str(x_fit_real_coor) + ', ' + str(y_fit_real_coor) + ') | (' + str(
                    pos[1]) + ', ' + str(pos[0]) + ')')
                self.num_columns = self.num_columns + 1
                self.num_un = self.num_un + 1
                counter = counter + 1

            self.column_circumference_mat = mat_op.gen_de_framed_mat(self.column_circumference_mat, self.r + self.overhead)
            self.search_mat = mat_op.gen_de_framed_mat(self.search_mat, self.r + self.overhead)
            self.im_mat = mat_op.gen_de_framed_mat(self.im_mat, self.r + self.overhead)
            self.calc_avg_gamma()
            self.summarize_stats()

        else:

            print('No such search type')

    def column_analyser(self, i, search_type='1'):

        if search_type == '0':

            # Tidy the flag states:
            self.reset_all_flags()
            self.find_edge_columns()

            # Reset some stuff
            self.precipitate_boarder = np.ndarray([1], dtype=int)
            self.boarder_size = 0

            for y in range(0, self.num_columns):

                if self.columns[y].is_edge_column:

                    self.reset_prop_vector(y, bias=3)

                else:

                    self.reset_prop_vector(y)
                    self.apply_intensity_score(y)
                    self.apply_angle_score(y)

            self.precipitate_controller(i)
            self.define_levels(i)
            self.summarize_stats()

        if search_type == '1':

            self.set_alloy_mat()

            # Tidy the flag states:
            self.reset_all_flags()
            self.reset_popularity_flags()
            self.find_edge_columns()

            # Reset some stuff
            self.precipitate_boarder = np.ndarray([1], dtype=int)
            self.boarder_size = 0

            for y in range(0, self.num_columns):

                self.columns[y].neighbour_indices, junk_1, junk_2 = self.find_nearest(y, False, num=8)

                if self.columns[y].is_edge_column:

                    self.reset_prop_vector(y, bias=3)

                else:

                    self.reset_prop_vector(y)
                    self.apply_intensity_score(y)
                    self.apply_angle_score(y)

            self.precipitate_controller(i)
            self.define_levels(i)

        elif search_type == '2':

            for y in range(0, self.num_columns):

                self.find_consistent_perturbations_simple(y, True)

        elif search_type == '3':

            for y in range(0, self.num_columns):

                self.find_consistent_perturbations_simple(y)

        elif search_type == '4':

            for y in range(0, self.num_columns):

                    self.find_consistent_perturbations_advanced(y)

        elif search_type == '5':

            for y in range(0, self.num_columns):

                    self.find_consistent_perturbations_advanced(y, experimental=True)

        elif search_type == '6':

            for y in range(0, self.num_columns):

                if self.columns[y].is_popular:

                    self.connection_shift_on_level(y)

        elif search_type == '7':

            for x in range(0, self.num_columns):

                for y in range(0, self.columns[x].n()):

                    if self.columns[self.columns[x].neighbour_indices[y]].level == self.columns[x].level:

                        if self.columns[x].is_edge_column or self.columns[self.columns[x].neighbour_indices[y]].is_edge_column:

                            pass

                        else:

                            self.resolve_edge_inconsistency(x, self.columns[x].neighbour_indices[y])

        elif search_type == '8':

            for x in range(0, self.num_columns):

                if self.columns[x].is_unpopular and not self.columns[x].is_edge_column:

                    self.resolve_edge_inconsistency(x, self.columns[x].neighbour_indices[self.columns[x].n() - 1])

        elif search_type == '9':

            self.find_edge_columns()
            self.define_levels(i, 0)

        elif search_type == '10':

            self.column_analyser(i, search_type='1')
            self.column_analyser(i, search_type='2')
            self.column_analyser(i, search_type='0')
            self.column_analyser(i, search_type='3')
            self.column_analyser(i, search_type='0')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='0')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='0')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='0')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='0')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='6')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='0')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='7')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='8')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='7')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='8')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='5')

        elif search_type == '11':

            self.column_analyser(i, search_type='1')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='0')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='0')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='0')
            self.column_analyser(i, search_type='6')

            Previous_chi = self.chi
            has_decreased = True

            while has_decreased:

                self.column_analyser(i, search_type='7')
                self.column_analyser(i, search_type='5')
                self.column_analyser(i, search_type='5')
                self.column_analyser(i, search_type='8')
                self.column_analyser(i, search_type='5')
                self.column_analyser(i, search_type='5')

                self.precipitate_controller(i)
                self.define_levels(i)
                self.summarize_stats()

                if self.chi < Previous_chi:
                    has_decreased = True
                else:
                    has_decreased = False

                Previous_chi = self.chi

        elif search_type == '20':

            pass

        elif search_type == '21':

            self.column_analyser(i, search_type='1')
            self.column_analyser(i, search_type='2')
            self.column_analyser(i, search_type='0')
            self.column_analyser(i, search_type='3')
            self.column_analyser(i, search_type='0')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='0')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='0')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='0')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='0')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='4')

        elif search_type == '22':

            self.column_analyser(i, search_type='6')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='0')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='4')
            self.column_analyser(i, search_type='5')

        elif search_type == '23':

            self.column_analyser(i, search_type='7')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='8')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='7')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='8')
            self.column_analyser(i, search_type='5')
            self.column_analyser(i, search_type='5')

        else:

            print('Search type not implemented!')

    def mesh_crawler(self):

        pass

    def classify_pair(self, i, j):

        neighbour_type = 0
        partner_type = 0
        intersects = False

        i_neighbour_to_j = False
        j_neighbour_to_i = False
        i_partner_to_j = False
        j_partner_to_i = False

        for x in range(0, 8):

            if self.columns[i].neighbour_indices[x] == j:
                j_neighbour_to_i = True
                if x < self.columns[i].n():
                    j_partner_to_i = True

            if self.columns[j].neighbour_indices[x] == i:
                i_neighbour_to_j = True
                if x < self.columns[j].n():
                    i_partner_to_j = True

        if not i_neighbour_to_j and not j_neighbour_to_i:
            neighbour_type = 0
        elif not i_neighbour_to_j and j_neighbour_to_i:
            neighbour_type = 1
        elif i_neighbour_to_j and not j_neighbour_to_i:
            neighbour_type = 2
        elif i_neighbour_to_j and j_neighbour_to_i:
            neighbour_type = 3

        if not i_partner_to_j and not j_partner_to_i:
            partner_type = 0
        elif not i_partner_to_j and j_partner_to_i:
            partner_type = 1
        elif i_partner_to_j and not j_partner_to_i:
            partner_type = 2
        elif i_partner_to_j and j_partner_to_i:
            partner_type = 3

        if self.columns[i].level == self.columns[j].level:
            level_type = 0
        else:
            level_type = 1

        if partner_type == 0:

            geometry_type_clockwise = -1
            geometry_type_anticlockwise = -1
            geo_type_symmetry = -1

        else:

            indices, num_edges_clockwise_right = self.find_shape(i, j, clockwise=True)
            indices, num_edges_clockwise_left = self.find_shape(j, i, clockwise=True)
            indices, num_edges_anticlockwise_right = self.find_shape(j, i, clockwise=False)
            indices, num_edges_anticlockwise_left = self.find_shape(i, j, clockwise=False)

            if num_edges_clockwise_right == 3 and num_edges_clockwise_left == 3:
                geometry_type_clockwise = 1
            elif num_edges_clockwise_right == 5 and num_edges_clockwise_left == 3:
                geometry_type_clockwise = 2
            elif num_edges_clockwise_right == 3 and num_edges_clockwise_left == 5:
                geometry_type_clockwise = 3
            elif num_edges_clockwise_right == 4 and num_edges_clockwise_left == 3:
                geometry_type_clockwise = 4
            elif num_edges_clockwise_right == 3 and num_edges_clockwise_left == 4:
                geometry_type_clockwise = 5
            elif num_edges_clockwise_right == 4 and num_edges_clockwise_left == 4:
                geometry_type_clockwise = 6
            elif num_edges_clockwise_right == 5 and num_edges_clockwise_left == 5:
                geometry_type_clockwise = 7
            else:
                geometry_type_clockwise = 0

            if num_edges_anticlockwise_right == 3 and num_edges_anticlockwise_left == 3:
                geometry_type_anticlockwise = 1
            elif num_edges_anticlockwise_right == 5 and num_edges_anticlockwise_left == 3:
                geometry_type_anticlockwise = 2
            elif num_edges_anticlockwise_right == 3 and num_edges_anticlockwise_left == 5:
                geometry_type_anticlockwise = 3
            elif num_edges_anticlockwise_right == 4 and num_edges_anticlockwise_left == 3:
                geometry_type_anticlockwise = 4
            elif num_edges_anticlockwise_right == 3 and num_edges_anticlockwise_left == 4:
                geometry_type_anticlockwise = 5
            elif num_edges_anticlockwise_right == 4 and num_edges_anticlockwise_left == 4:
                geometry_type_anticlockwise = 6
            elif num_edges_anticlockwise_right == 5 and num_edges_anticlockwise_left == 5:
                geometry_type_anticlockwise = 7
            else:
                geometry_type_anticlockwise = 0

            if geometry_type_clockwise == geometry_type_anticlockwise:
                geo_type_symmetry = 0
            else:
                geo_type_symmetry = 1

        # Implement method to find intersections

        return neighbour_type, partner_type, level_type, geometry_type_clockwise, geometry_type_anticlockwise,\
            geo_type_symmetry, intersects

    def increase_h_value(self, i):

        changed = False
        h = self.columns[i].h_index

        if h == 0 or h == 1:

            self.reset_prop_vector(i, bias=3)
            changed = True

        elif h == 3:

            self.reset_prop_vector(i, bias=5)
            changed = True

        return changed

    def decrease_h_value(self, i):

        changed = False
        h = self.columns[i].h_index

        if h == 5:

            self.reset_prop_vector(i, bias=3)
            changed = True

        elif h == 3:

            self.reset_prop_vector(i)
            self.columns[i].prob_vector[0] = self.columns[i].prob_vector[0] + 0.1
            self.columns[i].prob_vector[1] = self.columns[i].prob_vector[1] + 0.1
            self.renorm_prop(i)
            self.redefine_species(i)
            changed = True

        return changed

    def resolve_edge_inconsistency(self, i, j, clockwise=True):

        neighbour_type, partner_type, level_type, geometry_type_clockwise, geometry_type_anticlockwise, \
            geo_type_symmetry, intersects = self.classify_pair(i, j)

        if geo_type_symmetry == 0:
            geometry_type = geometry_type_clockwise
        else:
            geometry_type = 0

        if neighbour_type == 0:
            i_neighbour_to_j = False
            j_neighbour_to_i = False
        elif neighbour_type == 1:
            i_neighbour_to_j = False
            j_neighbour_to_i = True
        elif neighbour_type == 2:
            i_neighbour_to_j = True
            j_neighbour_to_i = False
        elif neighbour_type == 3:
            i_neighbour_to_j = True
            j_neighbour_to_i = True
        else:
            i_neighbour_to_j = False
            j_neighbour_to_i = False

        if partner_type == 0:
            i_partner_to_j = False
            j_partner_to_i = False
        elif partner_type == 1:
            i_partner_to_j = False
            j_partner_to_i = True
        elif partner_type == 2:
            i_partner_to_j = True
            j_partner_to_i = False
        elif partner_type == 3:
            i_partner_to_j = True
            j_partner_to_i = True
        else:
            i_partner_to_j = False
            j_partner_to_i = False

        i_index_in_j = -1
        j_index_in_i = -1

        for x in range(0, 8):
            if self.columns[i].neighbour_indices[x] == j:
                j_index_in_i = x

        if j_index_in_i == -1:
            self.columns[i].neighbour_indices[7] = j
            j_index_in_i = 7

        for x in range(0, 8):
            if self.columns[j].neighbour_indices[x] == i:
                i_index_in_j = x

        if i_index_in_j == -1:
            self.columns[j].neighbour_indices[7] = i
            i_index_in_j = 7

        if i_partner_to_j:
            # Perturb neighbours of j such that i is last element in k^j
            self.perturbator(j, i_index_in_j, self.columns[j].n() - 1)
        else:
            # Perturb neighbours of j such that i is last element in k^j
            self.perturbator(j, i_index_in_j, self.columns[j].n())

        if j_partner_to_i:
            # Perturb neighbours of i such that j is last k
            self.perturbator(i, j_index_in_i, self.columns[i].n() - 1)
        else:
            # Perturb neighbours of i such that j is last k
            self.perturbator(i, j_index_in_i, self.columns[i].n())

        if clockwise:

            shape_1_indices, num_edge_1 = self.find_shape(i, j, clockwise=clockwise)
            shape_2_indices, num_edge_2 = self.find_shape(j, i, clockwise=clockwise)

        else:

            shape_1_indices, num_edge_1 = self.find_shape(j, i, clockwise=clockwise)
            shape_2_indices, num_edge_2 = self.find_shape(i, j, clockwise=clockwise)

        print(str(i) + ', ' + str(j) + ': ')

        if geometry_type == 1:
            # This means we want to break the connection!

            if partner_type == 1:
                if not self.try_connect(i, j_index_in_i):
                    if not self.decrease_h_value(i):
                        print('Could not reconnect!')

            elif partner_type == 2:
                if not self.try_connect(j, i_index_in_j):
                    if not self.decrease_h_value(j):
                        print('Could not reconnect!')

            elif partner_type == 3:
                if not self.try_connect(j, i_index_in_j):
                    if not self.decrease_h_value(j):
                        print('Could not reconnect!')
                if not self.try_connect(i, j_index_in_i):
                    if not self.decrease_h_value(i):
                        print('Could not reconnect!')

        if geometry_type == 2 or geometry_type == 3:
            # This means we want to switch connections to make geometry type 6

            loser_index_in_stayer = -1
            loser_connected_to_stayer = False
            stayer_connected_to_loser = False
            new_index_in_stayer = -1

            if geometry_type == 2:

                ind_1 = shape_1_indices[2]
                ind_2 = shape_1_indices[4]

            else:

                ind_1 = shape_2_indices[4]
                ind_2 = shape_2_indices[2]

            distance_1 = np.sqrt((self.columns[ind_1].x - self.columns[i].x) ** 2 + (
                    self.columns[ind_1].y - self.columns[i].y) ** 2)
            distance_2 = np.sqrt((self.columns[j].x - self.columns[ind_2].x) ** 2 + (
                    self.columns[j].y - self.columns[ind_2].y) ** 2)

            if distance_1 < distance_2:
                index_stayer = i
                index_loser = j
                stayer_index_in_loser = i_index_in_j
                index_new = ind_1
                if i_partner_to_j:
                    loser_connected_to_stayer = True
                if j_partner_to_i:
                    stayer_connected_to_loser = True
            else:
                index_stayer = j
                index_loser = i
                stayer_index_in_loser = j_index_in_i
                index_new = ind_2
                if j_partner_to_i:
                    loser_connected_to_stayer = True
                if i_partner_to_j:
                    stayer_connected_to_loser = True

            for x in range(self.columns[index_stayer].n(), 8):
                if self.columns[index_stayer].neighbour_indices[x] == index_new:
                    new_index_in_stayer = x

            if new_index_in_stayer == -1:
                self.columns[index_stayer].neighbour_indices[7] = index_new
                new_index_in_stayer = 7

            self.perturbator(index_stayer, self.columns[index_stayer].n(), new_index_in_stayer)

            if loser_connected_to_stayer:
                if not self.try_connect(index_loser, stayer_index_in_loser):
                    if not self.decrease_h_value(index_loser):
                        print('Could not reconnect!')

            if stayer_connected_to_loser:
                self.perturbator(index_stayer, loser_index_in_stayer, new_index_in_stayer)
            else:
                if not self.increase_h_value(index_stayer):
                    print('Could not reconnect!')

        if geometry_type == 4 or geometry_type == 5:

            pass

        if geometry_type == 6:
            # This means we want to keep the connection

            if partner_type == 1:
                if not self.increase_h_value(j):
                    print('Could not reconnect!')

            elif partner_type == 2:
                if not self.increase_h_value(i):
                    print('Could not reconnect!')

        if geometry_type == 0:

            print(str(num_edge_1) + ', ' + str(num_edge_2))
            print(shape_1_indices)
            print(shape_2_indices)

    def try_connect(self, i, j_index_in_i):

        changed = False
        better_friend = False
        friend_index_in_i = -1

        for x in range(self.columns[i].n(), 8):

            if self.test_reciprocality(i, self.columns[i].neighbour_indices[x]):

                if not self.columns[i].level == self.columns[self.columns[i].neighbour_indices[x]].level:

                    better_friend = True
                    friend_index_in_i = x

                else:

                    print('Maybe should have?')

        if better_friend:

            self.perturbator(i, j_index_in_i, friend_index_in_i)
            changed = True

        return changed

    def distance(self, i, j):

        x_i = self.columns[i].x
        y_i = self.columns[i].y
        x_j = self.columns[j].x
        y_j = self.columns[j].y

        return np.sqrt((x_j - x_i)**2 + (y_j - y_i)**2)

    def perturbator(self, i, a, b):

        val_a = self.columns[i].neighbour_indices[a]
        val_b = self.columns[i].neighbour_indices[b]

        self.columns[i].neighbour_indices[a] = val_b
        self.columns[i].neighbour_indices[b] = val_a

    def find_edge_columns(self):

        for y in range(0, self.num_columns):

            x_coor = self.columns[y].x
            y_coor = self.columns[y].y
            margin = 6 * self.r

            if x_coor < margin or x_coor > self.M - margin - 1 or y_coor < margin or y_coor > self.N - margin - 1:

                self.columns[y].is_edge_column = True
                self.reset_prop_vector(y, bias=3)

            else:

                self.columns[y].is_edge_column = False

    def find_consistent_perturbations_simple(self, y, sub=False):

        if not self.columns[y].is_edge_column:

            n = self.columns[y].n()

            if sub:

                n = 3

            indices = self.columns[y].neighbour_indices
            new_indices = np.zeros([indices.shape[0]], dtype=int)
            found = 0

            for x in range(0, indices.shape[0]):

                n2 = 3

                if self.columns[indices[x]].h_index == 0 or self.columns[indices[x]].h_index == 1:
                    n2 = 3
                elif self.columns[indices[x]].h_index == 3:
                    n2 = 4
                elif self.columns[indices[x]].h_index == 5:
                    n2 = 5
                else:
                    print('Problem in find_consistent_perturbations_simple!')

                neighbour_indices = self.columns[indices[x]].neighbour_indices

                for z in range(0, n2):

                    if neighbour_indices[z] == y:
                        new_indices[found] = indices[x]
                        found = found + 1

            if found == n:

                index_positions = np.zeros([found], dtype=int)

                for k in range(0, found):

                    for z in range(0, indices.shape[0]):

                        if indices[z] == new_indices[k]:
                            index_positions[k] = z

                counter = found - 1

                for i in range(0, indices.shape[0]):

                    are_used = False

                    for z in range(0, found):

                        if i == index_positions[z]:

                            are_used = True

                    if not are_used:

                        counter = counter + 1
                        new_indices[counter] = indices[i]

                self.columns[y].neighbour_indices = new_indices
                self.columns[y].is_popular = False
                self.columns[y].is_unpopular = False

            elif found > n:

                index_positions = np.zeros([found], dtype=int)

                for k in range(0, found):

                    for z in range(0, indices.shape[0]):

                        if indices[z] == new_indices[k]:
                            index_positions[k] = z

                counter = found - 1

                for i in range(0, indices.shape[0]):

                    are_used = False

                    for z in range(0, found):

                        if i == index_positions[z]:
                            are_used = True

                    if not are_used:
                        counter = counter + 1
                        new_indices[counter] = indices[i]

                self.columns[y].neighbour_indices = new_indices
                self.columns[y].is_unpopular = False
                self.columns[y].is_popular = True

            else:

                index_positions = np.zeros([found], dtype=int)

                for k in range(0, found):

                    for z in range(0, indices.shape[0]):

                        if indices[z] == new_indices[k]:
                            index_positions[k] = z

                counter = found - 1

                for i in range(0, indices.shape[0]):

                    are_used = False

                    for z in range(0, found):

                        if i == index_positions[z]:
                            are_used = True

                    if not are_used:
                        counter = counter + 1
                        new_indices[counter] = indices[i]

                self.columns[y].neighbour_indices = new_indices
                self.columns[y].is_unpopular = True
                self.columns[y].is_popular = False

    def sort_neighbours_by_level(self, y):

        n = self.columns[y].n()

        num_wrong_flags = 0

        for x in range(0, n):

            if self.columns[self.columns[y].neighbour_indices[x]].level == self.columns[y].level:
                num_wrong_flags = num_wrong_flags + 1

        if num_wrong_flags >= n - 1:

            if self.columns[y].level == 0:
                self.columns[y].level = 1
            else:
                self.columns[y].level = 0

            num_wrong_flags = 0

            for x in range(0, n):

                if self.columns[self.columns[y].neighbour_indices[x]].level == self.columns[y].level:
                    num_wrong_flags = num_wrong_flags + 1

        finished = False
        debug_counter = 0

        while not finished:

            print(debug_counter)

            num_perturbations = 0

            for x in range(0, n - 1):

                if not self.columns[self.columns[y].neighbour_indices[x]].level == self.columns[y].level:
                    pass
                else:
                    self.perturbator(y, x, x + 1)
                    num_perturbations = num_perturbations + 1

                if x == n - num_wrong_flags - 2 and num_perturbations == 0:

                    finished = True

            debug_counter = debug_counter + 1

    def find_consistent_perturbations_advanced(self, y, experimental=False):

        if not self.columns[y].is_edge_column:

            n = self.columns[y].n()

            indices = deepcopy(self.columns[y].neighbour_indices)
            new_indices = np.zeros([indices.shape[0]], dtype=int)
            index_of_unpopular_neighbours = np.zeros([indices.shape[0]], dtype=int)
            found = 0

            for x in range(0, indices.shape[0]):

                if self.columns[indices[x]].is_unpopular:
                    index_of_unpopular_neighbours[x] = indices[x]
                else:
                    index_of_unpopular_neighbours[x] = -1

                n2 = 3

                if self.columns[indices[x]].h_index == 0 or self.columns[indices[x]].h_index == 1:
                    n2 = 3
                elif self.columns[indices[x]].h_index == 3:
                    n2 = 4
                elif self.columns[indices[x]].h_index == 5:
                    n2 = 5
                else:
                    print('Problem in find_consistent_perturbations_simple!')

                neighbour_indices = self.columns[indices[x]].neighbour_indices

                for z in range(0, n2):

                    if neighbour_indices[z] == y:
                        new_indices[found] = indices[x]
                        found = found + 1
                        index_of_unpopular_neighbours[x] = -1

            if found == n:

                index_positions = np.zeros([found], dtype=int)

                for k in range(0, found):

                    for z in range(0, indices.shape[0]):

                        if indices[z] == new_indices[k]:
                            index_positions[k] = z

                counter = found - 1

                for i in range(0, indices.shape[0]):

                    are_used = False

                    for z in range(0, found):

                        if i == index_positions[z]:

                            are_used = True

                    if not are_used:

                        counter = counter + 1
                        new_indices[counter] = indices[i]

                self.columns[y].neighbour_indices = new_indices
                self.columns[y].is_unpopular = False
                self.columns[y].is_popular = False

                if experimental:
                    self.sort_neighbours_by_level(y)

            elif found > n:

                index_positions = np.zeros([found], dtype=int)

                for k in range(0, found):

                    for z in range(0, indices.shape[0]):

                        if indices[z] == new_indices[k]:
                            index_positions[k] = z

                counter = found - 1

                for i in range(0, indices.shape[0]):

                    are_used = False

                    for z in range(0, found):

                        if i == index_positions[z]:
                            are_used = True

                    if not are_used:
                        counter = counter + 1
                        new_indices[counter] = indices[i]

                self.columns[y].neighbour_indices = new_indices
                self.columns[y].is_popular = True
                self.columns[y].is_unpopular = False

                if experimental:
                    self.sort_neighbours_by_level(y)

            else:

                print(index_of_unpopular_neighbours)

                index_positions = np.zeros([found], dtype=int)

                for k in range(0, found):

                    for z in range(0, indices.shape[0]):

                        if indices[z] == new_indices[k]:
                            index_positions[k] = z

                counter = found - 1

                for i in range(0, indices.shape[0]):

                    are_used = False

                    for z in range(0, found):

                        if i == index_positions[z]:
                            are_used = True

                    if not are_used:
                        counter = counter + 1
                        new_indices[counter] = indices[i]

                self.columns[y].neighbour_indices = new_indices
                self.columns[y].is_unpopular = True
                self.columns[y].is_popular = False

                friend_index = -1
                distance = self.N

                for x in range(0, indices.shape[0]):

                    if not index_of_unpopular_neighbours[x] == -1:

                        temp_distance = np.sqrt((self.columns[y].x -
                                                 self.columns[index_of_unpopular_neighbours[x]].x)**2 +
                                                (self.columns[y].y -
                                                 self.columns[index_of_unpopular_neighbours[x]].y)**2)

                        if temp_distance < distance:
                            distance = temp_distance
                            friend_index = index_of_unpopular_neighbours[x]

                if not friend_index == -1:

                    i_1 = -1

                    for j in range(0, indices.shape[0]):

                        if new_indices[j] == friend_index:
                            i_1 = j

                    print('y: ' + str(y) + ', found: ' + str(found) + ', i_1: ' + str(i_1) + ', friend_index: ' + str(friend_index))

                    self.columns[y].neighbour_indices[i_1] = self.columns[y].neighbour_indices[found]
                    self.columns[y].neighbour_indices[found] = friend_index

                    self.find_consistent_perturbations_simple(friend_index)
                    self.find_consistent_perturbations_simple(y)

                else:

                    distance = self.N
                    friend_index = -1

                    for x in range(found, indices.shape[0]):

                        if not self.columns[self.columns[y].neighbour_indices[x]].level == self.columns[y].level:

                            temp_distance = np.sqrt((self.columns[y].x -
                                                     self.columns[self.columns[y].neighbour_indices[x]].x) ** 2 +
                                                    (self.columns[y].y -
                                                     self.columns[self.columns[y].neighbour_indices[x]].y) ** 2)

                            if temp_distance < distance:
                                distance = temp_distance
                                friend_index = self.columns[y].neighbour_indices[x]

                    if not friend_index == -1:

                        i_1 = -1

                        for j in range(0, indices.shape[0]):

                            if new_indices[j] == friend_index:
                                i_1 = j

                        self.columns[y].neighbour_indices[i_1] = self.columns[y].neighbour_indices[found]
                        self.columns[y].neighbour_indices[found] = friend_index

                        self.find_consistent_perturbations_simple(friend_index)
                        self.find_consistent_perturbations_simple(y)

                if experimental:
                    self.sort_neighbours_by_level(y)

    def connection_shift_on_level(self, i, experimental=False):

        n = self.columns[i].n()
        indices = self.columns[i].neighbour_indices

        bad_index = -1
        good_index = -1

        print(str(i) + ': n = ' + str(n) + '\n----------------')

        for x in range(0, n):

            print('k: ' + str(x))

            if self.columns[indices[x]].level == self.columns[i].level:

                bad_index = x

        if experimental:

            high = n + 1

        else:

            high = 8

        for x in range(n, high):

            print('j: ' + str(n + high - 1 - x))

            if not self.columns[indices[n + high - 1 - x]].level == self.columns[i].level:

                good_index = n + high - 1 - x

        if not bad_index == -1 and not good_index == -1:
            print(
                str(i) + ' | ' + str(bad_index) + ': ' + str(indices[bad_index]) + ' | ' + str(good_index) + ': ' + str(
                    indices[good_index]))
            self.perturbator(i, bad_index, good_index)

    def reset_all_flags(self, excepting=0):

        for x in range(0, self.num_columns):

            if not excepting == 1:
                self.columns[x].flag_1 = False
            if not excepting == 2:
                self.columns[x].flag_2 = False
            if not excepting == 3:
                self.columns[x].flag_3 = False
            if not excepting == 4:
                self.columns[x].flag_4 = False

    def reset_popularity_flags(self):

        for x in range(0, self.num_columns):
            self.columns[x].is_popular = False
            self.columns[x].is_unpopular = False

        self.num_unpopular = 0
        self.num_popular = 0
        self.num_inconsistencies = 0

    def find_nearest(self, i, is_certain, num=3):

        if is_certain:
            n = self.columns[i].n()
        else:
            n = num

        indices = np.ndarray([n], dtype=np.int)
        distances = np.ndarray([n], dtype=np.float64)
        r = int(SuchSoftware.al_lattice_const / self.scale)
        r = int(2 * r)
        x_0 = self.columns[i].x
        y_0 = self.columns[i].y
        found = 0

        for x in range(x_0 - r, x_0 + r):
            for y in range(y_0 - r, y_0 + r):

                # Look for column at (x, y)

                if y < 0 or y > self.N - 1 or x < 0 or x > self.M - 1:

                    pass

                elif x == x_0 and y == y_0:

                    pass

                else:

                    if self.column_centre_mat[y, x, 0] == 1:

                        # add to results

                        distance = np.sqrt((x_0 - x) ** 2 + (y_0 - y) ** 2)

                        if found >= n:

                            # special case
                            if distance < distances.max():

                                pos = distances.argmax()
                                indices[pos] = self.column_centre_mat[y, x, 1]
                                distances[pos] = distance

                        else:

                            indices[found] = self.column_centre_mat[y, x, 1]
                            distances[found] = distance

                        found = found + 1

        if found < n:
            print('Did not go to plan!')

        indices, distances = Utilities.sort_neighbours(indices, distances, n)

        return indices, distances, n

    def analyse_prop_vector(self, i):

        h_value = self.columns[i].prob_vector.max()
        nh_value = 0.0
        h_value_index = self.columns[i].prob_vector.argmax()
        is_certain = False

        for x in range(0, SuchSoftware.num_poss):
            if h_value > self.columns[i].prob_vector[x] >= nh_value:
                nh_value = self.columns[i].prob_vector[x]

        if h_value > 0.0:
            self.columns[i].confidence = 1 - nh_value / h_value
            if 1 - nh_value / h_value > self.certainty_threshold:
                is_certain = True
            if self.columns[i].confidence == 0:
                h_value_index = 6
        else:
            h_value_index = 6

        if h_value - nh_value < 0.00001:
            h_value_index = 6

        return h_value_index, is_certain

    def apply_dist(self, i, other_radii):

        if self.columns[i].prob_vector.max() <= 0.0:

            for y in range(0, SuchSoftware.num_poss):
                self.columns[i].prob_vector[y] = 1.0

            self.apply_dist(i, other_radii)

        else:

            for y in range(0, SuchSoftware.num_poss):

                if SuchSoftware.radii[y] <= other_radii:
                    self.columns[i].prob_vector[y] = self.columns[i].prob_vector[y] * Utilities.normal_dist(
                        SuchSoftware.radii[y], other_radii, self.dist_1_std)
                else:
                    self.columns[i].prob_vector[y] = self.columns[i].prob_vector[y] * Utilities.normal_dist(
                        SuchSoftware.radii[y], other_radii, self.dist_2_std)

            self.renorm_prop(i)
            self.redefine_species(i)

    def apply_intensity_score(self, i):

        if self.columns[i].prob_vector.max() <= 0.0:

            self.reset_prop_vector(i)

        for x in range(0, SuchSoftware.num_poss):
            self.columns[i].prob_vector[x] = self.columns[i].prob_vector[x] * Utilities.normal_dist(
                self.columns[i].peak_gamma, SuchSoftware.intensities[self.alloy][x], self.dist_8_std)

        self.renorm_prop(i)
        self.redefine_species(i)

    def apply_angle_score(self, i):

        if self.columns[i].prob_vector.max() <= 0.0:

            self.reset_prop_vector(i)

        n = 3

        a = np.ndarray([n], dtype=np.int)
        b = np.ndarray([n], dtype=np.int)
        alpha = np.ndarray([n], dtype=np.float64)

        for x in range(0, n):

            a[x] = self.columns[self.columns[i].neighbour_indices[x]].x - self.columns[i].x
            b[x] = self.columns[self.columns[i].neighbour_indices[x]].y - self.columns[i].y

        for x in range(0, n):

            x_pluss = 0

            if x == n - 1:

                pass

            else:

                x_pluss = x + 1

            alpha[x] = Utilities.find_angle(a[x], a[x_pluss], b[x], b[x_pluss])

        # Deal with cases where the angle is over pi radians:
        max_index = alpha.argmax()
        angle_sum = 0.0
        for x in range(0, n):
            if x == max_index:
                pass
            else:
                angle_sum = angle_sum + alpha[x]
        if alpha.max() == angle_sum:
            alpha[max_index] = 2 * np.pi - alpha.max()

        symmetry_3 = 2 * np.pi / 3
        symmetry_4 = np.pi / 2
        symmetry_5 = 2 * np.pi / 5

        correction_factor_3 = Utilities.normal_dist(alpha.max(), symmetry_3, self.dist_3_std)
        correction_factor_4 = Utilities.normal_dist(alpha.max(), 2 * symmetry_4, self.dist_4_std)
        correction_factor_5 = Utilities.normal_dist(alpha.min(), symmetry_5, self.dist_5_std)

        if alpha.min() < symmetry_5:
            correction_factor_5 = Utilities.normal_dist(symmetry_5, symmetry_5, self.dist_5_std)

        self.columns[i].prob_vector = self.columns[i].prob_vector *\
            [correction_factor_3, correction_factor_3, 1.0, correction_factor_4, 1.0,
             correction_factor_5, 1.0]
        self.renorm_prop(i)

        correction_factor_3 = Utilities.normal_dist(alpha.min(), symmetry_3, self.dist_3_std)
        correction_factor_4 = Utilities.normal_dist(alpha.min(), symmetry_4, self.dist_4_std)

        self.columns[i].prob_vector = self.columns[i].prob_vector *\
            [correction_factor_3, correction_factor_3, 1.0,
             correction_factor_4, 1.0, correction_factor_5, 1.0]

        self.renorm_prop(i)
        self.redefine_species(i)

    def redefine_species(self, i):

        h_value = 0.0
        h_value_index = 0

        for y in range(0, SuchSoftware.num_poss):

            if self.columns[i].prob_vector[y] > h_value:
                h_value = self.columns[i].prob_vector[y]
                h_value_index = y

        if h_value_index == 0:
            self.columns[i].atomic_species = 'Si'
            self.columns[i].h_index = 0
        elif h_value_index == 1:
            self.columns[i].atomic_species = 'Cu'
            self.columns[i].h_index = 1
        elif h_value_index == 2:
            self.columns[i].atomic_species = 'Zn'
            self.columns[i].h_index = 2
        elif h_value_index == 3:
            self.columns[i].atomic_species = 'Al'
            self.columns[i].h_index = 3
        elif h_value_index == 4:
            self.columns[i].atomic_species = 'Ag'
            self.columns[i].h_index = 4
        elif h_value_index == 5:
            self.columns[i].atomic_species = 'Mg'
            self.columns[i].h_index = 5
        elif h_value_index == 6:
            self.columns[i].atomic_species = 'Un'
            self.columns[i].h_index = 6
        else:
            print('Problem in self.redefine_species()')

        self.analyse_prop_vector(i)

        if not h_value > 0.0:
            self.columns[i].atomic_species = 'Un'
            self.columns[i].h_index = 6

    def renorm_prop(self, i):

        prop_sum = 0.0
        for y in range(0, SuchSoftware.num_poss):
            prop_sum = prop_sum + self.columns[i].prob_vector[y]
        if prop_sum == 0.0:
            pass
        else:
            correction_factor = 1 / prop_sum
            self.columns[i].prob_vector = correction_factor * self.columns[i].prob_vector

    def reset_prop_vector(self, i, bias=-1):

        for y in range(0, SuchSoftware.num_poss):
            self.columns[i].prob_vector[y] = 1.0

        if bias == -1:
            pass
        elif bias == 0:
            self.columns[i].prob_vector[0] = 1.1
        elif bias == 1:
            self.columns[i].prob_vector[1] = 1.1
        elif bias == 2:
            self.columns[i].prob_vector[2] = 1.1
        elif bias == 3:
            self.columns[i].prob_vector[3] = 1.1
        elif bias == 4:
            self.columns[i].prob_vector[4] = 1.1
        elif bias == 5:
            self.columns[i].prob_vector[5] = 1.1
        elif bias == 6:
            self.columns[i].prob_vector[6] = 1.1

        self.columns[i].prob_vector = self.columns[i].prob_vector * self.alloy_mat

        self.renorm_prop(i)

        self.redefine_species(i)

    def precipitate_controller(self, i):

        self.boarder_size = 0
        self.precipitate_boarder = np.ndarray([1], dtype=int)

        self.reset_all_flags()

        self.precipitate_finder(i)

        counter = 0

        for x in range(0, self.num_columns):

            if self.columns[x].flag_1 or self.columns[x].h_index == 6:
                self.columns[x].is_in_precipitate = False
            else:
                self.columns[x].is_in_precipitate = True

            if self.columns[x].flag_2:

                if counter == 0:
                    self.precipitate_boarder[0] = x
                else:
                    self.precipitate_boarder = np.append(self.precipitate_boarder, x)

                counter = counter + 1

        self.boarder_size = counter
        self.reset_all_flags()
        self.sort_boarder()

    def sort_boarder(self):

        temp_boarder = deepcopy(self.precipitate_boarder)
        selected = np.ndarray([self.boarder_size], dtype=bool)
        for y in range(0, self.boarder_size):
            selected[y] = False
        next_index = 0
        index = 0
        cont_var = True
        selected[0] = True

        while cont_var:

            distance = self.N * self.M

            for x in range(0, self.boarder_size):

                current_distance = np.sqrt((self.columns[self.precipitate_boarder[x]].x -
                                            self.columns[temp_boarder[index]].x)**2 +
                                           (self.columns[self.precipitate_boarder[x]].y -
                                            self.columns[temp_boarder[index]].y)**2)

                if current_distance < distance and not temp_boarder[index] == self.precipitate_boarder[x] and not selected[x]:
                    distance = current_distance
                    next_index = x

            selected[next_index] = True
            index = index + 1

            temp_boarder[index] = self.precipitate_boarder[next_index]

            if index == self.boarder_size - 1:
                cont_var = False

        self.precipitate_boarder = deepcopy(temp_boarder)

    def precipitate_finder(self, i):

        indices, distances, n = self.find_nearest(i, True)

        self.columns[i].flag_1 = True

        for x in range(0, n):

            if not self.columns[indices[x]].h_index == 3:

                if not self.columns[indices[x]].h_index == 6:
                    self.columns[i].flag_2 = True

            else:

                if not self.columns[indices[x]].flag_1:
                    self.precipitate_finder(indices[x])

    def calc_avg_gamma(self):

        if self.num_columns > 0:

            self.im_mat = mat_op.gen_framed_mat(self.im_mat, self.r)

            for x in range(0, self.num_columns):

                self.columns[x].avg_gamma, self.columns[x].peak_gamma = mat_op.average(self.im_mat, self.columns[x].x +
                                                                                       self.r, self.columns[x].y +
                                                                                       self.r, self.r)

            self.im_mat = mat_op.gen_de_framed_mat(self.im_mat, self.r)

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

    def redraw_search_mat(self):

        self.search_mat = self.im_mat
        if self.num_columns > 0:
            self.search_mat = mat_op.gen_framed_mat(self.search_mat, self.r + self.overhead)
            for i in range(0, self.num_columns):
                self.search_mat = mat_op.delete_pixels(self.search_mat, self.columns[i].x + self.r + self.overhead,
                                                       self.columns[i].y + self.r + self.overhead, self.r +
                                                       self.overhead)
            self.search_mat = mat_op.gen_de_framed_mat(self.search_mat, self.r + self.overhead)

    def redraw_circumference_mat(self):

        self.column_circumference_mat = np.zeros((self.N, self.M), dtype=type(self.im_mat))
        if self.num_columns > 0:
            self.column_circumference_mat = mat_op.gen_framed_mat(self.column_circumference_mat, self.r + self.overhead)
            for x in range(0, self.num_columns):
                self.column_circumference_mat = mat_op.draw_circle(self.column_circumference_mat, self.columns[x].x +
                                                                   self.r + self.overhead, self.columns[x].y + self.r +
                                                                   self.overhead, self.r)
            self.column_circumference_mat = mat_op.gen_de_framed_mat(self.column_circumference_mat, self.r +
                                                                     self.overhead)

    def redraw_centre_mat(self):

        self.column_centre_mat = np.zeros((self.N, self.M, 2), dtype=type(self.im_mat))
        if self.num_columns > 0:
            for x in range(0, self.num_columns):
                self.column_centre_mat[self.columns[x].y, self.columns[x].x, 0] = 1
                self.column_centre_mat[self.columns[x].y, self.columns[x].x, 1] = x

    def find_shape(self, i, j, clockwise=True):

        closed = False
        start_index = i
        shape_indices = np.ndarray([2], dtype=int)
        shape_indices[0] = i
        shape_indices[1] = j

        while not closed:

            i = shape_indices[shape_indices.shape[0] - 2]
            j = shape_indices[shape_indices.shape[0] - 1]

            if j == start_index or shape_indices.shape[0] > 7:

                closed = True

            else:

                next_index = -1

                if not self.test_reciprocality(i, j):

                    if clockwise:
                        sorted_indices, alpha = self.clockwise_neighbour_sort(j, j=i)
                    else:
                        sorted_indices, alpha = self.anticlockwise_neighbour_sort(j, j=i)

                    next_index = self.columns[j].n()

                else:

                    if clockwise:
                        sorted_indices, alpha = self.clockwise_neighbour_sort(j)
                    else:
                        sorted_indices, alpha = self.anticlockwise_neighbour_sort(j)

                    for x in range(0, self.columns[j].n()):

                        if sorted_indices[x] == i:

                            if x == 0:
                                next_index = self.columns[j].n() - 1
                            else:
                                next_index = x - 1

                next_index = sorted_indices[next_index]

                shape_indices = np.append(shape_indices, next_index)

        return shape_indices, shape_indices.shape[0] - 1

    def clockwise_neighbour_sort(self, i, j=-1):

        n = self.columns[i].n()
        print('n: ' + str(n))

        if not j == -1:
            n = self.columns[i].n() + 1
            print('changed n: ' + str(n))

        a = np.ndarray([n], dtype=np.int)
        b = np.ndarray([n], dtype=np.int)
        alpha = np.ndarray([n - 1], dtype=np.float64)
        indices = np.ndarray([n - 1], dtype=int)
        sorted_indices = np.ndarray([n], dtype=int)

        if not j == -1:

            sorted_indices[0] = j

            a[0] = self.columns[j].x - self.columns[i].x
            b[0] = self.columns[j].y - self.columns[i].y

            for x in range(1, n):
                a[x] = self.columns[self.columns[i].neighbour_indices[x - 1]].x - self.columns[i].x
                b[x] = self.columns[self.columns[i].neighbour_indices[x - 1]].y - self.columns[i].y

            for x in range(0, n - 1):
                indices[x] = self.columns[i].neighbour_indices[x]

            for x in range(1, n):

                alpha[x - 1] = Utilities.find_angle(a[0], a[x], b[0], b[x])

                if Utilities.vector_cross_product_magnitude(a[0], a[x], b[0], b[x]) < 0:
                    alpha[x - 1] = 2 * np.pi - alpha[x - 1]

            alpha, indices = Utilities.dual_sort(alpha, indices)

            for x in range(0, n - 1):
                sorted_indices[x + 1] = indices[x]

            alpha = np.append(alpha, 2 * np.pi)

        else:

            sorted_indices[0] = self.columns[i].neighbour_indices[0]

            for x in range(1, n):
                indices[x - 1] = self.columns[i].neighbour_indices[x]

            for x in range(0, n):
                a[x] = self.columns[self.columns[i].neighbour_indices[x]].x - self.columns[i].x
                b[x] = self.columns[self.columns[i].neighbour_indices[x]].y - self.columns[i].y

            for x in range(1, n):

                indices[x - 1] = self.columns[i].neighbour_indices[x]

                alpha[x - 1] = Utilities.find_angle(a[0], a[x], b[0], b[x])

                if Utilities.vector_cross_product_magnitude(a[0], a[x], b[0], b[x]) < 0:

                    alpha[x - 1] = 2 * np.pi - alpha[x - 1]

            alpha, indices = Utilities.dual_sort(alpha, indices)

            for x in range(0, n - 1):

                sorted_indices[x + 1] = indices[x]

            alpha = np.append(alpha, 2 * np.pi)

        return sorted_indices, alpha

    def anticlockwise_neighbour_sort(self, i, j=-1):

        n = self.columns[i].n()
        print('n: ' + str(n))

        if not j == -1:
            n = self.columns[i].n() + 1
            print('changed n: ' + str(n))

        a = np.ndarray([n], dtype=np.int)
        b = np.ndarray([n], dtype=np.int)
        alpha = np.ndarray([n - 1], dtype=np.float64)
        indices = np.ndarray([n - 1], dtype=int)
        sorted_indices = np.ndarray([n], dtype=int)

        if not j == -1:

            sorted_indices[0] = j

            a[0] = self.columns[j].x - self.columns[i].x
            b[0] = self.columns[j].y - self.columns[i].y

            for x in range(1, n):
                a[x] = self.columns[self.columns[i].neighbour_indices[x - 1]].x - self.columns[i].x
                b[x] = self.columns[self.columns[i].neighbour_indices[x - 1]].y - self.columns[i].y

            for x in range(0, n - 1):
                indices[x] = self.columns[i].neighbour_indices[x]

            for x in range(1, n):

                alpha[x - 1] = Utilities.find_angle(a[0], a[x], b[0], b[x])

                if Utilities.vector_cross_product_magnitude(a[0], a[x], b[0], b[x]) > 0:
                    alpha[x - 1] = 2 * np.pi - alpha[x - 1]

            alpha, indices = Utilities.dual_sort(alpha, indices)

            for x in range(0, n - 1):
                sorted_indices[x + 1] = indices[x]

            alpha = np.append(alpha, 2 * np.pi)

        else:

            sorted_indices[0] = self.columns[i].neighbour_indices[0]

            for x in range(1, n):
                indices[x - 1] = self.columns[i].neighbour_indices[x]

            for x in range(0, n):
                a[x] = self.columns[self.columns[i].neighbour_indices[x]].x - self.columns[i].x
                b[x] = self.columns[self.columns[i].neighbour_indices[x]].y - self.columns[i].y

            for x in range(1, n):

                indices[x - 1] = self.columns[i].neighbour_indices[x]

                alpha[x - 1] = Utilities.find_angle(a[0], a[x], b[0], b[x])

                if Utilities.vector_cross_product_magnitude(a[0], a[x], b[0], b[x]) > 0:

                    alpha[x - 1] = 2 * np.pi - alpha[x - 1]

            alpha, indices = Utilities.dual_sort(alpha, indices)

            for x in range(0, n - 1):

                sorted_indices[x + 1] = indices[x]

            alpha = np.append(alpha, 2 * np.pi)

        return sorted_indices, alpha

    def define_levels(self, i, level=0):

        self.reset_all_flags()

        self.mesh_levels(i, level)

        complete = False
        emer_abort = False
        overcounter = 0
        neighbour_level = 0

        while not complete and not emer_abort:

            found = False
            counter = 0

            while counter < self.num_columns:

                if self.columns[counter].is_in_precipitate and not self.columns[counter].flag_1:

                    x = 0

                    while x <= neighbour_level:

                        if self.columns[self.columns[counter].neighbour_indices[x]].is_in_precipitate and\
                                        self.columns[self.columns[counter].neighbour_indices[x]].flag_1:

                            neighbour = self.columns[counter].neighbour_indices[x]
                            if self.columns[neighbour].level == 0:
                                self.columns[counter].level = 1
                            else:
                                self.columns[counter].level = 0
                            self.columns[counter].flag_1 = True
                            found = True

                        x = x + 1

                counter = counter + 1

            complete = True

            for y in range(0, self.num_columns):

                if self.columns[y].is_in_precipitate and not self.columns[y].flag_1:

                    complete = False

            if found and neighbour_level > 0:

                neighbour_level = neighbour_level - 1

            if not found and neighbour_level < 2:

                neighbour_level = neighbour_level + 1

            overcounter = overcounter + 1
            if overcounter > 100:

                emer_abort = True
                print('Emergency abort')

            print(neighbour_level)

        self.reset_all_flags()

    def mesh_levels(self, i, level):

        if self.columns[i].is_in_precipitate:

            self.columns[i].flag_1 = True
            self.set_level(i, level)

        else:

            self.columns[i].flag_1 = True

            next_level = 0
            if level == 0:
                next_level = 1
            elif level == 1:
                next_level = 0
            else:
                print('Disaster!')

            self.set_level(i, level)

            indices = self.columns[i].neighbour_indices

            for x in range(0, self.columns[i].n()):

                reciprocal = self.test_reciprocality(i, indices[x])

                if not self.columns[indices[x]].flag_1 and not self.columns[i].is_edge_column and reciprocal:

                    self.mesh_levels(indices[x], next_level)

    def precipitate_levels(self, i, level):

        if not self.columns[i].is_in_precipitate:

            self.columns[i].flag_1 = True

        else:

            self.columns[i].flag_1 = True

            next_level = 0
            if level == 0:
                next_level = 1
            elif level == 1:
                next_level = 0
            else:
                print('Disaster!')

            self.set_level(i, level)

            indices = self.columns[i].neighbour_indices

            complete = False
            counter_1 = 0
            counter_2 = 0

            while not complete:

                if not self.columns[indices[counter_1]].flag_1:

                    if self.test_reciprocality(i, indices[counter_1]):

                        self.precipitate_levels(indices[counter_1], next_level)
                        counter_1 = counter_1 + 1
                        counter_2 = counter_2 + 1

                    else:

                        counter_1 = counter_1 + 1

                else:

                    counter_1 = counter_1 + 1

                if counter_2 == self.columns[i].n() - 2 or counter_1 == self.columns[i].n() - 2:

                    complete = True

    def set_level(self, i, level):

        previous_level = self.columns[i].level
        self.columns[i].level = level

        if level == previous_level:
            return False
        else:
            return True

    def invert_levels(self, only_precipitate=False):

        if self.num_columns > 0:
            for x in range(0, self.num_columns):
                if only_precipitate:
                    if self.columns[x].is_in_precipitate:
                        if self.columns[x].level == 0:
                            self.columns[x].level = 1
                        else:
                            self.columns[x].level = 0
                    else:
                        pass
                else:
                    if self.columns[x].level == 0:
                        self.columns[x].level = 1
                    else:
                        self.columns[x].level = 0

    def test_reciprocality(self, i, j):

        found = False

        for x in range(0, self.columns[j].n()):

            if self.columns[j].neighbour_indices[x] == i:

                found = True

        return found

    def delete_columns(self):

        dummy_instance = AtomicColumn(0, 1, 1, 1, 1, 1, SuchSoftware.num_poss)
        self.columns = np.ndarray([1], dtype=type(dummy_instance))
        self.num_columns = 0
        self.redraw_centre_mat()
        self.redraw_circumference_mat()
        self.redraw_search_mat()
        self.summarize_stats()

    # Save the entire class instance to file
    def save(self, filename_full):
        with open(filename_full, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    # Load an instance from file
    @staticmethod
    def load(filename_full):
        with open(filename_full, 'rb') as f:
            obj = pickle.load(f)
            return obj

    def summarize_stats(self):

        self.avg_peak_gamma = 0
        self.avg_avg_gamma = 0

        self.num_inconsistencies = 0
        self.num_popular = 0
        self.num_unpopular = 0
        self.chi = 0.0

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

        self.num_si = 0
        self.num_cu = 0
        self.num_zn = 0
        self.num_al = 0
        self.num_ag = 0
        self.num_mg = 0
        self.num_un = 0

        self.num_precipitate_si = 0
        self.num_precipitate_cu = 0
        self.num_precipitate_zn = 0
        self.num_precipitate_al = 0
        self.num_precipitate_ag = 0
        self.num_precipitate_mg = 0
        self.num_precipitate_un = 0

        self.num_precipitate_columns = 0

        if self.num_columns > 0:

            for x in range(0, self.num_columns):

                self.avg_peak_gamma = self.avg_peak_gamma + self.columns[x].peak_gamma
                self.avg_avg_gamma = self.avg_avg_gamma + self.columns[x].avg_gamma

                if self.columns[x].is_unpopular and not self.columns[x].is_edge_column:
                    self.num_unpopular = self.num_unpopular + 1
                    self.num_inconsistencies = self.num_inconsistencies + 1

                if self.columns[x].is_popular and not self.columns[x].is_edge_column:
                    self.num_popular = self.num_popular + 1
                    self.num_inconsistencies = self.num_inconsistencies + 1

                if self.columns[x].neighbour_indices is not None and not self.columns[x].is_edge_column:
                    for y in range(0, self.columns[x].n()):
                        if self.columns[x].level == self.columns[self.columns[x].neighbour_indices[y]].level:
                            self.num_inconsistencies = self.num_inconsistencies + 1

                if self.columns[x].h_index == 0:
                    self.num_si = self.num_si + 1
                    self.number_percentage_si = self.number_percentage_si + 1
                    self.avg_si_peak_gamma = self.avg_si_peak_gamma + self.columns[x].peak_gamma
                    self.avg_si_avg_gamma = self.avg_si_avg_gamma + self.columns[x].avg_gamma
                    if self.columns[x].is_in_precipitate:
                        self.num_precipitate_columns = self.num_precipitate_columns + 1
                        self.num_precipitate_si = self.num_precipitate_si + 1
                        self.precipitate_number_percentage_si = self.precipitate_number_percentage_si + 1
                elif self.columns[x].h_index == 1:
                    self.num_cu = self.num_cu + 1
                    self.number_percentage_cu = self.number_percentage_cu + 1
                    self.avg_cu_peak_gamma = self.avg_cu_peak_gamma + self.columns[x].peak_gamma
                    self.avg_cu_avg_gamma = self.avg_cu_avg_gamma + self.columns[x].avg_gamma
                    if self.columns[x].is_in_precipitate:
                        self.num_precipitate_columns = self.num_precipitate_columns + 1
                        self.num_precipitate_cu = self.num_precipitate_cu + 1
                        self.precipitate_number_percentage_cu = self.precipitate_number_percentage_cu + 1
                elif self.columns[x].h_index == 2:
                    self.num_zn = self.num_zn + 1
                    self.number_percentage_zn = self.number_percentage_zn + 1
                    self.avg_zn_peak_gamma = self.avg_zn_peak_gamma + self.columns[x].peak_gamma
                    self.avg_zn_avg_gamma = self.avg_zn_avg_gamma + self.columns[x].avg_gamma
                    if self.columns[x].is_in_precipitate:
                        self.num_precipitate_columns = self.num_precipitate_columns + 1
                        self.num_precipitate_zn = self.num_precipitate_zn + 1
                        self.precipitate_number_percentage_zn = self.precipitate_number_percentage_zn + 1
                elif self.columns[x].h_index == 3:
                    self.num_al = self.num_al + 1
                    self.number_percentage_al = self.number_percentage_al + 1
                    self.avg_al_peak_gamma = self.avg_al_peak_gamma + self.columns[x].peak_gamma
                    self.avg_al_avg_gamma = self.avg_al_avg_gamma + self.columns[x].avg_gamma
                    if self.columns[x].is_in_precipitate:
                        self.num_precipitate_columns = self.num_precipitate_columns + 1
                        self.num_precipitate_al = self.num_precipitate_al + 1
                        self.precipitate_number_percentage_al = self.precipitate_number_percentage_al + 1
                elif self.columns[x].h_index == 4:
                    self.num_ag = self.num_ag + 1
                    self.number_percentage_ag = self.number_percentage_ag + 1
                    self.avg_ag_peak_gamma = self.avg_ag_peak_gamma + self.columns[x].peak_gamma
                    self.avg_ag_avg_gamma = self.avg_ag_avg_gamma + self.columns[x].avg_gamma
                    if self.columns[x].is_in_precipitate:
                        self.num_precipitate_columns = self.num_precipitate_columns + 1
                        self.num_precipitate_ag = self.num_precipitate_ag + 1
                        self.precipitate_number_percentage_ag = self.precipitate_number_percentage_ag + 1
                elif self.columns[x].h_index == 5:
                    self.num_mg = self.num_mg + 1
                    self.number_percentage_mg = self.number_percentage_mg + 1
                    self.avg_mg_peak_gamma = self.avg_mg_peak_gamma + self.columns[x].peak_gamma
                    self.avg_mg_avg_gamma = self.avg_mg_avg_gamma + self.columns[x].avg_gamma
                    if self.columns[x].is_in_precipitate:
                        self.num_precipitate_columns = self.num_precipitate_columns + 1
                        self.num_precipitate_mg = self.num_precipitate_mg + 1
                        self.precipitate_number_percentage_mg = self.precipitate_number_percentage_mg + 1
                elif self.columns[x].h_index == 6:
                    self.num_un = self.num_un + 1
                    self.number_percentage_un = self.number_percentage_un + 1
                    self.avg_un_peak_gamma = self.avg_un_peak_gamma + self.columns[x].peak_gamma
                    self.avg_un_avg_gamma = self.avg_un_avg_gamma + self.columns[x].avg_gamma
                    if self.columns[x].is_in_precipitate:
                        self.num_precipitate_columns = self.num_precipitate_columns + 1
                        self.num_precipitate_un = self.num_precipitate_un + 1
                        self.precipitate_number_percentage_un = self.precipitate_number_percentage_un + 1
                else:
                    print('Error in summarize_stats()')

            self.avg_peak_gamma = self.avg_peak_gamma / self.num_columns
            self.avg_avg_gamma = self.avg_avg_gamma / self.num_columns

            if not self.num_si == 0:
                self.avg_si_peak_gamma = self.avg_si_peak_gamma / self.num_si
                self.avg_si_avg_gamma = self.avg_si_avg_gamma / self.num_si

            if not self.num_cu == 0:
                self.avg_cu_peak_gamma = self.avg_cu_peak_gamma / self.num_cu
                self.avg_cu_avg_gamma = self.avg_cu_avg_gamma / self.num_cu

            if not self.num_zn == 0:
                self.avg_zn_peak_gamma = self.avg_zn_peak_gamma / self.num_zn
                self.avg_zn_avg_gamma = self.avg_zn_avg_gamma / self.num_zn

            if not self.num_al == 0:
                self.avg_al_peak_gamma = self.avg_al_peak_gamma / self.num_al
                self.avg_al_avg_gamma = self.avg_al_avg_gamma / self.num_al

            if not self.num_ag == 0:
                self.avg_ag_peak_gamma = self.avg_ag_peak_gamma / self.num_ag
                self.avg_ag_avg_gamma = self.avg_ag_avg_gamma / self.num_ag

            if not self.num_mg == 0:
                self.avg_mg_peak_gamma = self.avg_mg_peak_gamma / self.num_mg
                self.avg_mg_avg_gamma = self.avg_mg_avg_gamma / self.num_mg

            if not self.num_un == 0:
                self.avg_un_peak_gamma = self.avg_un_peak_gamma / self.num_un
                self.avg_un_avg_gamma = self.avg_un_avg_gamma / self.num_un

            self.number_percentage_si = self.number_percentage_si / self.num_columns
            self.number_percentage_cu = self.number_percentage_cu / self.num_columns
            self.number_percentage_zn = self.number_percentage_zn / self.num_columns
            self.number_percentage_al = self.number_percentage_al / self.num_columns
            self.number_percentage_ag = self.number_percentage_ag / self.num_columns
            self.number_percentage_mg = self.number_percentage_mg / self.num_columns
            self.number_percentage_un = self.number_percentage_un / self.num_columns

            if not self.num_precipitate_columns == 0:
                self.precipitate_number_percentage_si = self.precipitate_number_percentage_si / self.num_precipitate_columns
                self.precipitate_number_percentage_cu = self.precipitate_number_percentage_cu / self.num_precipitate_columns
                self.precipitate_number_percentage_zn = self.precipitate_number_percentage_zn / self.num_precipitate_columns
                self.precipitate_number_percentage_al = self.precipitate_number_percentage_al / self.num_precipitate_columns
                self.precipitate_number_percentage_ag = self.precipitate_number_percentage_ag / self.num_precipitate_columns
                self.precipitate_number_percentage_mg = self.precipitate_number_percentage_mg / self.num_precipitate_columns
                self.precipitate_number_percentage_un = self.precipitate_number_percentage_un / self.num_precipitate_columns

        self.chi = self.num_inconsistencies / self.num_columns

        self.build_stat_string()
        self.build_data_string()

    def build_stat_string(self):

        self.display_stats_string = ('Statistics:\n\n'
            'Number of detected columns: ' + str(self.num_columns) + '\n'
            'Number of detected precipitate columns: ' + str(self.num_precipitate_columns) + '\n\n'
            'Number of inconsistencies: ' + str(self.num_inconsistencies) + '\n'
            'Number of popular: ' + str(self.num_popular) + '\n'
            'Number of unpopular: ' + str(self.num_unpopular) + '\n'
            'Chi: ' + str(self.chi) + '\n\n'
            'Average peak intensity: ' + str(self.avg_peak_gamma) + '\n'
            'Average average intensity: ' + str(self.avg_avg_gamma) + '\n\n'
            'Average Si peak intensity: ' + str(self.avg_si_peak_gamma) + '\n'
            'Average Cu peak intensity: ' + str(self.avg_cu_peak_gamma) + '\n'
            'Average Zn peak intensity: ' + str(self.avg_zn_peak_gamma) + '\n'
            'Average Al peak intensity: ' + str(self.avg_al_peak_gamma) + '\n'
            'Average Ag peak intensity: ' + str(self.avg_ag_peak_gamma) + '\n'
            'Average Mg peak intensity: ' + str(self.avg_mg_peak_gamma) + '\n'
            'Average Un peak intensity: ' + str(self.avg_un_peak_gamma) + '\n\n'
            'Average Si average intensity: ' + str(self.avg_si_avg_gamma) + '\n'
            'Average Cu average intensity: ' + str(self.avg_cu_avg_gamma) + '\n'
            'Average Zn average intensity: ' + str(self.avg_zn_avg_gamma) + '\n'
            'Average Al average intensity: ' + str(self.avg_al_avg_gamma) + '\n'
            'Average Ag average intensity: ' + str(self.avg_ag_avg_gamma) + '\n'
            'Average Mg average intensity: ' + str(self.avg_mg_avg_gamma) + '\n'
            'Average Un average intensity: ' + str(self.avg_un_avg_gamma) + '\n\n'
            'Number of Si-columns: ' + str(self.num_si) + '\n'
            'Number of Cu-columns: ' + str(self.num_cu) + '\n'
            'Number of Zn-columns: ' + str(self.num_zn) + '\n'
            'Number of Al-columns: ' + str(self.num_al) + '\n'
            'Number of Ag-columns: ' + str(self.num_ag) + '\n'
            'Number of Mg-columns: ' + str(self.num_mg) + '\n'
            'Number of Un-columns: ' + str(self.num_un) + '\n\n'
            'Number procentage of Si: ' + str(self.number_percentage_si) + '\n'
            'Number procentage of Cu: ' + str(self.number_percentage_cu) + '\n'
            'Number procentage of Zn: ' + str(self.number_percentage_zn) + '\n'
            'Number procentage of Al: ' + str(self.number_percentage_al) + '\n'
            'Number procentage of Ag: ' + str(self.number_percentage_ag) + '\n'
            'Number procentage of Mg: ' + str(self.number_percentage_mg) + '\n'
            'Number procentage of Un: ' + str(self.number_percentage_un) + '\n\n'
            'Number of precipitate Si-columns: ' + str(self.num_precipitate_si) + '\n'
            'Number of precipitate Cu-columns: ' + str(self.num_precipitate_cu) + '\n'
            'Number of precipitate Zn-columns: ' + str(self.num_precipitate_zn) + '\n'
            'Number of precipitate Al-columns: ' + str(self.num_precipitate_al) + '\n'
            'Number of precipitate Ag-columns: ' + str(self.num_precipitate_ag) + '\n'
            'Number of precipitate Mg-columns: ' + str(self.num_precipitate_mg) + '\n'
            'Number of precipitate Un-columns: ' + str(self.num_precipitate_un) + '\n\n'
            'Number procentage of precipitate Si: ' + str(self.precipitate_number_percentage_si) + '\n'
            'Number procentage of precipitate Cu: ' + str(self.precipitate_number_percentage_cu) + '\n'
            'Number procentage of precipitate Zn: ' + str(self.precipitate_number_percentage_zn) + '\n'
            'Number procentage of precipitate Al: ' + str(self.precipitate_number_percentage_al) + '\n'
            'Number procentage of precipitate Ag: ' + str(self.precipitate_number_percentage_ag) + '\n'
            'Number procentage of precipitate Mg: ' + str(self.precipitate_number_percentage_mg) + '\n'
            'Number procentage of precipitate Un: ' + str(self.precipitate_number_percentage_un) + '\n\n')

    def build_data_string(self):

        pass

