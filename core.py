# This file contains the SuchSoftware class that is the algorithms.
import numpy as np
import dm3_lib as dm3
import mat_op
import graph
import utils
import graph_op
import sys
import pickle
import compatibility
import legacy_items
from matplotlib import pyplot as plt
import weak_untangling
import strong_untangling
import dev_module
import logging

# Instantiate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SuchSoftware:

    # Version
    version = [0, 0, 5]

    # Number of elements in the probability vectors
    num_selections = 7

    # Number of closest neighbours that are included in local search-spaces
    map_size = 8

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

    # Indexable list of atomic radii
    atomic_radii = (si_radii, cu_radii, zn_radii, al_radii, ag_radii, mg_radii, un_radii)

    # Relative mean peak intensities for the different implemented alloys:
    intensities_0 = [0.44, 0.88, 0.00, 0.40, 0.00, 0.33, 0.00]
    intensities_1 = [0.70, 0.00, 0.00, 0.67, 0.00, 0.49, 0.00]

    # Indexable list
    intensities = [intensities_0, intensities_1]

    # Indexable species strings
    species_strings = ['Si', 'Cu', 'Zn', 'Al', 'Ag', 'Mg', 'Un']

    # Constructor
    def __init__(self, filename_full, debug_obj=None):

        self.filename_full = filename_full
        self.im_mat = None
        self.scale = 1
        self.im_height = 0
        self.im_width = 0
        self.version_saved = None
        self.starting_index = None

        # For communicating with the interface, if any:
        self.debug_obj = debug_obj
        self.debug_mode = False

        # Alloy info: This vector is used to multiply away elements in the AtomicColumn.prob_vector that are not in
        # the alloy being studied. Currently supported alloys are:
        # self.alloy = alloy
        # 0 = Al-Si-Mg-Cu
        # 1 = Al-Si-Mg
        self.alloy = 0
        self.alloy_mat = np.ndarray([SuchSoftware.num_selections], dtype=int)
        self.set_alloy_mat()

        if not (filename_full == 'Empty' or filename_full == 'empty'):
            dm3f = dm3.DM3(self.filename_full)
            self.im_mat = dm3f.imagedata
            (self.scale, junk) = dm3f.pxsize
            self.scale = 1000 * self.scale
            self.im_mat = mat_op.normalize_static(self.im_mat)
            (self.im_height, self.im_width) = self.im_mat.shape
            self.fft_im_mat = mat_op.gen_fft(self.im_mat)

        # Data matrices: These hold much of the information gathered by the different algorithms
        self.search_mat = self.im_mat
        self.column_centre_mat = np.zeros((self.im_height, self.im_width, 2), dtype=type(self.im_mat))
        self.column_circumference_mat = np.zeros((self.im_height, self.im_width), dtype=type(self.im_mat))

        # Counting and statistical variables
        self.num_columns = 0
        self.num_precipitate_columns = 0

        self.num_inconsistencies = 0
        self.num_popular = 0
        self.num_unpopular = 0

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

        self.stats_string = 'Empty'
        self.export_data_string = ' '

        # These are hyper-parameters of the algorithms. See the documentation.
        self.threshold = 0.2586
        self.search_size = 1
        self.r = int(100 / self.scale)
        self.certainty_threshold = 0.8
        self.overhead = int(6 * (self.r / 10))

        # Initialize an empty graph
        self.graph = graph.AtomicGraph(map_size=self.map_size)

        logger.info('Generated instance from {}'.format(filename_full))

        # DEPRECATED:
        self.dist_1_std = 0
        self.dist_2_std = 0
        self.dist_3_std = 0
        self.dist_4_std = 0
        self.dist_5_std = 0
        self.dist_8_std = 0

    def alloy_string(self):
        if self.alloy == 0:
            return 'Alloy: Al-Mg-Si-(Cu)'
        elif self.alloy == 1:
            return 'Alloy: Al-Mg-Si'
        else:
            return 'Alloy: Unknown'

    def stats_summary(self):
        self.summarize_stats()
        string = 'Image summary: ----------\n'
        for line in iter(self.stats_string.splitlines()):
            string += '    ' + line + '\n'
        logger.info(string)

    def vertex_report(self, i):
        vertex = self.graph.vertices[i]
        alpha_max, alpha_min = graph_op.base_angle_score(self.graph, i, apply=False)
        rotation_map, angles, variance = self.graph.calc_central_angle_variance(i)
        string = 'Vertex summary: ----------\n' + \
                 '    Index: {}\n'.format(vertex.i) + \
                 '    Image pos: ({}, {})\n'.format(vertex.im_coor_x, vertex.im_coor_y) + \
                 '    Real pos: ({}, {})\n'.format(vertex.real_coor_x, vertex.real_coor_y) + \
                 '    Atomic Species: {}\n'.format(vertex.species()) + \
                 '    Probability vector: {}\n'.format(vertex.prob_vector).replace('\n', '') + \
                 '    Partner vector: {}\n'.format(vertex.partners()) + \
                 '    Alpha max: {}\n'.format(alpha_max) + \
                 '    Alpha min: {}\n'.format(alpha_min) + \
                 '    Central angle variance = {}\n'.format(variance) + \
                 '    Rotation map: {}\n'.format(str(rotation_map)) + \
                 '    Central angles: {}\n'.format(str(angles)) + \
                 '    Is edge column: {}\n'.format(vertex.is_edge_column) + \
                 '    Level: {}\n'.format(vertex.level) + \
                 '    Anti-level: {}\n'.format(vertex.anti_level()) + \
                 '    Flag 1: {}\n'.format(vertex.flag_1) + \
                 '    Flag 2: {}\n'.format(vertex.flag_2) + \
                 '    Flag 3: {}\n'.format(vertex.flag_3) + \
                 '    Flag 4: {}'.format(vertex.flag_4)
        logger.info(string)

    def set_alloy_mat(self):
        if self.alloy == 0:
            for x in range(0, SuchSoftware.num_selections):
                if x == 2 or x == 4:
                    self.alloy_mat[x] = 0
                else:
                    self.alloy_mat[x] = 1
        elif self.alloy == 1:
            for x in range(0, SuchSoftware.num_selections):
                if x == 2 or x == 4 or x == 1:
                    self.alloy_mat[x] = 0
                else:
                    self.alloy_mat[x] = 1
        else:
            logger.error('Could not set alloy vector. Unknown alloy')

    def save(self, filename_full):
        with open(filename_full, 'wb') as f:
            self.debug_obj = None
            self.version_saved = self.version
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        logger.info('Saved {}'.format(filename_full))

    def export(self, format_, filename, column_centered=True):
        """Export data to csv file."""
        if column_centered:
            print(format_)
            print(filename)
        else:
            print(format_)
            print(filename)


    @staticmethod
    def load(filename_full):
        with open(filename_full, 'rb') as f:
            try:
                obj = pickle.load(f)
            except:
                obj = None
                logger.error('Failed to load save-file!')
            else:
                if not obj.version_saved == SuchSoftware.version:
                    logger.info('Attempted to load un-compatible save-file. Running conversion script...')
                    obj = compatibility.convert(obj, obj.version_saved, SuchSoftware.version)
                    if obj is None:
                        logger.error('Conversion unsuccessful!')
                        logger.error('Failed to load save-file!')
                    else:
                        logger.info('Conversion successful!')
                        logger.info('Loaded {}'.format(filename_full))
                else:
                    logger.info('Loaded {}'.format(filename_full))
            return obj

    def column_detection(self, search_type='s'):
        if self.num_columns == 0:
            logger.info('Starting column detection. Search mode is \'{}\''.format(search_type))
        else:
            logger.info('Continuing column detection. Search mode is \'{}\''.format(search_type))
        cont = True
        counter = self.num_columns
        self.column_circumference_mat = mat_op.gen_framed_mat(self.column_circumference_mat, self.r + self.overhead)
        self.search_mat = mat_op.gen_framed_mat(self.search_mat, self.r + self.overhead)
        self.im_mat = mat_op.gen_framed_mat(self.im_mat, self.r + self.overhead)

        while cont:

            pos = np.unravel_index(self.search_mat.argmax(),
                                   (self.im_height + 2 * (self.r + self.overhead),
                                    self.im_width + 2 * (self.r + self.overhead)))
            max_val = self.search_mat[pos]
            x_fit, y_fit = utils.cm_fit(self.im_mat, pos[1], pos[0], self.r)

            x_fit_real_coor = x_fit - self.r - self.overhead
            y_fit_real_coor = y_fit - self.r - self.overhead
            x_fit_real_coor_pix = int(np.floor(x_fit_real_coor))
            y_fit_real_coor_pix = int(np.floor(y_fit_real_coor))
            x_fit_pix = int(np.floor(x_fit))
            y_fit_pix = int(np.floor(y_fit))

            self.search_mat = mat_op.delete_pixels(self.search_mat, x_fit_pix, y_fit_pix, self.r + self.overhead)

            vertex = graph.Vertex(counter, x_fit_real_coor, y_fit_real_coor, self.r, max_val, 0,
                                  self.alloy_mat,
                                  num_selections=SuchSoftware.num_selections,
                                  species_strings=SuchSoftware.species_strings,
                                  certainty_threshold=self.certainty_threshold,
                                  scale=self.scale)
            vertex.reset_prob_vector(bias=6)
            self.graph.add_vertex(vertex)

            self.column_centre_mat[y_fit_real_coor_pix, x_fit_real_coor_pix, 0] = 1
            self.column_centre_mat[y_fit_real_coor_pix, x_fit_real_coor_pix, 1] = counter
            self.column_circumference_mat = mat_op.draw_circle(self.column_circumference_mat, x_fit_pix, y_fit_pix,
                                                               self.r)

            logger.debug(str(counter) + ': (' + str(x_fit_real_coor) + ', ' + str(y_fit_real_coor) + ') | (' + str(
                pos[1]) + ', ' + str(pos[0]) + ')')

            self.num_columns += 1
            self.num_un += 1
            counter += 1

            if search_type == 's':
                if counter >= self.search_size:
                    cont = False
            elif search_type == 't':
                if np.max(self.search_mat) < self.threshold:
                    cont = False
            else:
                logger.error('Invalid search_type')

        self.column_circumference_mat = mat_op.gen_de_framed_mat(self.column_circumference_mat, self.r + self.overhead)
        self.search_mat = mat_op.gen_de_framed_mat(self.search_mat, self.r + self.overhead)
        self.im_mat = mat_op.gen_de_framed_mat(self.im_mat, self.r + self.overhead)
        self.calc_avg_gamma()
        self.summarize_stats()
        logger.info('Column detection complete! Found {} columns.'.format(self.num_columns))

    def find_nearest(self, i, n, weight=4):

        x_0 = self.graph.vertices[i].im_coor_x
        y_0 = self.graph.vertices[i].im_coor_y

        indices = np.ndarray([n], dtype=np.int)
        distances = np.ndarray([n], dtype=np.float64)
        num_found = 0
        total_num = 0

        for x in range(x_0 - weight * self.r, x_0 + weight * self.r):
            for y in range(y_0 - weight * self.r, y_0 + weight * self.r):

                if 0 <= x < self.im_width and 0 <= y < self.im_height and not (x == x_0 and y == y_0):

                    if self.column_centre_mat[y, x, 0] == 1:
                        j = self.column_centre_mat[y, x, 1]
                        dist = self.graph.spatial_distance(i, j)
                        if num_found >= n:
                            if dist < distances.max():
                                ind = distances.argmax()
                                indices[ind] = j
                                distances[ind] = dist
                        else:
                            indices[num_found] = j
                            distances[num_found] = dist
                            num_found += 1
                        total_num += 1

        if num_found < n:

            indices, distances = self.find_nearest(i, n, weight=2 * weight)
            logger.debug('        Did not find enough neighbours for vertex {}. increasing search area.'.format(i))

        else:

            logger.debug('        Found {} total neighbours for vertex {}'.format(total_num, i))

            # Use built-in sort instead of this home-made shit:
            temp_indices = np.ndarray([n], dtype=np.int)
            temp_distances = np.ndarray([n], dtype=np.float64)

            for k in range(0, n):

                ind = distances.argmin()
                temp_indices[k] = indices[ind]
                temp_distances[k] = distances[ind]
                distances[ind] = distances.max() + k + 1

            indices = temp_indices
            distances = temp_distances

        return list(indices), list(distances)

    def find_edge_columns(self):

        for y in range(0, self.num_columns):

            x_coor = self.graph.vertices[y].real_coor_x
            y_coor = self.graph.vertices[y].real_coor_y
            margin = 6 * self.r

            if x_coor < margin or x_coor > self.im_width - margin - 1 or y_coor < margin or y_coor > self.im_height - margin - 1:

                self.graph.vertices[y].is_edge_column = True
                self.graph.vertices[y].reset_prob_vector(bias=3)

            else:

                self.graph.vertices[y].is_edge_column = False

    def column_characterization(self, starting_index, search_type=0):

        sys.setrecursionlimit(10000)

        if search_type == 0:

            logger.info('Starting column characterization from vertex {}...'.format(starting_index))
            logger.info('Setting alloy...')
            self.set_alloy_mat()
            logger.info('Alloy set.')
            logger.info('Finding edge columns....')
            self.find_edge_columns()
            for i in range(0, self.num_columns):
                if self.graph.vertices[i].is_edge_column and not self.graph.vertices[i].set_by_user:
                    self.graph.vertices[i].reset_prob_vector(bias=3)
            logger.info('Found edge columns.')
            # Reset prob vectors:
            self.column_characterization(starting_index, search_type=12)
            # Spatial mapping:
            self.column_characterization(starting_index, search_type=2)
            # Angle analysis:
            self.column_characterization(starting_index, search_type=3)
            # Intensity analysis
            self.column_characterization(starting_index, search_type=4)
            # Find particle:
            self.column_characterization(starting_index, search_type=5)
            # Set levels:
            self.column_characterization(starting_index, search_type=6)
            # Add edges:
            self.column_characterization(starting_index, search_type=7)
            # Summarize:
            logger.info('Summarizing stats.')
            self.summarize_stats()
            # Legacy weak untanglng
            self.column_characterization(starting_index, search_type=8)
            # Legacy strong untangling
            self.column_characterization(starting_index, search_type=9)
            # Summarize:
            logger.info('Summarizing stats.')
            self.summarize_stats()
            # Complete:
            logger.info('Column characterization complete.')
            logger.info(' ')

        elif search_type == 1:

            logger.info('Starting column characterization from vertex {}...'.format(starting_index))
            logger.info('Setting alloy')
            self.set_alloy_mat()
            logger.info('Alloy set.')

            self.column_characterization(starting_index, search_type=2)

            self.column_characterization(starting_index, search_type=3)

            self.column_characterization(starting_index, search_type=4)

            self.column_characterization(starting_index, search_type=5)

            self.column_characterization(starting_index, search_type=6)

            self.column_characterization(starting_index, search_type=7)

            logger.info('Summarizing stats.')
            self.summarize_stats()

            self.column_characterization(starting_index, search_type=10)

            self.column_characterization(starting_index, search_type=11)

            logger.info('Summarizing stats.')
            self.summarize_stats()

            logger.info('Column characterization complete.')

        elif search_type == 2:

            logger.info('Mapping spatial locality...')
            self.redraw_centre_mat()
            self.redraw_circumference_mat()
            for i in range(0, self.num_columns):
                self.graph.vertices[i].neighbour_indices, _ = self.find_nearest(i, self.map_size)
            self.find_edge_columns()
            logger.info('Spatial mapping complete.')

        elif search_type == 3:

            logger.info('Analysing angles...')
            for i, vertex in enumerate(self.graph.vertices):
                if not vertex.set_by_user and not vertex.is_edge_column:
                    vertex.reset_prob_vector(bias=vertex.h_index)
                    graph_op.apply_angle_score(self.graph, i, self.dist_3_std, self.dist_4_std, self.dist_5_std,
                                               self.num_selections)

            logger.info('Angle analysis complete.')

        elif search_type == 4:

            logger.info('Analyzing intensities...')
            for i in range(0, self.num_columns):
                if not self.graph.vertices[i].set_by_user and not self.graph.vertices[i].is_edge_column:

                    graph_op.apply_intensity_score(self.graph, i, self.num_selections, self.intensities[self.alloy],
                                                   self.dist_8_std)
            logger.info('Intensity analysis complete.')

        elif search_type == 5:

            logger.info('Finding particle with legacy method....')
            legacy_items.precipitate_controller(self.graph, starting_index)
            # graph_op.precipitate_controller(self.graph, starting_index)
            logger.info('Found particle.')

        elif search_type == 6:

            logger.info('Running legacy level definition algorithm....')
            legacy_items.define_levels(self.graph, starting_index, self.graph.vertices[starting_index].level)
            logger.info('    Levels set.')

        elif search_type == 7:

            logger.info('Adding edges to graph...')
            self.graph.redraw_edges()
            logger.info('Edges added.')

        elif search_type == 8:

            logger.info('Starting legacy weak untangling...')
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_simple(self.graph, i, sub=True)
            self.graph.redraw_edges()
            self.column_characterization(starting_index, search_type=12)
            self.column_characterization(starting_index, search_type=4)
            self.column_characterization(starting_index, search_type=3)
            self.column_characterization(starting_index, search_type=5)
            self.column_characterization(starting_index, search_type=6)
            self.summarize_stats()
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_simple(self.graph, i)
            self.graph.redraw_edges()
            self.column_characterization(starting_index, search_type=12)
            self.column_characterization(starting_index, search_type=4)
            self.column_characterization(starting_index, search_type=3)
            self.column_characterization(starting_index, search_type=5)
            self.column_characterization(starting_index, search_type=6)
            self.summarize_stats()
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_advanced(self.graph, i)
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_advanced(self.graph, i)
            self.graph.redraw_edges()
            self.column_characterization(starting_index, search_type=12)
            self.column_characterization(starting_index, search_type=4)
            self.column_characterization(starting_index, search_type=3)
            self.column_characterization(starting_index, search_type=5)
            self.column_characterization(starting_index, search_type=6)
            self.summarize_stats()
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_advanced(self.graph, i)
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_advanced(self.graph, i)
            self.graph.redraw_edges()
            self.column_characterization(starting_index, search_type=12)
            self.column_characterization(starting_index, search_type=4)
            self.column_characterization(starting_index, search_type=3)
            self.column_characterization(starting_index, search_type=5)
            self.column_characterization(starting_index, search_type=6)
            self.summarize_stats()
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_advanced(self.graph, i, experimental=True)
            self.graph.redraw_edges()
            self.column_characterization(starting_index, search_type=12)
            self.column_characterization(starting_index, search_type=4)
            self.column_characterization(starting_index, search_type=3)
            self.column_characterization(starting_index, search_type=5)
            self.column_characterization(starting_index, search_type=6)
            self.summarize_stats()
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_advanced(self.graph, i)
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_advanced(self.graph, i)
            self.graph.redraw_edges()
            self.column_characterization(starting_index, search_type=12)
            self.column_characterization(starting_index, search_type=4)
            self.column_characterization(starting_index, search_type=3)
            self.column_characterization(starting_index, search_type=5)
            self.column_characterization(starting_index, search_type=6)
            self.summarize_stats()
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_advanced(self.graph, i)
            for i in range(0, self.num_columns):
                if self.graph.vertices[i].is_popular and not self.graph.vertices[i].set_by_user and not self.graph.vertices[i].is_edge_column:
                    legacy_items.connection_shift_on_level(self.graph, i)
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_advanced(self.graph, i)
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_advanced(self.graph, i, experimental=True)
            self.graph.redraw_edges()
            self.column_characterization(starting_index, search_type=12)
            self.column_characterization(starting_index, search_type=4)
            self.column_characterization(starting_index, search_type=3)
            self.column_characterization(starting_index, search_type=5)
            self.column_characterization(starting_index, search_type=6)
            self.graph.redraw_edges()
            self.summarize_stats()
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_advanced(self.graph, i)
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_advanced(self.graph, i)
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_advanced(self.graph, i, experimental=True)
            self.graph.redraw_edges()
            self.summarize_stats()
            logger.info('Legacy weak untangling complete!')

        elif search_type == 9:

            logger.info('Starting strong untangling...')
            logger.info('Could not start strong untangling because it is not implemented yet!')

        elif search_type == 10:

            logger.info('Starting experimental weak untangling...')

            static = False
            total_changes = 0
            total_counter = 0

            while not static:

                for type_num in range(1, 6):

                    cont = True
                    counter = 0
                    while cont:
                        self.graph.redraw_edges()
                        chi_before = self.graph.chi
                        self.column_characterization(starting_index, search_type=14)
                        logger.info('Looking for type {}:'.format(type_num))
                        logger.info('Chi: {}'.format(chi_before))
                        if type_num == 1:
                            num_types, changes = weak_untangling.process_type_1(self.graph)
                        elif type_num == 2:
                            num_types, changes = weak_untangling.process_type_2(self.graph)
                        elif type_num == 3:
                            num_types, changes = weak_untangling.process_type_3(self.graph)
                        elif type_num == 4:
                            num_types, changes = weak_untangling.process_type_4(self.graph)
                        elif type_num == 5:
                            num_types, changes = weak_untangling.process_type_5(self.graph)
                        else:
                            changes = 0
                            num_types = 0
                        total_changes += changes
                        self.graph.redraw_edges()
                        chi_after = self.graph.chi
                        logger.info('Found {} type {}\'s, made {} changes'.format(num_types, type_num, changes))
                        logger.info('Chi: {}'.format(chi_before))

                        if chi_after <= chi_before:
                            logger.info('repeating...')
                            counter += 1
                        else:
                            cont = False

                        if changes == 0:
                            cont = False
                            logger.info('No changes made, continuing...')

                        if counter > 4:
                            cont = False
                            logger.info('Emergency abort!')

                total_counter += 1

                if total_changes == 0:
                    static = True

                if total_counter > 3:
                    static = True

            self.graph.redraw_edges()
            logger.info('Weak untangling complete')

        elif search_type == 11:

            logger.info('Starting experimental strong untangling...')

            static = False
            total_changes = 0
            total_counter = 0

            while not static:

                for type_num in range(1, 2):

                    cont = True
                    counter = 0
                    while cont:
                        self.graph.redraw_edges()
                        chi_before = self.graph.chi
                        logger.info('Looking for type {}:'.format(type_num))
                        logger.info('Chi: {}'.format(chi_before))
                        if type_num == 1:
                            changes = 0
                        else:
                            changes = 0
                        total_changes += changes
                        self.graph.redraw_edges()
                        chi_after = self.graph.chi
                        logger.info('Made {} changes'.format(type_num, changes))
                        logger.info('Chi: {}'.format(chi_before))

                        if chi_after <= chi_before and counter > 0:
                            logger.info('repeating...')
                            counter += 1
                        else:
                            cont = False

                        if changes == 0:
                            cont = False
                            logger.info('No changes made, continuing...')

                        if counter > 4:
                            cont = False
                            logger.info('Emergency abort!')

                total_counter += 1

                if total_changes == 0:
                    static = True

                if total_counter > 3:
                    static = True

            self.graph.redraw_edges()
            logger.info('Strong untangling complete')

        elif search_type == 12:

            logger.info('Resetting probability vectors with zero bias...')
            for i in range(0, self.num_columns):
                if not self.graph.vertices[i].set_by_user and not self.graph.vertices[i].is_edge_column:
                    self.graph.vertices[i].reset_prob_vector()
            logger.info('Probability vectors reset.')

        elif search_type == 13:

            logger.info('Resetting user-set columns...')
            for i in range(0, self.num_columns):
                if self.graph.vertices[i].set_by_user:
                    self.graph.vertices[i].reset_prob_vector()
            logger.info('User-set columns was re-set.')

        elif search_type == 14:

            logger.info('Looking for intersections')
            intersections = self.graph.find_intersects()
            num_intersections = len(intersections)
            # not_removed, strong_intersections, ww, ss = graph_op.remove_intersections(self.graph)
            graph_op.experimental_remove_intersections(self.graph)
            intersections = self.graph.find_intersects()
            logger.info('Found {} intersections'.format(num_intersections))
            # self.report('        Found {} strong intersections'.format(ss), force=True)
            # self.report('        Found {} weak-weak intersections'.format(ww), force=True)
            # self.report('        {} weak intersections were not removed'.format(not_removed), force=True)
            logger.info('{} literal intersections still remain'.format(len(intersections)))

        elif search_type == 15:

            logger.info('Starting column characterization from vertex {}...'.format(starting_index))
            logger.info('Setting alloy...')
            self.set_alloy_mat()
            logger.info('Alloy set.')
            logger.info('Finding edge columns....')
            self.find_edge_columns()
            for vertex in self.graph.vertices:
                if vertex.is_edge_column and not vertex.set_by_user:
                    vertex.reset_prob_vector(bias=3)
                    vertex.reset_symmetry_vector(bias=-1)
            logger.info('Found edge columns.')
            # Reset prob vectors:
            self.column_characterization(starting_index, search_type=12)
            # Spatial mapping:
            self.column_characterization(starting_index, search_type=2)
            # Angle analysis:
            self.column_characterization(starting_index, search_type=3)
            # Intensity analysis
            self.column_characterization(starting_index, search_type=4)
            # Find particle:
            self.column_characterization(starting_index, search_type=5)
            # Set levels:
            self.column_characterization(starting_index, search_type=6)
            # Add edges:
            self.column_characterization(starting_index, search_type=7)
            # Remove crossing edges
            self.column_characterization(starting_index, search_type=14)
            # Weak
            logger.info('Starting experimental weak untangling...')
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_simple(self.graph, i, sub=True)
            self.graph.redraw_edges()
            self.column_characterization(starting_index, search_type=12)
            self.column_characterization(starting_index, search_type=4)
            self.column_characterization(starting_index, search_type=16)
            self.column_characterization(starting_index, search_type=5)
            self.column_characterization(starting_index, search_type=6)
            self.summarize_stats()
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_simple(self.graph, i)
            self.graph.redraw_edges()
            self.column_characterization(starting_index, search_type=12)
            self.column_characterization(starting_index, search_type=4)
            self.column_characterization(starting_index, search_type=16)
            self.column_characterization(starting_index, search_type=5)
            self.column_characterization(starting_index, search_type=6)
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_advanced(self.graph, i)
            for i in range(0, self.num_columns):
                legacy_items.find_consistent_perturbations_advanced(self.graph, i)
            self.graph.redraw_edges()
            self.column_characterization(starting_index, search_type=12)
            self.column_characterization(starting_index, search_type=4)
            self.column_characterization(starting_index, search_type=16)
            self.column_characterization(starting_index, search_type=5)
            self.column_characterization(starting_index, search_type=6)
            # Summarize:
            logger.info('Summarizing stats.')
            self.summarize_stats()
            # Summarize:
            logger.info('Summarizing stats.')
            self.summarize_stats()
            # Complete:
            logger.info('Column characterization complete.')

        elif search_type == 16:

            logger.info('Running experimental angle analysis')

            for i in range(0, self.num_columns):
                if not self.graph.vertices[i].set_by_user and not self.graph.vertices[i].is_edge_column:
                    self.graph.vertices[i].reset_symmetry_vector()
                    self.graph.vertices[i].reset_prob_vector()
                    print('vertex {}: --------------------'.format(i))
                    print('    Reset to: {}'.format(self.graph.vertices[i].prob_vector))
                    self.graph.vertices[i].prob_vector =\
                        graph_op.base_angle_score(self.graph, i)
                    self.graph.vertices[i].prob_vector = np.array(self.graph.vertices[i].prob_vector)
                    self.graph.vertices[i].define_species()

            logger.info('Angle analysis complete!')

        elif search_type == 17:

            for vertex in self.graph.vertices:
                vertex.reset_level_vector()
            runs, measures = graph_op.statistical_level_bleed(self.graph, starting_index, self.graph.vertices[starting_index].level)
            plt.plot(runs, measures)
            plt.show()
            graph_op.sort_neighbourhood(self.graph)

        elif search_type == 18:

            logger.info('Finding edge columns....')
            self.find_edge_columns()
            for vertex in self.graph.vertices:
                if vertex.is_edge_column and not vertex.set_by_user:
                    vertex.reset_prob_vector(bias=3)
                    vertex.reset_symmetry_vector(bias=-1)
            logger.info('Found edge columns.')

        else:

            logger.error('No such search type!')

    def calc_avg_gamma(self):
        if self.num_columns > 0:
            temp_mat = mat_op.gen_framed_mat(self.im_mat, self.r)
            for vertex in self.graph.vertices:
                vertex.avg_gamma, vertex.peak_gamma = mat_op.average(temp_mat, vertex.im_coor_x + self.r,
                                                                     vertex.im_coor_y + self.r, self.r)

    def summarize_stats(self):

        self.graph.summarize_stats()

        self.avg_peak_gamma = 0
        self.avg_avg_gamma = 0

        self.num_inconsistencies = 0
        self.num_popular = 0
        self.num_unpopular = 0

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

                self.avg_peak_gamma += self.graph.vertices[x].peak_gamma
                self.avg_avg_gamma += self.graph.vertices[x].avg_gamma

                if self.graph.vertices[x].is_unpopular and not self.graph.vertices[x].is_edge_column:
                    self.num_unpopular += 1
                    self.num_inconsistencies += 1

                if self.graph.vertices[x].is_popular and not self.graph.vertices[x].is_edge_column:
                    self.num_popular += 1
                    self.num_inconsistencies += 1

                if self.graph.vertices[x].h_index == 0:
                    self.num_si += 1
                    self.number_percentage_si += 1
                    self.avg_si_peak_gamma += self.graph.vertices[x].peak_gamma
                    self.avg_si_avg_gamma += self.graph.vertices[x].avg_gamma
                    if self.graph.vertices[x].is_in_precipitate:
                        self.num_precipitate_columns += 1
                        self.num_precipitate_si += 1
                        self.precipitate_number_percentage_si += 1
                elif self.graph.vertices[x].h_index == 1:
                    self.num_cu += 1
                    self.number_percentage_cu += 1
                    self.avg_cu_peak_gamma += self.graph.vertices[x].peak_gamma
                    self.avg_cu_avg_gamma += self.graph.vertices[x].avg_gamma
                    if self.graph.vertices[x].is_in_precipitate:
                        self.num_precipitate_columns += 1
                        self.num_precipitate_cu += 1
                        self.precipitate_number_percentage_cu += 1
                elif self.graph.vertices[x].h_index == 2:
                    self.num_zn += 1
                    self.number_percentage_zn += 1
                    self.avg_zn_peak_gamma += self.graph.vertices[x].peak_gamma
                    self.avg_zn_avg_gamma += self.graph.vertices[x].avg_gamma
                    if self.graph.vertices[x].is_in_precipitate:
                        self.num_precipitate_columns += 1
                        self.num_precipitate_zn +=1
                        self.precipitate_number_percentage_zn += 1
                elif self.graph.vertices[x].h_index == 3:
                    self.num_al += 1
                    self.number_percentage_al += 1
                    self.avg_al_peak_gamma += self.graph.vertices[x].peak_gamma
                    self.avg_al_avg_gamma += self.graph.vertices[x].avg_gamma
                    if self.graph.vertices[x].is_in_precipitate:
                        self.num_precipitate_columns += 1
                        self.num_precipitate_al += 1
                        self.precipitate_number_percentage_al += 1
                elif self.graph.vertices[x].h_index == 4:
                    self.num_ag += 1
                    self.number_percentage_ag += 1
                    self.avg_ag_peak_gamma += self.graph.vertices[x].peak_gamma
                    self.avg_ag_avg_gamma += self.graph.vertices[x].avg_gamma
                    if self.graph.vertices[x].is_in_precipitate:
                        self.num_precipitate_columns += 1
                        self.num_precipitate_ag += 1
                        self.precipitate_number_percentage_ag += 1
                elif self.graph.vertices[x].h_index == 5:
                    self.num_mg += 1
                    self.number_percentage_mg += 1
                    self.avg_mg_peak_gamma += self.graph.vertices[x].peak_gamma
                    self.avg_mg_avg_gamma += self.graph.vertices[x].avg_gamma
                    if self.graph.vertices[x].is_in_precipitate:
                        self.num_precipitate_columns += 1
                        self.num_precipitate_mg += 1
                        self.precipitate_number_percentage_mg += 1
                elif self.graph.vertices[x].h_index == 6:
                    self.num_un += 1
                    self.number_percentage_un += 1
                    self.avg_un_peak_gamma += self.graph.vertices[x].peak_gamma
                    self.avg_un_avg_gamma += self.graph.vertices[x].avg_gamma
                    if self.graph.vertices[x].is_in_precipitate:
                        self.num_precipitate_columns += 1
                        self.num_precipitate_un += 1
                        self.precipitate_number_percentage_un += 1
                else:
                    logger.warning('Unexpected behaviour in core.SuchSoftware.summarize_stats()')

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
                self.precipitate_number_percentage_si =\
                    self.precipitate_number_percentage_si / self.num_precipitate_columns
                self.precipitate_number_percentage_cu =\
                    self.precipitate_number_percentage_cu / self.num_precipitate_columns
                self.precipitate_number_percentage_zn =\
                    self.precipitate_number_percentage_zn / self.num_precipitate_columns
                self.precipitate_number_percentage_al =\
                    self.precipitate_number_percentage_al / self.num_precipitate_columns
                self.precipitate_number_percentage_ag =\
                    self.precipitate_number_percentage_ag / self.num_precipitate_columns
                self.precipitate_number_percentage_mg =\
                    self.precipitate_number_percentage_mg / self.num_precipitate_columns
                self.precipitate_number_percentage_un =\
                    self.precipitate_number_percentage_un / self.num_precipitate_columns

        self.build_stat_string()

    def build_stat_string(self):

        self.stats_string = ('Number of detected columns: ' + str(self.num_columns) + '\n'
            'Number of detected precipitate columns: ' + str(self.num_precipitate_columns) + '\n\n'
            'Number of inconsistencies: ' + str(self.num_inconsistencies) + '\n'
            'Number of popular: ' + str(self.num_popular) + '\n'
            'Number of unpopular: ' + str(self.num_unpopular) + '\n'
            'Chi: ' + str(self.graph.chi) + '\n\n'
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
            'Number procentage of precipitate Un: ' + str(self.precipitate_number_percentage_un))

    def redraw_search_mat(self):

        self.search_mat = self.im_mat
        if self.num_columns > 0:
            self.search_mat = mat_op.gen_framed_mat(self.search_mat, self.r + self.overhead)
            for i in range(0, self.num_columns):
                self.search_mat = mat_op.delete_pixels(self.search_mat,
                                                       self.graph.vertices[i].im_coor_x + self.r + self.overhead,
                                                       self.graph.vertices[i].im_coor_y + self.r + self.overhead,
                                                       self.r + self.overhead)
            self.search_mat = mat_op.gen_de_framed_mat(self.search_mat, self.r + self.overhead)

    def redraw_circumference_mat(self):

        self.column_circumference_mat = np.zeros((self.im_height, self.im_width), dtype=type(self.im_mat))
        if self.num_columns > 0:
            self.column_circumference_mat = mat_op.gen_framed_mat(self.column_circumference_mat, self.r + self.overhead)
            for x in range(0, self.num_columns):
                self.column_circumference_mat = mat_op.draw_circle(self.column_circumference_mat,
                                                                   self.graph.vertices[x].im_coor_x + self.r +
                                                                   self.overhead,
                                                                   self.graph.vertices[x].im_coor_y + self.r +
                                                                   self.overhead, self.r)
            self.column_circumference_mat = mat_op.gen_de_framed_mat(self.column_circumference_mat, self.r +
                                                                     self.overhead)

    def redraw_centre_mat(self):

        self.column_centre_mat = np.zeros((self.im_height, self.im_width, 2), dtype=type(self.im_mat))
        if self.num_columns > 0:
            for x in range(0, self.num_columns):
                self.column_centre_mat[self.graph.vertices[x].im_coor_y, self.graph.vertices[x].im_coor_x, 0] = 1
                self.column_centre_mat[self.graph.vertices[x].im_coor_y, self.graph.vertices[x].im_coor_x, 1] = x

    def reset_graph(self):
        self.graph = graph.AtomicGraph()
        self.num_columns = 0
        self.redraw_centre_mat()
        self.redraw_circumference_mat()
        self.redraw_search_mat()
        self.summarize_stats()

    def reset_vertex_properties(self):
        self.graph.reset_vertex_properties()
        self.summarize_stats()

