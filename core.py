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
import weak_untangling
import strong_untangling


class SuchSoftware:

    # Version
    version = [0, 0, 3]

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

        self.display_stats_string = 'Empty'
        self.export_data_string = ' '

        # These are hyper-parameters of the algorithms. See the documentation.
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

        # Initialize an empty graph
        self.graph = graph.AtomicGraph(map_size=self.map_size)

    def report(self, string, force=False, update=False):
        if self.debug_mode or force:
            if self.debug_obj is not None:
                if not string == '':
                    self.debug_obj('core: ' + string, update)
                else:
                    self.debug_obj(string, update)
            else:
                print(string)

    def run_test(self):

        deviations = []
        averages = []
        number_of_vertices = []
        filenames = []
        number_of_files = 0

        with open('Saves/validation_set/filenames.txt', mode='r') as f:
            for line in f:
                line = line.replace('\n', '')
                filename, control = line.split(',')
                filenames.append(filename)
                filename = 'Saves/validation_set/' + filename
                control = 'Saves/validation_set/' + control
                deviation, avg = SuchSoftware.run_individual_test(filename, control)
                deviations.append(deviation)
                averages.append(avg)
                number_of_vertices.append(int(deviation / avg))
                number_of_files += 1

        self.report('Algorithm test:-----', force=True)
        self.report('Number of files tested: {}'.format(number_of_files), force=True)
        self.report('Total deviations found: {}'.format(sum(deviations)), force=True)
        self.report('Total vertices checked: {}'.format(sum(number_of_vertices)), force=True)
        self.report('Average deviation: {}'.format(sum(deviations) / sum(number_of_vertices)), force=True)
        self.report('Individual results: (filename, deviations, number of vertices, average)', force=True)
        for i, item in enumerate(deviations):
            self.report('    {}, {}, {}, {}'.format(filenames[i], deviations[i], number_of_vertices[i], averages[i]), force=True)

    @staticmethod
    def run_individual_test(filename, control_file, mode=0):

        file = SuchSoftware.load(filename)
        control = SuchSoftware.load(control_file)
        i = file.starting_index

        file.column_characterization(i, search_type=mode)

        deviations, avg = SuchSoftware.measure_deviance(file, control)

        file.save(filename + '_test_result')

        return deviations, avg

    @staticmethod
    def measure_deviance(obj, control):
        if obj.graph.num_vertices == control.graph.num_vertices:
            deviations = 0
            for vertex, control_vertex in zip(obj.graph.vertices, control.graph.vertices):
                if not vertex.is_edge_column:
                    if not vertex.h_index == control_vertex.h_index:
                        deviations = deviations + 1
            return deviations, deviations / obj.graph.num_vertices
        else:
            return None

    def vertex_report(self, i):
        self.report(' ', force=True)
        self.report('Vertex properties: ------------', force=True)
        self.report('    Index: {}'.format(self.graph.vertices[i].i), force=True)
        self.report('    Image pos: ({}, {})'.format(self.graph.vertices[i].im_coor_x,
                                                     self.graph.vertices[i].im_coor_y), force=True)
        self.report('    Real pos: ({}, {})'.format(self.graph.vertices[i].real_coor_x,
                                                    self.graph.vertices[i].real_coor_y), force=True)
        self.report('    Atomic Species: {}'.format(self.graph.vertices[i].atomic_species), force=True)
        self.report(('    Probability vector: {}'.format(self.graph.vertices[i].prob_vector).replace('\n', '')), force=True)
        self.report('    Partner vector: {}'.format(self.graph.vertices[i].partners()), force=True)
        self.report(' ', force=True)

    def image_report(self):
        self.summarize_stats()
        self.report(' ', force=True)
        self.report('Project properties: ---------', force=True)
        for line in iter(self.display_stats_string.splitlines()):
            self.report('    {}'.format(line), force=True)
        self.report(' ', force=True)

    def set_alloy_mat(self):

        if self.alloy == 0:

            for x in range(0, SuchSoftware.num_selections):
                if x == 2 or x == 4:
                    self.alloy_mat[x] = 0
                else:
                    self.alloy_mat[x] = 1

            string = 'Alloy set to \'Al-Mg-Si-(Cu)\'.'

        elif self.alloy == 1:

            for x in range(0, SuchSoftware.num_selections):
                if x == 2 or x == 4 or x == 1:
                    self.alloy_mat[x] = 0
                else:
                    self.alloy_mat[x] = 1

            string = 'Alloy set to \'Al-Mg-Si\'.'

        else:

            string = 'Failed to set alloy. Unknown alloy number'

        if not (self.filename_full == 'empty' or self.filename_full == 'Empty'):
            self.report(string, force=True)

    def save(self, filename_full):
        with open(filename_full, 'wb') as f:
            _ = self.debug_obj
            self.debug_obj = None
            self.version_saved = self.version
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            self.debug_obj = _

    @staticmethod
    def load(filename_full, report=None):
        with open(filename_full, 'rb') as f:
            try:
                obj = pickle.load(f)
            except:
                obj = None
                if report is not None:
                    report('core: Failed to load save-file!', update=True)
                else:
                    print('core: Failed to load save-file!')
            else:
                if not obj.version_saved == SuchSoftware.version:
                    if report is not None:
                        report('core: Attempted to load uncompatible save-file. Running conversion script...', update=False)
                    else:
                        print('core: Attempted to load uncompatible save-file. Running conversion script...')
                    obj = compatibility.convert(obj, obj.version_saved, SuchSoftware.version)
            return obj

    def column_detection(self, search_type='s'):
        self.report(' ', force=True)
        if self.num_columns == 0:
            self.report('Starting column detection. Search mode is \'{}\''.format(search_type), force=True)
        else:
            self.report('Continuing column detection. Search mode is \'{}\''.format(search_type), force=True)
        cont = True
        counter = self.num_columns
        self.column_circumference_mat = mat_op.gen_framed_mat(self.column_circumference_mat, self.r + self.overhead)
        self.search_mat = mat_op.gen_framed_mat(self.search_mat, self.r + self.overhead)
        self.im_mat = mat_op.gen_framed_mat(self.im_mat, self.r + self.overhead)

        while cont:

            pos = np.unravel_index(self.search_mat.argmax(),
                                   (self.im_height + 2 * (self.r + self.overhead),
                                    self.im_width + 2 * (self.r + self.overhead)))
            max_val = self.search_mat.max()
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
                                  certainty_threshold=self.certainty_threshold)
            vertex.reset_prob_vector(bias=6)
            self.graph.add_vertex(vertex)

            self.column_centre_mat[y_fit_real_coor_pix, x_fit_real_coor_pix, 0] = 1
            self.column_centre_mat[y_fit_real_coor_pix, x_fit_real_coor_pix, 1] = counter
            self.column_circumference_mat = mat_op.draw_circle(self.column_circumference_mat, x_fit_pix, y_fit_pix,
                                                               self.r)

            self.report(str(counter) + ': (' + str(x_fit_real_coor) + ', ' + str(y_fit_real_coor) + ') | (' + str(
                pos[1]) + ', ' + str(pos[0]) + ')', force=False)

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
                self.report('Invalid search type sent to SuchSoftware.column_detection', force=True)

        self.column_circumference_mat = mat_op.gen_de_framed_mat(self.column_circumference_mat, self.r + self.overhead)
        self.search_mat = mat_op.gen_de_framed_mat(self.search_mat, self.r + self.overhead)
        self.im_mat = mat_op.gen_de_framed_mat(self.im_mat, self.r + self.overhead)
        self.calc_avg_gamma()
        self.summarize_stats()
        self.report('Column detection complete! Found {} columns'.format(self.num_columns), force=True)
        self.report(' ', force=True)

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
            self.report('        Did not find enough neighbours for vertex {}. increasing search area.'.format(i), force=False)

        else:

            self.report('        Found {} total neighbours for vertex {}'.format(total_num, i), force=False)

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

            self.report('Starting column characterization from vertex {}...'.format(starting_index), force=True)
            self.report('    Setting alloy...', force=True)
            self.set_alloy_mat()
            self.report('    Alloy set.', force=True)
            self.report('    Finding edge columns....', force=True)
            self.find_edge_columns()
            for i in range(0, self.num_columns):
                if self.graph.vertices[i].is_edge_column and not self.graph.vertices[i].set_by_user:
                    self.graph.vertices[i].reset_prob_vector(bias=3)
            self.report('    Found edge columns.', force=True)
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
            self.report('    Summarizing stats.', force=True)
            self.summarize_stats()
            # Legacy weak untanglng
            self.column_characterization(starting_index, search_type=8)
            # Legacy strong untangling
            self.column_characterization(starting_index, search_type=9)
            # Summarize:
            self.report('    Summarizing stats.', force=True)
            self.summarize_stats()
            # Complete:
            self.report('Column characterization complete.', force=True)
            self.report(' ', force=True)

        elif search_type == 1:

            self.report('Starting column characterization from vertex {}...'.format(starting_index), force=True)
            self.report('    Setting alloy', force=True)
            self.set_alloy_mat()
            self.report('    Alloy set.', force=True)

            self.column_characterization(starting_index, search_type=2)

            self.column_characterization(starting_index, search_type=3)

            self.column_characterization(starting_index, search_type=4)

            self.column_characterization(starting_index, search_type=5)

            self.column_characterization(starting_index, search_type=6)

            self.column_characterization(starting_index, search_type=7)

            self.report('    Summarizing stats.', force=True)
            self.summarize_stats()

            self.column_characterization(starting_index, search_type=10)

            self.column_characterization(starting_index, search_type=11)

            self.report('    Summarizing stats.', force=True)
            self.summarize_stats()

            self.report('Column characterization complete.', force=True)
            self.report(' ', force=True)

        elif search_type == 2:

            self.report('    Mapping spatial locality...', force=True)
            self.redraw_centre_mat()
            self.redraw_circumference_mat()
            for i in range(0, self.num_columns):
                self.graph.vertices[i].neighbour_indices, _ = self.find_nearest(i, self.map_size)
            self.report('    Spatial mapping complete.', force=True)

        elif search_type == 3:

            self.report('    Analysing angles...', force=True)
            for i in range(0, self.num_columns):
                if not self.graph.vertices[i].set_by_user and not self.graph.vertices[i].is_edge_column:
                    graph_op.apply_angle_score(self.graph, i, self.dist_3_std, self.dist_4_std, self.dist_5_std,
                                               self.num_selections)
            self.report('    Angle analysis complete.', force=True, update=True)

        elif search_type == 4:

            self.report('    Analyzing intensities...', force=True)
            for i in range(0, self.num_columns):
                if not self.graph.vertices[i].set_by_user and not self.graph.vertices[i].is_edge_column:
                    graph_op.apply_intensity_score(self.graph, i, self.num_selections, self.intensities[self.alloy],
                                                   self.dist_8_std)
            self.report('    Intensity analysis complete.', force=True, update=True)

        elif search_type == 5:

            self.report('    Finding particle with legacy method....', force=True)
            legacy_items.precipitate_controller(self.graph, starting_index)
            # graph_op.precipitate_controller(self.graph, starting_index)
            self.report('    Found particle.', force=True)

        elif search_type == 6:

            self.report('    Running legacy level definition algorithm....', force=True)
            legacy_items.define_levels(self.graph, starting_index, self.graph.vertices[starting_index].level)
            # graph_op.set_levels(self.graph, starting_index, self.report, self.graph.vertices[starting_index].level,
                               # self.num_selections)
            self.report('    Levels set.', force=True, update=True)

        elif search_type == 7:

            self.report('    adding edges to graph...', force=True)
            self.graph.redraw_edges()
            self.report('    Edges added.', force=True, update=True)

        elif search_type == 8:

            self.report('    Starting legacy weak untangling...', force=True)
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
            self.report('    Legacy weak untangling complete!', force=True)

        elif search_type == 9:

            self.report('    Starting strong untangling...', force=True)
            self.report('        Could not start strong untangling because it is not implemented yet!', force=True)

        elif search_type == 10:

            self.report('    Starting experimental weak untangling...', force=True)
            self.report('        Could not start weak untangling because it is not implemented yet!', force=True)

        elif search_type == 11:

            self.report('    Starting strong untangling...', force=True)
            self.report('        Could not start strong untangling because it is not implemented yet!', force=True)

        elif search_type == 12:

            self.report('    Resetting probability vectors with zero bias...', force=True)
            for i in range(0, self.num_columns):
                if not self.graph.vertices[i].set_by_user and not self.graph.vertices[i].is_edge_column:
                    self.graph.vertices[i].reset_prob_vector()
            self.report('    Probability vectors reset.', force=True)

        elif search_type == 13:

            self.report('    Resetting user-set columns...', force=True)
            for i in range(0, self.num_columns):
                if self.graph.vertices[i].set_by_user:
                    self.graph.vertices[i].reset_prob_vector()
            self.report('    User-set columns was re-set.', force=True)

        elif search_type == 14:
            self.report('    Looking for intersections', force=True)
            intersections = self.graph.find_intersects()
            num_intersections = len(intersections)
            self.report('        Found:', force=True)
            for intersection in intersections:
                self.report('            {}'.format(str(intersection)), force=True)
            not_removed, strong_intersections, ww, ss = graph_op.remove_intersections(self.graph)
            intersections = self.graph.find_intersects()
            self.report('        Found {} intersections'.format(num_intersections), force=True)
            self.report('        Found {} strong intersections'.format(ss), force=True)
            self.report('        Found {} weak-weak intersections'.format(ww), force=True)
            self.report('        {} weak intersections were not removed'.format(not_removed), force=True)
            self.report('        {} literal intersections still remain'.format(len(intersections)), force=True)
            self.report('        Intersections:', force=True)
            for intersection in intersections:
                self.report('            {}'.format(str(intersection)), force=True)
            self.report('        Strong intersections:', force=True)
            for strong_intersection in strong_intersections:
                self.report('            {}'.format(str(strong_intersection)), force=True)

        if search_type == 15:

            self.report('Starting column characterization from vertex {}...'.format(starting_index), force=True)
            self.report('    Setting alloy...', force=True)
            self.set_alloy_mat()
            self.report('    Alloy set.', force=True)
            self.report('    Finding edge columns....', force=True)
            self.find_edge_columns()
            for vertex in self.graph.vertices:
                if vertex.is_edge_column and not vertex.set_by_user:
                    vertex.reset_prob_vector(bias=3)
                    vertex.reset_symmetry_vector(bias=-1)
            self.report('    Found edge columns.', force=True)
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
            self.report('    Starting experimental weak untangling...', force=True)
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
            self.report('    Summarizing stats.', force=True)
            self.summarize_stats()
            # Summarize:
            self.report('    Summarizing stats.', force=True)
            self.summarize_stats()
            # Complete:
            self.report('Column characterization complete.', force=True)
            self.report(' ', force=True)

        elif search_type == 16:

            for vertex in self.graph.vertices:
                if not vertex.set_by_user:
                    graph_op.base_angle_score(self.graph, vertex.i, self.dist_3_std, self.dist_4_std, self.dist_5_std)
                    graph_op.mesh_angle_score(self.graph, vertex.i, self.dist_3_std, self.dist_4_std, self.dist_5_std)
                    vertex.reset_prob_vector()
                    vertex.multiply_symmetry()

        else:

            self.report('Error: No such search type!', force=True)

    def calc_avg_gamma(self):
        if self.num_columns > 0:

            self.im_mat = mat_op.gen_framed_mat(self.im_mat, self.r)

            for x in range(0, self.num_columns):

                self.graph.vertices[x].avg_gamma, self.graph.vertices[x].peak_gamma =\
                    mat_op.average(self.im_mat, self.graph.vertices[x].im_coor_x + self.r,
                                   self.graph.vertices[x].im_coor_y + self.r, self.r)

            self.im_mat = mat_op.gen_de_framed_mat(self.im_mat, self.r)

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

        self.display_stats_string = ('Number of detected columns: ' + str(self.num_columns) + '\n'
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
        self.graph = graph.AtomicGraph
        self.num_columns = 0
        self.redraw_centre_mat()
        self.redraw_circumference_mat()
        self.redraw_search_mat()
        self.summarize_stats()

    def reset_vertex_properties(self):
        self.graph.reset_vertex_properties()
        self.summarize_stats()

