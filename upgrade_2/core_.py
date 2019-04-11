# This file contains the SuchSoftware class that is the algorithms.
import numpy as np
import dm3_lib as dm3
import mat_op
import graph
import utils
import graph_op
import sys
import pickle


class SuchSoftware:

    # Version
    version = [0, 0, 0]

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

    # Indexable species strings
    species_strings = ['Si', 'Cu', 'Zn', 'Al', 'Ag', 'Mg', 'Un']

    # Constructor
    def __init__(self, filename_full, debug_obj=None):

        self.filename_full = filename_full
        self.im_mat = None
        self.scale = 1
        self.im_height = 0
        self.im_width = 0

        if not (filename_full == 'Empty' or filename_full == 'empty'):
            dm3f = dm3.DM3(self.filename_full)
            self.im_mat = dm3f.imagedata
            (self.scale, junk) = dm3f.pxsize
            self.scale = 1000 * self.scale
            self.im_mat = mat_op.normalize_static(self.im_mat)
            (self.im_height, self.im_width) = self.im_mat.shape
            self.fft_im_mat = mat_op.gen_fft(self.im_mat)

        # For communicating with the interface, if any:
        self.debug_obj = debug_obj
        self.debug_mode = False

        # Data matrices: These hold much of the information gathered by the different algorithms
        self.search_mat = self.im_mat
        self.column_centre_mat = np.zeros((self.im_height, self.im_width, 2), dtype=type(self.im_mat))
        self.column_circumference_mat = np.zeros((self.im_height, self.im_width), dtype=type(self.im_mat))

        # Alloy info: This vector is used to multiply away elements in the AtomicColumn.prob_vector that are not in
        # the alloy being studied. Currently supported alloys are:
        # self.alloy = alloy
        # 0 = Al-Si-Mg-Cu
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

        # Initialize an empty graph
        self.graph = graph.AtomicGraph()

    def report(self, string, force=False, update=False):
        if self.debug_mode or force:
            if self.debug_obj is not None:
                if not string == '':
                    self.debug_obj('core: ' + string, update)
                else:
                    self.debug_obj(string, update)
            else:
                print(string)

    def vertex_report(self, i):
        self.report('Vertex properties: ------------', force=True)
        self.report('    Index: {}'.format(self.graph.vertices[i].i), force=True)
        self.report('    Image pos: ({}, {})'.format(self.graph.vertices[i].im_coor_x,
                                                     self.graph.vertices[i].im_coor_y), force=True)
        self.report('    Real pos: ({}, {})'.format(self.graph.vertices[i].real_coor_x,
                                                    self.graph.vertices[i].real_coor_y), force=True)
        self.report('    Atomic Species: {}'.format(self.graph.vertices[i].atomic_species), force=True)
        self.report(('    Probability vector: {}'.format(self.graph.vertices[i].prob_vector).replace('\n', '')), force=True)

    def image_report(self):
        self.summarize_stats()
        self.report('Project properties: ---------', force=True)
        for line in iter(self.display_stats_string.splitlines()):
            self.report('    {}'.format(line), force=True)

    def set_alloy_mat(self, alloy=0):

        self.alloy = alloy

        if alloy == 0:

            for x in range(0, SuchSoftware.num_selections):
                if x == 2 or x == 4:
                    self.alloy_mat[x] = 0
                else:
                    self.alloy_mat[x] = 1

        elif alloy == 1:

            for x in range(0, SuchSoftware.num_selections):
                if x == 2 or x == 4 or x == 1:
                    self.alloy_mat[x] = 0
                else:
                    self.alloy_mat[x] = 1

        else:

            print('Not a supported alloy number!')

    def save(self, filename_full):
        with open(filename_full, 'wb') as f:
            _ = self.debug_obj
            self.debug_obj = None
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            self.debug_obj = _

    @staticmethod
    def load(filename_full):
        with open(filename_full, 'rb') as f:
            obj = pickle.load(f)
            return obj

    def column_detection(self, search_type='s'):
        if self.num_columns == 0:
            self.report('Starting column detection. Search mode is \'{}\''.format(search_type), force=True)
        else:
            self.report('Continuing column detection. Search mode is \'{}\''.format(search_type), force=True)
        cont = True
        counter = self.num_columns
        self.set_alloy_mat(self.alloy)
        self.column_circumference_mat = mat_op.gen_framed_mat(self.column_circumference_mat, self.r + self.overhead)
        self.search_mat = mat_op.gen_framed_mat(self.search_mat, self.r + self.overhead)
        self.im_mat = mat_op.gen_framed_mat(self.im_mat, self.r + self.overhead)

        while cont:

            pos = np.unravel_index(self.search_mat.argmax(),
                                   (self.im_height + 2 * (self.r + self.overhead),
                                    self.im_width + 2 * (self.r + self.overhead)))
            column_portrait, x_fit, y_fit = utils.cm_fit(self.im_mat, pos[1], pos[0], self.r)

            x_fit_real_coor = x_fit - self.r - self.overhead
            y_fit_real_coor = y_fit - self.r - self.overhead
            x_fit_real_coor_pix = int(np.floor(x_fit_real_coor))
            y_fit_real_coor_pix = int(np.floor(y_fit_real_coor))
            x_fit_pix = int(np.floor(x_fit))
            y_fit_pix = int(np.floor(y_fit))

            self.search_mat = mat_op.delete_pixels(self.search_mat, x_fit_pix, y_fit_pix, self.r + self.overhead)

            vertex = graph.Vertex(counter, x_fit_real_coor, y_fit_real_coor, self.r, np.max(column_portrait), 0,
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

    def column_characterization(self, starting_index, search_type=0):

        if search_type == 0:
            self.report('Starting column characterization from vertex {}...'.format(starting_index), force=True)
            self.report('    Mapping spatial locality...', force=True)
            self.graph.map_spatial_neighbours()
            self.report('    Spatial mapping complete.', force=True)
            self.report('    Analysing angles...', force=True)
            for i in range(0, self.num_columns):
                if not self.graph.vertices[i].set_by_user:
                    self.graph.vertices[i].reset_prob_vector()
                    graph_op.apply_angle_score(self.graph, i, self.dist_3_std, self.dist_4_std, self.dist_5_std,
                                               self.num_selections)
            self.report('    Angle analysis complete.', force=True, update=True)
            self.report('    Analyzing intensities...', force=True)
            for i in range(0, self.num_columns):
                if not self.graph.vertices[i].set_by_user:
                    graph_op.apply_intensity_score(self.graph, i, self.num_selections, self.intensities[self.alloy],
                                                   self.dist_8_std)
            self.report('    Intensity analysis complete.', force=True, update=True)
            self.report('    Running basic level definition...', force=True)
            self.report('        Could not set basic levels because it is not implemented', force=True)
            self.report('    Levels set.', force=True)
            self.report('    Finding particle....', force=True)
            graph_op.precipitate_controller(self.graph, starting_index)
            self.report('    Found particle.', force=True)
            self.report('    Running advanced level definition algorithm....', force=True)
            graph_op.set_levels(self.graph, starting_index, self.report, self.graph.vertices[starting_index].level, self.num_selections)
            self.report('    Levels set.', force=True, update=True)
            self.report('    adding edges to graph...', force=True)
            self.graph.redraw_edges()
            self.report('    Edges added.', force=True, update=True)
            self.report('    Summarizing stats', force=True)
            self.summarize_stats()
            self.report('    Starting weak untangling...', force=True)
            self.report('        Could not start weak untangling because it is not implemented yet!', force=True)
            self.report('    Starting strong untangling...', force=True)
            self.report('        Could not start strong untangling because it is not implemented yet!', force=True)
            self.report('Column characterization complete.', force=True)

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

        self.display_stats_string = ('Statistics:\n\n'
            'Number of detected columns: ' + str(self.num_columns) + '\n'
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
            'Number procentage of precipitate Un: ' + str(self.precipitate_number_percentage_un) + '\n\n')

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
