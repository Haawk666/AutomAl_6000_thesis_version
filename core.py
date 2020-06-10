"""Module for the 'SuchSoftware' class that handles a *project instance*."""

# Program imports:
import utils
import graph_2
import graph_op
import compatibility
import legacy_items
import untangling
import column_characterization
import statistics
# External imports:
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import scipy.ndimage
import dm3_lib as dm3
import sys
import copy
import time
import pickle
import configparser
import logging
# Instantiate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Project:
    """The main API through which to build and access the data extracted from HAADF-STEM images.

    :param filename_full: The full path and/or relative path and filename of the .dm3 image to import. A project can
        be instantiated with filename_full='empty', but this is only meant to be used as a placeholder.
    :param debug_obj: An instance of an AutomAl 6000 GUI.MainUI(). Optional, default=None.
    :type filename_full: string
    :type debug_obj: <GUI.MainUI()>

    """

    # Version
    version = [0, 1, 0]

    default_dict = {
        'Si_1': {'symmetry': 3, 'atomic_species': 'Si', 'atomic_radii': 117.50, 'color': (255, 20, 20), 'species_color': (255, 20, 20), 'description': 'Q-prime Si'},
        'Si_2': {'symmetry': 3, 'atomic_species': 'Si', 'atomic_radii': 117.50, 'color': (235, 40, 40), 'species_color': (255, 20, 20), 'description': 'Beta-pprime Si'},
        'Cu_1': {'symmetry': 3, 'atomic_species': 'Cu', 'atomic_radii': 127.81, 'color': (255, 255, 20), 'species_color': (255, 255, 20), 'description': ''},
        'Al_1': {'symmetry': 4, 'atomic_species': 'Al', 'atomic_radii': 143.00, 'color': (20, 255, 20), 'species_color': (20, 255, 20), 'description': ''},
        'Al_2': {'symmetry': 4, 'atomic_species': 'Al', 'atomic_radii': 143.00, 'color': (40, 235, 40), 'species_color': (20, 255, 20), 'description': ''},
        'Mg_1': {'symmetry': 5, 'atomic_species': 'Mg', 'atomic_radii': 160.00, 'color': (138, 43, 226), 'species_color': (138, 43, 226), 'description': ''},
        'Mg_2': {'symmetry': 5, 'atomic_species': 'Mg', 'atomic_radii': 160.00, 'color': (118, 63, 206), 'species_color': (138, 43, 226), 'description': ''},
        'Un_1': {'symmetry': 3, 'atomic_species': 'Un', 'atomic_radii': 100.00, 'color': (20, 20, 255), 'species_color': (20, 20, 255), 'description': ''}
    }

    # District size
    district_size = 8

    al_lattice_const = 404.95

    def __init__(self, filename_full, debug_obj=None, species_dict=None):

        self.filename_full = filename_full
        self.im_mat = None
        self.fft_im_mat = None
        self.scale = 1
        self.im_height = 0
        self.im_width = 0
        self.version_saved = None
        self.starting_index = None

        # In AutomAl 6000, each column is modelled by a single atomic species, more advanced categorization can be
        # applied however. Each advanced category must map to one of the simple categories and also to a symmetry.
        parser = configparser.ConfigParser()
        parser.read('config.ini')
        if species_dict is None:
            self.species_dict = Project.default_dict
        else:
            self.species_dict = species_dict

        # For communicating with the interface, if any:
        self.debug_obj = debug_obj
        self.debug_mode = False

        self.im_meta_data = {}
        if not (filename_full == 'Empty' or filename_full == 'empty'):
            dm3f = dm3.DM3(self.filename_full)
            self.im_mat = dm3f.imagedata
            (self.scale, _) = dm3f.pxsize
            self.scale = 1000 * self.scale  # Scale is now in nm/pixel
            if self.scale > 7.0:
                self.im_mat = scipy.ndimage.zoom(self.im_mat, 2, order=1)
                self.scale = self.scale / 2
            (self.im_height, self.im_width) = self.im_mat.shape
            self.im_mat = utils.normalize_static(self.im_mat)
            self.fft_im_mat = utils.gen_fft(self.im_mat)

        # Data matrices:
        self.search_mat = copy.deepcopy(self.im_mat)

        # Counting and statistical variables.
        self.num_columns = 0

        # These are hyper-parameters of the algorithms. See the documentation.
        self.threshold = 0.2586
        self.search_size = 10
        self.r = int(100 / self.scale)
        self.overhead = int(6 * (self.r / 10))

        # Initialize an empty graph
        self.graph = graph_2.AtomicGraph(self.scale, active_model=None, species_dict=self.species_dict)

        logger.info('Generated instance from {}'.format(filename_full))

    def __str__(self):
        return self.report()

    def report(self, supress_log=True):
        """Build a string representation of current statistical summary.

        """

        string = 'Project summary:\n'
        for line in self.graph.report().splitlines(keepends=True):
            string += '    ' + line
        string += '    General:\n'
        string += '        Number of columns: {}\n'.format(self.num_columns)

        if supress_log:
            return string
        else:
            logger.info(string)
            return None

    def vertex_report(self, i, supress_log=False):
        string = self.graph.vertices[i].report()
        if supress_log:
            return string
        else:
            logger.info(string)
            return None

    def get_alloy_string(self):
        species_string = ''
        species = set()
        for sp in self.species_dict.values():
            species.add(sp['atomic_species'])
        species.remove('Un')
        if 'Al' in species:
            species_string += 'Al-'
            species.remove('Al')
        if 'Mg' in species:
            species_string += 'Mg-'
            species.remove('Mg')
        if 'Si' in species:
            species_string += 'Si-'
            species.remove('Si')
        if 'Cu' in species:
            species_string += 'Cu-'
            species.remove('Cu')
        for remaining in species:
            species_string += '{}-'.format(remaining)
        return species_string[:-1]

    def save(self, filename_full):
        """Save the current project as a pickle file.

        :param filename_full: Path and name of save-file. The project will be pickled as *filename_full* without any
            file-extension identifier.
        :type filename_full: string

        """

        with open(filename_full, 'wb') as f:
            self.debug_obj = None
            self.version_saved = self.version
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        logger.info('Saved {}'.format(filename_full))

    @staticmethod
    def load(filename_full):
        """Load an instance from a pickle-file.

        Will use the compatibility module if the saved version is different from the current version. Returns None if
        loading failed.

        :param filename_full: Path-name of the file to be loaded.
        :type filename_full: string

        :return: project instance.
        :rtype: core.Project

        """
        with open(filename_full, 'rb') as f:
            try:
                obj = pickle.load(f)
            except:
                obj = None
                logger.error('Failed to load save-file!')
            else:
                if not obj.version_saved == Project.version:
                    logger.info('Attempted to load un-compatible save-file. Running conversion script...')
                    obj = compatibility.convert(obj, obj.version_saved)
                    if obj is None:
                        logger.error('Conversion unsuccessful, failed to load save-file!')
                    else:
                        logger.info('Conversion successful, loaded {}'.format(filename_full))
                else:
                    logger.info('Loaded {}'.format(filename_full))
        return obj

    def column_detection(self, search_type='s', plot=False):
        """Column detection algorithm.

        """
        peak_values = []
        non_edge_peak_values = []
        if len(self.graph.vertices) == 0:
            logger.info('Starting column detection. Search mode is \'{}\''.format(search_type))
            self.search_mat = copy.deepcopy(self.im_mat)
            cont = True
        else:
            logger.info('Continuing column detection. Search mode is \'{}\''.format(search_type))
            min_val = 2
            for vertex in self.graph.vertices:
                if vertex.peak_gamma < min_val:
                    min_val = vertex.peak_gamma
            if min_val < self.threshold:
                logger.info('Columns overdetected. Rolling back..')
                new_graph = graph_2.AtomicGraph(
                    self.scale,
                    active_model=self.graph.active_model,
                    species_dict=self.graph.species_dict
                )
                self.num_columns = 0
                self.search_mat = copy.deepcopy(self.im_mat)
                for vertex in self.graph.get_non_void_vertices():
                    if vertex.peak_gamma > self.threshold:
                        self.num_columns += 1
                        peak_values.append(vertex.peak_gamma)
                        new_graph.add_vertex(vertex)
                        self.search_mat = utils.delete_pixels(
                            self.search_mat,
                            int(vertex.im_coor_x),
                            int(vertex.im_coor_y),
                            self.r + self.overhead
                        )
                self.graph = new_graph
                cont = False
            else:
                self.redraw_search_mat()
                for vertex in self.graph.vertices:
                    peak_values.append(vertex.peak_gamma)
                cont = True

        counter = self.num_columns
        original_counter = copy.deepcopy(counter)

        time_1 = time.time()

        while cont:

            pos = np.unravel_index(self.search_mat.argmax(), (self.im_height, self.im_width))
            max_val = self.search_mat[pos]
            peak_values.append(max_val)

            x_fit, y_fit = utils.cm_fit(self.im_mat, pos[1], pos[0], self.r)
            x_fit_int = int(x_fit)
            y_fit_int = int(y_fit)

            self.search_mat = utils.delete_pixels(self.search_mat, x_fit_int, y_fit_int, self.r + self.overhead)

            vertex = graph_2.Vertex(counter, x_fit, y_fit, self.r, self.scale, parent_graph=self.graph)
            vertex.avg_gamma, vertex.peak_gamma = utils.circular_average(self.im_mat, x_fit_int, y_fit_int, self.r)
            if not max_val == vertex.peak_gamma:
                logger.debug(
                    'Vertex {}\n    Pos: ({}, {})\n    Fit: ({}, {})\n    max_val: {}\n    peak_gamma: {}\n'.format(
                        counter,
                        pos[1],
                        pos[0],
                        x_fit,
                        y_fit,
                        max_val,
                        vertex.peak_gamma
                    )
                )
            self.graph.add_vertex(vertex)

            self.num_columns += 1
            counter += 1

            if search_type == 's':
                if counter >= self.search_size:
                    cont = False
            elif search_type == 't':
                if np.max(self.search_mat) < self.threshold:
                    cont = False
            else:
                logger.error('Invalid search_type')

        self.summarize_stats()

        self.find_edge_columns()
        for vertex in self.graph.vertices:
            if not vertex.is_edge_column:
                non_edge_peak_values.append(vertex.peak_gamma)

        time_2 = time.time()

        logger.info('Column detection complete! Found {} columns in {} seconds.'.format(
            self.num_columns - original_counter,
            time_2 - time_1
        ))

        if plot:
            fig = plt.figure(constrained_layout=True)
            gs = GridSpec(4, 1, figure=fig)
            ax_values = fig.add_subplot(gs[0, 0])
            ax_slope = fig.add_subplot(gs[1, 0])
            ax_cum_var = fig.add_subplot(gs[2, 0])
            ax_cum_var_slope = fig.add_subplot(gs[3, 0])

            ax_values.plot(
                range(0, len(peak_values)),
                peak_values,
                c='k',
                label='Column peak intensity'
            )
            ax_values.set_title('Column detection summary')
            ax_values.set_xlabel('# Column')
            ax_values.set_ylabel('Peak intensity')
            ax_values.legend()

            slope = [0]
            for ind in range(1, len(peak_values)):
                slope.append(peak_values[ind] - peak_values[ind - 1])

            ax_slope.plot(
                range(0, len(peak_values)),
                slope,
                c='b',
                label='2 point slope of peak intensity'
            )
            ax_slope.set_title('Slope')
            ax_slope.set_xlabel('# Column')
            ax_slope.set_ylabel('slope')
            ax_slope.legend()

            cumulative_slope_variance = [0]
            cum_slp_var_slp = [0]  # lol
            for ind in range(1, len(peak_values)):
                cumulative_slope_variance.append(utils.variance(slope[0:ind]))
                cum_slp_var_slp.append(cumulative_slope_variance[ind] - cumulative_slope_variance[ind - 1])

            ax_cum_var.plot(
                range(0, len(non_edge_peak_values)),
                non_edge_peak_values,
                c='r',
                label='None edge peak values'
            )
            ax_cum_var.set_title('Cumulative variance')
            ax_cum_var.set_xlabel('# Column')
            ax_cum_var.set_ylabel('$\\sigma^2$')
            ax_cum_var.legend()

            ax_cum_var_slope.plot(
                range(0, len(peak_values)),
                cum_slp_var_slp,
                c='g',
                label='Slope of cumulative slope variance'
            )
            ax_cum_var_slope.set_title('Variance slope')
            ax_cum_var_slope.set_xlabel('# Column')
            ax_cum_var_slope.set_ylabel('2-point slope approximation')
            ax_cum_var_slope.legend()

            inflection_points = []
            for ind in range(1, len(cum_slp_var_slp)):
                if cum_slp_var_slp[ind] > cum_slp_var_slp[ind - 1]:
                    inflection_points.append(ind)

            plt.show()

    def column_characterization(self, starting_index, search_type=0, ui_obj=None):
        """Column characterization algorithm.

        Assumes a *starting_index*, that is taken to be an Al column in the matrix. The *search_type* enables access to
        the different sub-proccesses of the algorithm. These are

        ===================     ==================================
        :code:`search_type`     Process
        ===================     ==================================
        0
        1
        2
        3
        4
        5
        6
        7
        8
        9
        10
        11
        12
        13
        14
        ===================     ==================================

        :param starting_index: Index of a vertex that is a part of the matrix
        :param search_type: (optional, default=0) Which sub-process to access.
        :type starting_index: int
        :type search_type: int

        """

        sys.setrecursionlimit(10000)

        if search_type == 0:
            self.column_characterization(starting_index, search_type=1, ui_obj=ui_obj)
            self.column_characterization(starting_index, search_type=2, ui_obj=ui_obj)

        elif search_type == 1:
            logger.info('Doing the basics...')
            # Spatial map:
            self.column_characterization(starting_index, search_type=3, ui_obj=ui_obj)
            # Detect edges:
            self.column_characterization(starting_index, search_type=6, ui_obj=ui_obj)
            # Map connectivity:
            self.column_characterization(starting_index, search_type=16, ui_obj=ui_obj)
            # Zeta analysis:
            self.column_characterization(starting_index, search_type=5, ui_obj=ui_obj)
            # Alpha model:
            self.column_characterization(starting_index, search_type=7, ui_obj=ui_obj)
            # Find particle:
            self.column_characterization(starting_index, search_type=8, ui_obj=ui_obj)
            # Calc gamma:
            self.column_characterization(starting_index, search_type=9, ui_obj=ui_obj)
            # Map connectivity:
            self.column_characterization(starting_index, search_type=4, ui_obj=ui_obj)
            # Composite model:
            # self.column_characterization(starting_index, search_type=11, ui_obj=ui_obj)
            logger.info('Basics done')

        elif search_type == 2:
            logger.info('Running models and untangling...')

        elif search_type == 3:
            # Run spatial mapping
            logger.info('Mapping spatial locality...')
            column_characterization.map_districts(self.graph)
            logger.info('Spatial mapping complete.')

        elif search_type == 4:
            # Basic zeta
            logger.info('Running basic zeta analysis...')
            column_characterization.zeta_analysis(
                self.graph,
                starting_index,
                self.graph.vertices[starting_index].zeta,
                use_n=False,
                method='separation'
            )
            logger.info('zeta\'s set.')

        elif search_type == 5:
            # Advanced zeta
            logger.info('Running advanced zeta...')
            column_characterization.zeta_analysis(
                self.graph,
                starting_index,
                self.graph.vertices[starting_index].zeta,
                use_n=True,
                method='partners'
            )
            logger.info('zeta\'s set.')

        elif search_type == 6:
            # Identify edge columns
            logger.info('Finding edge columns....')
            column_characterization.find_edge_columns(self.graph, self.im_width, self.im_height)
            for vertex in self.graph.vertices:
                if vertex.is_edge_column:
                    self.graph.set_species(vertex.i, 'Al_1')
            logger.info('Edge columns found.')

        elif search_type == 7:
            # Applying alpha model
            logger.info('Calculating probabilities from alpha attributes...')
            column_characterization.apply_alpha_model(self.graph)
            logger.info('Calculated probabilities from alpha attributes.')

        elif search_type == 8:
            # Particle detection
            logger.info('Finding particle...')
            column_characterization.particle_detection(self.graph)
            logger.info('Found particle.')

        elif search_type == 9:
            # Determine normalized intensities
            logger.info('Finding normalized intensities...')
            self.normalize_gamma()
            logger.info('Found intensities.')

        elif search_type == 10:
            # Evaluate sub-species:
            logger.info('Evaluating advanced species...')
            self.graph.evaluate_sub_categories()
            logger.info('Advanced species set.')

        elif search_type == 11:
            # Applying composite model
            logger.info('Calculating probabilities from all attributes...')
            column_characterization.apply_composite_model(self.graph)
            logger.info('Calculated probabilities from all attributes.')

        elif search_type == 12:
            # reset probs
            logger.info('Resetting probability vectors with zero bias...')
            for vertex in self.graph.vertices:
                if not vertex.void and not vertex.is_set_by_user and not vertex.is_edge_column:
                    vertex.reset_probability_vector()
            logger.info('Probability vectors reset.')

        elif search_type == 13:
            # Reset user-input
            logger.info('Resetting user-set columns...')
            for i in range(0, self.num_columns):
                if self.graph.vertices[i].is_set_by_user:
                    self.graph.vertices[i].is_set_by_user = False
            logger.info('User-set columns was re-set.')

        elif search_type == 14:
            # Locate and remove edge intersections
            logger.info('Looking for intersections')
            intersections = self.graph.find_intersections()
            num_intersections = len(intersections)
            column_characterization.arc_intersection_denial(self.graph)
            intersections = self.graph.find_intersections()
            if ui_obj is not None:
                ui_obj.update_overlay()
                ui_obj.update_graph()
            logger.info('Found {} intersections'.format(num_intersections))
            logger.info('{} literal intersections still remain'.format(len(intersections)))

        elif search_type == 15:
            pass

        elif search_type == 16:
            # Run local graph mapping
            logger.info('Mapping vertex connectivity...')
            self.graph.build_local_maps(build_out=True)
            self.graph.build_local_zeta_maps(build_out=True)
            logger.info('Vertices mapped.')

        elif search_type == 17:
            # Run local graph mapping
            logger.info('Mapping vertex connectivity...')
            self.graph.build_local_maps(build_out=False)
            self.graph.build_local_zeta_maps(build_out=False)
            logger.info('Vertices mapped.')

        elif search_type == 18:
            pass

        elif search_type == 19:
            pass

        elif search_type == 20:
            pass

        elif search_type == 21:
            pass

        elif search_type == 22:
            pass

        elif search_type == 23:
            pass

        else:
            logger.error('No such search type!')

    def find_edge_columns(self):
        """Locate vertices that are close to the edge of the image.

        These vertices get special treatment throughout the program because information about their surroundings will
        be incomplete. This method will find all vertices that are within a distance 6 * self.r from the edge, and set
        the field self.graph.vertices[i].is_edge_column = True.

        """

        for vertex in self.graph.vertices:

            x_coor = vertex.im_coor_x
            y_coor = vertex.im_coor_y
            margin = 4 * self.r

            if x_coor < margin or x_coor > self.im_width - margin - 1 or y_coor < margin or y_coor > self.im_height - margin - 1:
                self.graph.vertices[vertex.i].is_edge_column = True
            else:
                self.graph.vertices[vertex.i].is_edge_column = False

    def normalize_gamma(self):
        """Find the mean of the intensity of the Al-matrix, and scale all intensities.

        Scale all intensities such that the mean of the Al-matrix is a fixed point. Store the result in each vertex *i*
        :code:`self.graph.vertices[i].normalized_peak_gamma` and
        :code:`self.graph.vertices[i].normalized_avg_gamma` fields.

        """
        self.graph.calc_normalized_gamma()

    def calc_avg_gamma(self):
        """Calculate average intensity for every vertex based on image information.

        """
        if self.graph.order > 0:
            for vertex in self.graph.vertices:
                vertex.avg_gamma, vertex.peak_gamma = utils.circular_average(
                    self.im_mat,
                    int(vertex.im_coor_x),
                    int(vertex.im_coor_y),
                    self.r
                )

    def summarize_stats(self):
        """Summarize current stats about the project file.

        """
        pass

    def redraw_search_mat(self):
        """Redraw the search matrix.

        """

        self.search_mat = copy.deepcopy(self.im_mat)
        if self.num_columns > 0:
            for vertex in self.graph.vertices:
                self.search_mat = utils.delete_pixels(
                    self.search_mat,
                    int(vertex.im_coor_x),
                    int(vertex.im_coor_y),
                    self.r + self.overhead)

    def get_im_length_from_spatial(self, spatial_length):
        return self.scale * spatial_length

    def get_spatial_length_from_im(self, im_length):
        return im_length / self.scale

