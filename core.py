"""Module for the 'SuchSoftware' class that handles a *project instance*."""

# Program imports:
import mat_op
import graph_2
import utils
import graph_op
import compatibility
import legacy_items
import untangling
import params
# External imports:
import numpy as np
import dm3_lib as dm3
import sys
import pickle
# from matplotlib import pyplot as plt
import logging
# Instantiate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SuchSoftware:
    """The main API through which to build and access the data extracted from HAADF-STEM images.

    :param filename_full: The full path and/or relative path and filename of the .dm3 image to import. A project can
        be instantiated with filename_full='empty', but this is only meant to be used as a placeholder.
    :param debug_obj: DEPRECATED
    :type filename_full: string

    .. code-block:: python
        :caption: Example

        >>> import core
        >>> my_project_instance = core.SuchSoftware('Saves/sample.dm3')
        >>> print(my_project_instance)
        Image summary: ----------
            Number of detected columns: 0
            Number of detected precipitate columns: 0

            Number of inconsistencies: 0
            Number of popular: 0
            Number of unpopular: 0
            Chi: 0

            Average peak intensity: 0
            Average average intensity: 0

            Average Si peak intensity: 0.0
            Average Cu peak intensity: 0.0
            Average Zn peak intensity: 0.0
            Average Al peak intensity: 0.0
            Average Ag peak intensity: 0.0
            Average Mg peak intensity: 0.0
            Average Un peak intensity: 0.0

            Average Si average intensity: 0.0
            Average Cu average intensity: 0.0
            Average Zn average intensity: 0.0
            Average Al average intensity: 0.0
            Average Ag average intensity: 0.0
            Average Mg average intensity: 0.0
            Average Un average intensity: 0.0

            Number of Si-columns: 0
            Number of Cu-columns: 0
            Number of Zn-columns: 0
            Number of Al-columns: 0
            Number of Ag-columns: 0
            Number of Mg-columns: 0
            Number of Un-columns: 0

            Number procentage of Si: 0.0
            Number procentage of Cu: 0.0
            Number procentage of Zn: 0.0
            Number procentage of Al: 0.0
            Number procentage of Ag: 0.0
            Number procentage of Mg: 0.0
            Number procentage of Un: 0.0

            Number of precipitate Si-columns: 0
            Number of precipitate Cu-columns: 0
            Number of precipitate Zn-columns: 0
            Number of precipitate Al-columns: 0
            Number of precipitate Ag-columns: 0
            Number of precipitate Mg-columns: 0
            Number of precipitate Un-columns: 0

            Number procentage of precipitate Si: 0.0
            Number procentage of precipitate Cu: 0.0
            Number procentage of precipitate Zn: 0.0
            Number procentage of precipitate Al: 0.0
            Number procentage of precipitate Ag: 0.0
            Number procentage of precipitate Mg: 0.0
            Number procentage of precipitate Un: 0.0

    """

    # Version
    version = [0, 0, 11]

    # Number of elements in the probability vectors
    num_selections = 7

    # Number of closest neighbours that are included in local search-spaces
    map_size = 8

    # Al lattice constant in pico-meters
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
        self.graph = graph_2.AtomicGraph(self.scale)

        logger.info('Generated instance from {}'.format(filename_full))

        # DEPRECATED:
        self.dist_1_std = 0
        self.dist_2_std = 0
        self.dist_3_std = 0
        self.dist_4_std = 0
        self.dist_5_std = 0
        self.dist_8_std = 0

    def __str__(self):
        return self.report()

    def report(self, supress_log=True):
        """Build a string representation of current statistical summary and store it in self.stats_string.

        """

        string = 'Project summary:\n'
        for line in self.graph.report().splitlines(keepends=True):
            string += '    ' + line
        string += '    General:\n'
        string += '        Number of columns: {}\n'.format(self.num_columns)
        string += '        Number of particle columns: {}\n'.format(self.num_precipitate_columns)
        string += '    Average peak intensities:\n'
        string += '        Average peak intensity: {:.2f}\n'.format(self.avg_peak_gamma)
        string += '        Average Si peak intensity: {:.2f}\n'.format(self.avg_si_peak_gamma)
        string += '        Average Cu peak intensity: {:.2f}\n'.format(self.avg_cu_peak_gamma)
        string += '        Average Zn peak intensity: {:.2f}\n'.format(self.avg_zn_peak_gamma)
        string += '        Average Al peak intensity: {:.2f}\n'.format(self.avg_al_peak_gamma)
        string += '        Average Ag peak intensity: {:.2f}\n'.format(self.avg_ag_peak_gamma)
        string += '        Average Mg peak intensity: {:.2f}\n'.format(self.avg_mg_peak_gamma)
        string += '        Average Un peak intensity: {:.2f}\n'.format(self.avg_un_peak_gamma)
        string += '    Average average intensities:\n'
        string += '        Average average intensity: {:.2f}\n'.format(self.avg_avg_gamma)
        string += '        Average Si average intensity: {:.2f}\n'.format(self.avg_si_avg_gamma)
        string += '        Average Cu average intensity: {:.2f}\n'.format(self.avg_cu_avg_gamma)
        string += '        Average Zn average intensity: {:.2f}\n'.format(self.avg_zn_avg_gamma)
        string += '        Average Al average intensity: {:.2f}\n'.format(self.avg_al_avg_gamma)
        string += '        Average Ag average intensity: {:.2f}\n'.format(self.avg_ag_avg_gamma)
        string += '        Average Mg average intensity: {:.2f}\n'.format(self.avg_mg_avg_gamma)
        string += '        Average Un average intensity: {:.2f}\n'.format(self.avg_un_avg_gamma)
        string += '    Image composition:\n'
        string += '        Number of Si-columns: {}\n'.format(self.num_si)
        string += '        Number of Cu-columns: {}\n'.format(self.num_cu)
        # string += '        Number of Zn-columns: {}\n'.format(self.num_zn)
        string += '        Number of Al-columns: {}\n'.format(self.num_al)
        # string += '        Number of Ag-columns: {}\n'.format(self.num_ag)
        string += '        Number of Mg-columns: {}\n'.format(self.num_mg)
        string += '        Number of Un-columns: {}\n'.format(self.num_un)
        string += '        Number percentage of Si: {:.4f}\n'.format(self.number_percentage_si)
        string += '        Number percentage of Cu: {:.4f}\n'.format(self.number_percentage_cu)
        # string += '        Number percentage of Zn: {:.4f}\n'.format(self.number_percentage_zn)
        string += '        Number percentage of Al: {:.4f}\n'.format(self.number_percentage_al)
        # string += '        Number percentage of Ag: {:.4f}\n'.format(self.number_percentage_ag)
        string += '        Number percentage of Mg: {:.4f}\n'.format(self.number_percentage_mg)
        string += '        Number percentage of Un: {:.4f}\n'.format(self.number_percentage_un)
        string += '    Particle composition:\n'
        string += '        Number of precipitate Si-columns: {}\n'.format(self.num_precipitate_si)
        string += '        Number of precipitate Cu-columns: {}\n'.format(self.num_precipitate_cu)
        # string += '        Number of precipitate Zn-columns: {}\n'.format(self.num_precipitate_zn)
        string += '        Number of precipitate Al-columns: {}\n'.format(self.num_precipitate_al)
        # string += '        Number of precipitate Ag-columns: {}\n'.format(self.num_precipitate_ag)
        string += '        Number of precipitate Mg-columns: {}\n'.format(self.num_precipitate_mg)
        string += '        Number of precipitate Un-columns: {}\n'.format(self.num_precipitate_un)
        string += '        Number percentage of precipitate Si: {:.4f}\n'.format(self.precipitate_number_percentage_si)
        string += '        Number percentage of precipitate Cu: {:.4f}\n'.format(self.precipitate_number_percentage_cu)
        # string += '        Number percentage of precipitate Zn: {:.4f}\n'.format(self.precipitate_number_percentage_zn)
        string += '        Number percentage of precipitate Al: {:.4f}\n'.format(self.precipitate_number_percentage_al)
        # string += '        Number percentage of precipitate Ag: {:.4f}\n'.format(self.precipitate_number_percentage_ag)
        string += '        Number percentage of precipitate Mg: {:.4f}\n'.format(self.precipitate_number_percentage_mg)
        string += '        Number percentage of precipitate Un: {:.4f}\n'.format(self.precipitate_number_percentage_un)

        if supress_log:
            return string
        else:
            logger.info(string)
            return None

    def alloy_string(self):
        """Get a string representation of the currently active alloy matrix.

        :return: string representation of the currently active alloy
        :rtype: string

        """
        if self.alloy == 0:
            return 'Alloy: Al-Mg-Si-(Cu)'
        elif self.alloy == 1:
            return 'Alloy: Al-Mg-Si'
        else:
            return 'Alloy: Unknown'

    def vertex_report(self, i, supress_log=False):
        string = self.graph.vertices[i].report()
        if supress_log:
            return string
        else:
            logger.info(string)
            return None

    def set_alloy_mat(self):
        """Set the alloy vector field of the project, :code:`self.alloy_mat`, based on the value of :code:`self.alloy`.

        This function will set the alloy vector based on the field :code:`self.alloy`. The interpretation of the alloy vector is
        that each element is 0 or 1 depending on weather the corresponding element is present in the image. The
        corresponding elements are [Si, Cu, Zn, Al, Ag, Mg, Un], where Un is a placeholder for an *Unknown* element. As
        an example, the alloy vector for Al-Mg-Si would be [1, 0, 0, 1, 0, 1, 0]. The currently implemented alloys are:

        ===================     ==================================  ===============
        :code:`self.alloy`      :code:`self.alloy_mat`              Alloy
        ===================     ==================================  ===============
        0                       [1, 1, 0, 1, 0, 1, 0]               Al-Mg-Si-(Cu)
        1                       [1, 0, 0, 1, 0, 1, 0]               Al-Mg-Si
        ===================     ==================================  ===============

        """

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
        :rtype: core.SuchSoftware

        """
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
        """Column detection algorithm.

        The column detection algorithm will attempt to locate the columns in a HAADF-STEM image. When a SuchSoftware
        object is instantiated, the image data is stored in self.im_mat, which is a numpy array with values normalized
        to the range (0, 1). In broad terms the algorithm works by identifying the brightest pixel in self.im_mat. It
        then does a centre-of-mass calculation using pixel values from a circular area around the max pixel, with radius
        self.r (Which is determined at init based on the scale information in the .dm3 metadata). It then sets all
        pixels in an area slightly larger than this to 0 in self.search_mat. The next max pixel will then be identified
        from the search_mat, while the CM-calculation will still be made from the im_mat.

        It will continue searching until an end condition is met, which depends on the search mode. If search mode is
        \'s\', it will search until it finds self.search_size number of columns. If search mode is \'t\', it will search
        until the brightest pixel is below self.threshold. For more details, see [link].

        :param search_type: (Optional, default=\'s\') The search mode for the algorithm.
        :type search_type: string

        """
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

            vertex = graph_2.Vertex(counter, x_fit_real_coor, y_fit_real_coor, self.r, max_val, 0, self.scale)
            vertex.reset_probability_vector(bias=6)
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
        """Use the image to determine the closest neighbours of vertex *i*.

        :param i: Index of the vertex for which closest neighbours are to be determined.
        :param n: How many neighbours to map.
        :param weight: (Optional, default=4) Used for internal recursion. If n neighbours was not found by searching a
            pre-set search area, then it will call itself with an increased weight, effectively increasing the search
            area.
        :type i: int
        :type n: int
        :type weight: int

        :return: returns tuple of a python list with the indices of the *n* closest neighbours of vertex *i* and a
            python list with the (projected) distance to each neighbour.
        :rtype: tuple(list(<int>), list(<float>)).

        """

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
                        dist = self.graph.get_projected_separation(i, j)
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
            logger.debug('Did not find enough neighbours for vertex {}. increasing search area.'.format(i))

        else:

            logger.debug('Found {} total neighbours for vertex {}'.format(total_num, i))

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
        """Locate vertices that are close to the edge of the image.

        These vertices get special treatment throughout the program because information about their surroundings will
        be incomplete. This method will find all vertices that are within a distance 6 * self.r from the edge, and set
        the field self.graph.vertices[i].is_edge_column = True.

        """

        for y in range(0, self.num_columns):

            x_coor = self.graph.vertices[y].spatial_coor_x
            y_coor = self.graph.vertices[y].spatial_coor_y
            margin = 6 * self.r

            if x_coor < margin or x_coor > self.im_width - margin - 1 or y_coor < margin or y_coor > self.im_height - margin - 1:

                self.graph.vertices[y].is_edge_column = True
                self.graph.vertices[y].reset_prob_vector(bias=3)

            else:

                self.graph.vertices[y].is_edge_column = False

    def normalize_gamma(self):
        """Find the mean of the intensity of the Al-matrix, and scale all intensities.

        Scale all intensities such that the mean of the Al-matrix is a fixed point. Store the result in each vertex *i*
        :code:`self.graph.vertices[i].normalized_peak_gamma` and
        :code:`self.graph.vertices[i].normalized_avg_gamma` fields.

        """
        self.graph.calc_normalized_gamma()

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
            logger.info('Setting alloy...')
            self.set_alloy_mat()
            logger.info('Alloy set.')
            # Reset prob vectors:
            self.column_characterization(starting_index, search_type=12)
            # Spatial mapping:
            self.column_characterization(starting_index, search_type=3)
            if ui_obj is not None:
                ui_obj.update_overlay()
                ui_obj.update_graph()
            # Angle analysis:
            self.column_characterization(starting_index, search_type=16)
            if ui_obj is not None:
                ui_obj.update_overlay()
                ui_obj.update_graph()
            # Find particle:
            self.column_characterization(starting_index, search_type=5)
            # Set levels:
            self.column_characterization(starting_index, search_type=6)
            # Add edges:
            self.column_characterization(starting_index, search_type=4)
            if ui_obj is not None:
                ui_obj.update_overlay()
                ui_obj.update_graph()
            # Search for intersections
            self.column_characterization(starting_index, search_type=14, ui_obj=ui_obj)
            # Map subsets:
            self.column_characterization(starting_index, search_type=21)
            # Summarize:
            logger.info('Summarizing stats.')
            self.summarize_stats()
            logger.info('Basics done')

        elif search_type == 2:

            logger.info('Running models and untangling...')
            # Weak untangling
            self.column_characterization(starting_index, search_type=10, ui_obj=ui_obj)
            # Base stat score:
            self.column_characterization(starting_index, search_type=22)
            if ui_obj is not None:
                ui_obj.update_overlay()
                ui_obj.update_graph()
            # Weak untangling
            self.column_characterization(starting_index, search_type=10, ui_obj=ui_obj)
            # Find particle
            self.column_characterization(starting_index, search_type=5)
            # Set levels:
            self.column_characterization(starting_index, search_type=6)
            self.column_characterization(starting_index, search_type=7)
            # Calc normalized gamma:
            self.column_characterization(starting_index, search_type=19)
            # Base stat score:
            self.column_characterization(starting_index, search_type=22)
            if ui_obj is not None:
                ui_obj.update_overlay()
                ui_obj.update_graph()
            # Weak untangling
            self.column_characterization(starting_index, search_type=10, ui_obj=ui_obj)
            # Base model score:
            self.column_characterization(starting_index, search_type=22)
            # Weak untangling
            self.column_characterization(starting_index, search_type=10, ui_obj=ui_obj)
            # Find particle:
            self.column_characterization(starting_index, search_type=5)
            # Set levels:
            self.column_characterization(starting_index, search_type=7)
            # Add edges:
            self.column_characterization(starting_index, search_type=4)
            # Search for intersections
            self.column_characterization(starting_index, search_type=14, ui_obj=ui_obj)
            # Map subsets:
            self.column_characterization(starting_index, search_type=21)
            # Summarize:
            logger.info('Summarizing stats.')
            self.summarize_stats()
            # Complete:
            logger.info('Column characterization complete.')

        elif search_type == 3:
            # Run spatial mapping
            logger.info('Mapping spatial locality...')
            self.redraw_centre_mat()
            self.redraw_circumference_mat()
            for i in range(0, self.num_columns):
                self.graph.vertices[i].neighbour_indices, _ = self.find_nearest(i, self.map_size)
            self.find_edge_columns()
            logger.info('Spatial mapping complete.')

        elif search_type == 4:
            # redraw edges
            logger.info('Adding edges to graph...')
            self.graph.redraw_edges()
            logger.info('Edges added.')

        elif search_type == 5:
            # Legacy particle detection
            logger.info('Finding particle with legacy method....')
            legacy_items.precipitate_controller(self.graph, starting_index)
            # graph_op.precipitate_controller(self.graph, starting_index)
            logger.info('Found particle.')

        elif search_type == 6:
            # Legacy level determination
            logger.info('Running legacy level definition algorithm....')
            legacy_items.define_levels(self.graph, starting_index, self.graph.vertices[starting_index].level)
            logger.info('Levels set.')

        elif search_type == 7:
            # Experimental level determination
            logger.info('Running experimental level definition algorithm....')
            self.graph.reset_all_flags()
            self.graph.sort_all_subsets_by_distance()
            graph_op.naive_determine_z(self.graph, starting_index, self.graph.vertices[starting_index].level)
            graph_op.revise_z(self.graph)
            graph_op.revise_z(self.graph)
            logger.info('Levels set.')

        elif search_type == 8:
            # Aggressive weak untangling
            logger.info('Starting experimental weak untangling...')

            self.column_characterization(starting_index, search_type=14, ui_obj=ui_obj)

            static = False
            total_changes = 0
            total_counter = 0

            while not static:

                for type_num in range(1, 7):

                    cont = True
                    counter = 0
                    while cont:
                        self.graph.redraw_edges()
                        chi_before = self.graph.chi
                        logger.info('Looking for type {}:'.format(type_num))
                        logger.info('Chi: {}'.format(chi_before))
                        self.graph.sort_all_subsets_by_distance()

                        num_types, changes = untangling.untangle(self.graph, type_num, strong=False, ui_obj=ui_obj, aggressive=True)

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

            self.column_characterization(starting_index, search_type=14, ui_obj=ui_obj)

            self.graph.redraw_edges()
            logger.info('Weak untangling complete')

        elif search_type == 9:
            # Basic Weak untangling
            logger.info('Starting experimental weak untangling...')

            self.column_characterization(starting_index, search_type=14)

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
                        self.graph.sort_all_subsets_by_distance()

                        num_types, changes = untangling.untangle(self.graph, type_num, strong=False, ui_obj=ui_obj)

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

            self.column_characterization(starting_index, search_type=14)

            self.graph.redraw_edges()
            logger.info('Weak untangling complete')

        elif search_type == 10:
            # Weak untangling
            logger.info('Starting experimental weak untangling...')

            self.column_characterization(starting_index, search_type=14, ui_obj=ui_obj)

            static = False
            total_changes = 0
            total_counter = 0

            while not static:

                for type_num in range(1, 7):

                    cont = True
                    counter = 0
                    while cont:
                        self.graph.redraw_edges()
                        chi_before = self.graph.chi
                        logger.info('Looking for type {}:'.format(type_num))
                        logger.info('Chi: {}'.format(chi_before))
                        self.graph.sort_all_subsets_by_distance()

                        num_types, changes = untangling.untangle(self.graph, type_num, strong=False, ui_obj=ui_obj)

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

            self.column_characterization(starting_index, search_type=14, ui_obj=ui_obj)

            self.graph.redraw_edges()
            logger.info('Weak untangling complete')

        elif search_type == 11:
            # Strong untangling
            logger.info('Starting experimental strong untangling...')

            static = False
            total_changes = 0
            total_counter = 0

            while not static:

                for type_num in range(1, 7):

                    cont = True
                    counter = 0
                    while cont:
                        self.graph.redraw_edges()
                        chi_before = self.graph.chi
                        self.column_characterization(starting_index, search_type=14)
                        logger.info('Looking for type {}:'.format(type_num))
                        logger.info('Chi: {}'.format(chi_before))
                        self.graph.sort_all_subsets_by_distance()

                        num_types, changes = untangling.untangle(self.graph, type_num, strong=True)

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
            logger.info('Strong untangling complete')

        elif search_type == 12:
            # reset probs
            logger.info('Resetting probability vectors with zero bias...')
            for i in range(0, self.num_columns):
                if not self.graph.vertices[i].set_by_user and not self.graph.vertices[i].is_edge_column:
                    self.graph.vertices[i].reset_prob_vector()
            logger.info('Probability vectors reset.')

        elif search_type == 13:
            # Reset user-input
            logger.info('Resetting user-set columns...')
            for i in range(0, self.num_columns):
                if self.graph.vertices[i].set_by_user:
                    self.graph.vertices[i].set_by_user = False
            logger.info('User-set columns was re-set.')

        elif search_type == 14:
            # Locate and remove edge intersections
            logger.info('Looking for intersections')
            intersections = self.graph.find_intersects()
            num_intersections = len(intersections)
            # not_removed, strong_intersections, ww, ss = graph_op.remove_intersections(self.graph)
            graph_op.experimental_remove_intersections(self.graph)
            intersections = self.graph.find_intersects()
            self.graph.map_friends()
            if ui_obj is not None:
                ui_obj.update_overlay()
                ui_obj.update_graph()
            logger.info('Found {} intersections'.format(num_intersections))
            # self.report('        Found {} strong intersections'.format(ss), force=True)
            # self.report('        Found {} weak-weak intersections'.format(ww), force=True)
            # self.report('        {} weak intersections were not removed'.format(not_removed), force=True)
            logger.info('{} literal intersections still remain'.format(len(intersections)))

        elif search_type == 15:
            pass

        elif search_type == 16:
            # Alpha angle analysis
            logger.info('Running experimental angle analysis')
            for i in range(0, self.num_columns):
                if not self.graph.vertices[i].set_by_user and not self.graph.vertices[i].is_edge_column:
                    self.graph.vertices[i].reset_symmetry_vector()
                    self.graph.vertices[i].reset_prob_vector()
                    self.graph.vertices[i].prob_vector =\
                        graph_op.base_angle_score(self.graph, i)
                    self.graph.vertices[i].prob_vector = np.array(self.graph.vertices[i].prob_vector)
                    self.graph.vertices[i].define_species()
            logger.info('Angle analysis complete!')

        elif search_type == 17:
            pass

        elif search_type == 18:
            # Find edge columns
            logger.info('Finding edge columns....')
            self.find_edge_columns()
            for vertex in self.graph.vertices:
                if vertex.is_edge_column and not vertex.set_by_user:
                    vertex.reset_prob_vector(bias=3)
                    vertex.reset_symmetry_vector(bias=-1)
            logger.info('Found edge columns.')

        elif search_type == 19:
            # Determine normalized intensities
            logger.info('Finding normalized intensities...')
            self.normalize_gamma()
            logger.info('Found intensities.')

        elif search_type == 20:
            # Mesh analysis with strong resolve
            logger.info('Running mesh analysis...')
            self.graph.map_meshes(starting_index)
            changes = untangling.mesh_analysis(self.graph)
            self.graph.map_meshes(starting_index)
            logger.info('Mesh analysis complete. Made {} changes'.format(changes))

        elif search_type == 21:
            # Sort neighbours
            logger.info('Sorting neighbours...')
            self.graph.sort_all_subsets_by_distance()
            logger.info('Neighbours sorted')

        elif search_type == 22:
            # product predictions
            logger.info('Apply product predictions')
            self.graph.map_friends()
            probs = []
            for i in range(0, self.num_columns):
                if not self.graph.vertices[i].set_by_user and not self.graph.vertices[i].is_edge_column:
                    probs.append(np.array(graph_op.base_stat_score(self.graph, i, get_individual_predictions=True)[8]))
                else:
                    probs.append([0, 0, 0, 0, 0, 0, 0])
            for i in range(0, self.num_columns):
                if not self.graph.vertices[i].set_by_user and not self.graph.vertices[i].is_edge_column:
                    self.graph.vertices[i].prob_vector = probs[i]
                    self.graph.vertices[i].define_species()
            self.graph.map_friends()
            logger.info('Applied product predictions!')

        elif search_type == 23:
            # Model predictions
            logger.info('Apply model predictions')
            self.graph.map_friends()
            probs = []
            for i in range(0, self.num_columns):
                if not self.graph.vertices[i].set_by_user and not self.graph.vertices[i].is_edge_column:
                    probs.append(np.array(graph_op.base_stat_score(self.graph, i, get_individual_predictions=False)))
                else:
                    probs.append([0, 0, 0, 0, 0, 0, 0])
            for i in range(0, self.num_columns):
                if not self.graph.vertices[i].set_by_user and not self.graph.vertices[i].is_edge_column:
                    self.graph.vertices[i].prob_vector = probs[i]
                    self.graph.vertices[i].define_species()
            self.graph.map_friends()
            logger.info('Applied model predictions!')

        else:
            logger.error('No such search type!')

    def calc_avg_gamma(self):
        """Calculate average intensity for every vertex based on image information.

        """
        if self.num_columns > 0:
            temp_mat = mat_op.gen_framed_mat(self.im_mat, self.r)
            for vertex in self.graph.vertices:
                vertex.avg_gamma, vertex.peak_gamma = mat_op.average(temp_mat, int(vertex.im_coor_x + self.r),
                                                                     int(vertex.im_coor_y + self.r), self.r)

    def summarize_stats(self):
        """Summarize current stats about the project file.

        """

        logger.info('Summarizing stats...')

        self.graph.refresh_graph()

        self.avg_peak_gamma = 0
        self.avg_avg_gamma = 0

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

            for vertex in self.graph.vertices:
                if not vertex.is_edge_column and not vertex.void:

                    self.avg_peak_gamma += vertex.peak_gamma
                    self.avg_avg_gamma += vertex.avg_gamma

                    if vertex.species_index == 0:
                        self.num_si += 1
                        self.number_percentage_si += 1
                        self.avg_si_peak_gamma += vertex.peak_gamma
                        self.avg_si_avg_gamma += vertex.avg_gamma
                        if vertex.is_in_precipitate:
                            self.num_precipitate_columns += 1
                            self.num_precipitate_si += 1
                            self.precipitate_number_percentage_si += 1
                    elif vertex.species_index == 1:
                        self.num_cu += 1
                        self.number_percentage_cu += 1
                        self.avg_cu_peak_gamma += vertex.peak_gamma
                        self.avg_cu_avg_gamma += vertex.avg_gamma
                        if vertex.is_in_precipitate:
                            self.num_precipitate_columns += 1
                            self.num_precipitate_cu += 1
                            self.precipitate_number_percentage_cu += 1
                    elif vertex.species_index == 2:
                        self.num_zn += 1
                        self.number_percentage_zn += 1
                        self.avg_zn_peak_gamma += vertex.peak_gamma
                        self.avg_zn_avg_gamma += vertex.avg_gamma
                        if vertex.is_in_precipitate:
                            self.num_precipitate_columns += 1
                            self.num_precipitate_zn += 1
                            self.precipitate_number_percentage_zn += 1
                    elif vertex.species_index == 3:
                        self.num_al += 1
                        self.number_percentage_al += 1
                        self.avg_al_peak_gamma += vertex.peak_gamma
                        self.avg_al_avg_gamma += vertex.avg_gamma
                        if vertex.is_in_precipitate:
                            self.num_precipitate_columns += 1
                            self.num_precipitate_al += 1
                            self.precipitate_number_percentage_al += 1
                    elif vertex.species_index == 4:
                        self.num_ag += 1
                        self.number_percentage_ag += 1
                        self.avg_ag_peak_gamma += vertex.peak_gamma
                        self.avg_ag_avg_gamma += vertex.avg_gamma
                        if vertex.is_in_precipitate:
                            self.num_precipitate_columns += 1
                            self.num_precipitate_ag += 1
                            self.precipitate_number_percentage_ag += 1
                    elif vertex.species_index == 5:
                        self.num_mg += 1
                        self.number_percentage_mg += 1
                        self.avg_mg_peak_gamma += vertex.peak_gamma
                        self.avg_mg_avg_gamma += vertex.avg_gamma
                        if vertex.is_in_precipitate:
                            self.num_precipitate_columns += 1
                            self.num_precipitate_mg += 1
                            self.precipitate_number_percentage_mg += 1
                    elif vertex.species_index == 6:
                        self.num_un += 1
                        self.number_percentage_un += 1
                        self.avg_un_peak_gamma += vertex.peak_gamma
                        self.avg_un_avg_gamma += vertex.avg_gamma
                        if vertex.is_in_precipitate:
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

        logger.info('Collected stats.')

    def redraw_search_mat(self):
        """Redraw the search matrix.

        """

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
        """Redraw the circumference matrix. DEPRECATED"""

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
        """Redraw the centre matrix."""

        self.column_centre_mat = np.zeros((self.im_height, self.im_width, 2), dtype=type(self.im_mat))
        if self.num_columns > 0:
            for x in range(0, self.num_columns):
                self.column_centre_mat[self.graph.vertices[x].im_coor_y, self.graph.vertices[x].im_coor_x, 0] = 1
                self.column_centre_mat[self.graph.vertices[x].im_coor_y, self.graph.vertices[x].im_coor_x, 1] = x

    def reset_graph(self):
        """Reset the graph.

        """
        self.graph = graph_2.AtomicGraph(self.scale)
        self.num_columns = 0
        self.redraw_centre_mat()
        self.redraw_circumference_mat()
        self.redraw_search_mat()
        self.summarize_stats()

    def reset_vertex_properties(self):
        """Reset all vertex properties.

        """
        self.graph.reset_vertex_properties()
        self.summarize_stats()


