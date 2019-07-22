"""Module for recieving the results of a pca wizard!"""

# Program imports:
import core
import graph_op
import utils
# External imports:
import numpy as np
import csv
import copy
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import logging
# Instantiate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class VertexNumericData:

    """Accumulate, prepare and perform principal component analysis on vertex data.

        Principal component analysis (PCA) is a popular method for analysing data that relies on statistics and linear
        algebra. This wikipedia article gives a good outline for the mathematical basis:
        https://en.wikipedia.org/wiki/Principal_component_analysis

        This class has several similarities with the VertexDictData, but is designed to only work with nominal data
        for pca.

        :param files: List of file-paths in string format.
        :param keys: List of keys in string format, instructing the data accumulation on what parameters of the
            data to include.
        :type files: list(<string>)
        :type keys: list(<string>)

        .. code-block:: python
            :caption: Example

            # in this example, we will not accumulate real data, but instead provide fake testable data. Therefore the
            # call to VertexNumericData will use mock arguments, and we'll instead set the fields manually. See the
            >>> import data_module
            >>> import numpy as np
            >>> data_obj = data_module.VertexNumericData(None, ['id', 'attribute_1', 'attribute_2'])
            >>> data_obj.num_files = 1
            >>> data_obj.num_vertices = 10
            >>> id = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int)
            >>> attribute_1 = np.array([2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1])
            >>> attribute_2 = np.array([2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9])
            >>> id = np.reshape(id, (10, 1))
            >>> attribute_2 = np.reshape(attribute_2, (10, 1))
            >>> attribute_1 = np.reshape(attribute_1, (10, 1))
            >>> data_obj.data = np.concatenate([id, attribute_1, attribute_2], axis=1)
            >>> data_obj.attribute_data = np.concatenate((attribute_1, attribute_2), axis=1)
            >>> data_obj.attribute_keys = ['attribute_1', 'attribute_2']
            >>> data_obj.normalize_attribute_data()
            >>> data_obj.principal_component_analysis()
            >>> data_obj.pca_data
            array([[ 1.08643242, -0.22352364],
                   [-2.3089372 ,  0.17808082],
                   [ 1.24191895,  0.501509  ],
                   [ 0.34078247,  0.16991864],
                   [ 2.18429003, -0.26475825],
                   [ 1.16073946,  0.23048082],
                   [-0.09260467, -0.45331721],
                   [-1.48210777,  0.05566672],
                   [-0.56722643,  0.02130455],
                   [-1.56328726, -0.21536146]])


        """

    def __init__(self, files, keys):

        self.files = files
        self.num_files = 0
        self.num_vertices = 0

        self.data = None
        self.keys = keys
        self.attribute_data = None
        self.attribute_keys = None
        self.normalization_params = []

        self.cov_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None

        self.pca_data = None

        self.cu_pca_data = None
        self.si_pca_data = None
        self.al_pca_data = None
        self.mg_pca_data = None

    def plot(self):

        logger.info('Generating plot...')

        alpha = np.linspace(-10, 10, 1000)

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig)
        ax_pc_1 = fig.add_subplot(gs[0, 0])
        ax_pc_2 = fig.add_subplot(gs[1, 0])
        ax_scatter = fig.add_subplot(gs[:, 1])

        ax_pc_1.plot(alpha, utils.normal_dist(alpha, utils.mean_val(self.cu_pca_data[:, 2]), np.sqrt(utils.variance(self.cu_pca_data[:, 2]))),
                     'y', label='Cu ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(utils.mean_val(self.cu_pca_data[:, 2]), np.sqrt(utils.variance(self.cu_pca_data[:, 2]))))
        ax_pc_1.plot(alpha, utils.normal_dist(alpha, utils.mean_val(self.si_pca_data[:, 2]), np.sqrt(utils.variance(self.si_pca_data[:, 2]))),
                     'r', label='Si ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(utils.mean_val(self.si_pca_data[:, 2]), np.sqrt(utils.variance(self.si_pca_data[:, 2]))))
        ax_pc_1.plot(alpha, utils.normal_dist(alpha, utils.mean_val(self.al_pca_data[:, 2]), np.sqrt(utils.variance(self.al_pca_data[:, 2]))),
                     'g', label='Al ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(utils.mean_val(self.al_pca_data[:, 2]), np.sqrt(utils.variance(self.al_pca_data[:, 2]))))
        ax_pc_1.plot(alpha, utils.normal_dist(alpha, utils.mean_val(self.mg_pca_data[:, 2]), np.sqrt(utils.variance(self.mg_pca_data[:, 2]))),
                     'm', label='Mg ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(utils.mean_val(self.mg_pca_data[:, 2]), np.sqrt(utils.variance(self.mg_pca_data[:, 2]))))

        ax_pc_1.set_title('Principle component 1 fitted density')
        ax_pc_1.set_xlabel('Principle component 1')
        ax_pc_1.legend()

        ax_pc_2.plot(alpha, utils.normal_dist(alpha, utils.mean_val(self.cu_pca_data[:, 3]), np.sqrt(utils.variance(self.cu_pca_data[:, 3]))),
                     'y', label='Cu ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(utils.mean_val(self.cu_pca_data[:, 3]), np.sqrt(utils.variance(self.cu_pca_data[:, 3]))))
        ax_pc_2.plot(alpha, utils.normal_dist(alpha, utils.mean_val(self.si_pca_data[:, 3]), np.sqrt(utils.variance(self.si_pca_data[:, 3]))),
                     'r', label='Si ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(utils.mean_val(self.si_pca_data[:, 3]), np.sqrt(utils.variance(self.si_pca_data[:, 3]))))
        ax_pc_2.plot(alpha, utils.normal_dist(alpha, utils.mean_val(self.al_pca_data[:, 3]), np.sqrt(utils.variance(self.al_pca_data[:, 3]))),
                     'g', label='Al ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(utils.mean_val(self.al_pca_data[:, 3]), np.sqrt(utils.variance(self.al_pca_data[:, 3]))))
        ax_pc_2.plot(alpha, utils.normal_dist(alpha, utils.mean_val(self.mg_pca_data[:, 3]), np.sqrt(utils.variance(self.mg_pca_data[:, 3]))),
                     'm', label='Mg ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(utils.mean_val(self.mg_pca_data[:, 3]), np.sqrt(utils.variance(self.mg_pca_data[:, 3]))))

        ax_pc_2.set_title('Principle component 2 fitted density')
        ax_pc_2.set_xlabel('Principle component 2')
        ax_pc_2.legend()

        ax_scatter.scatter(self.cu_pca_data[:, 2], self.cu_pca_data[:, 3], c='y', label='Cu', s=4)
        ax_scatter.scatter(self.si_pca_data[:, 2], self.si_pca_data[:, 3], c='r', label='Si', s=4)
        ax_scatter.scatter(self.al_pca_data[:, 2], self.al_pca_data[:, 3], c='g', label='Al', s=4)
        ax_scatter.scatter(self.mg_pca_data[:, 2], self.mg_pca_data[:, 3], c='m', label='Mg', s=4)

        ax_scatter.set_title('Scatter-plot of two first principle components')
        ax_scatter.set_xlabel('PC 1')
        ax_scatter.set_ylabel('PC 2')
        ax_scatter.legend()

        fig.suptitle('Principle component analysis')

        logger.info('Plotted PCA alpha over {} files and {} vertices!'.format(self.num_files, self.num_vertices))

        plt.show()

    def principal_component_analysis(self, data=None):

        logger.info('Running principle component analysis...')

        if data is not None:
            # Do pca on the provided data instead of self
            pass
        else:
            self.cov_matrix = (np.cov(self.attribute_data.T))
            self.eigenvalues, self.eigenvectors = np.linalg.eig(self.cov_matrix)
            idx = np.argsort(self.eigenvalues)[::-1]
            self.eigenvalues = self.eigenvalues[idx]
            self.eigenvectors = self.eigenvectors[:, idx]
            self.pca_data = np.matmul(self.eigenvectors.T, self.attribute_data.T).T
            self.pca_data = np.concatenate((self.data[:, :2], self.pca_data), axis=1)

            for row in range(self.pca_data.shape[0]):
                entry = np.reshape(self.pca_data[row, :], (1, self.pca_data.shape[1]))
                if self.pca_data[row, 1] == 0.0:
                    if self.si_pca_data is None:
                        self.si_pca_data = entry
                    else:
                        self.si_pca_data = np.concatenate((self.si_pca_data, entry), axis=0)
                elif self.pca_data[row, 1] == 1.0:
                    if self.cu_pca_data is None:
                        self.cu_pca_data = entry
                    else:
                        self.cu_pca_data = np.concatenate((self.cu_pca_data, entry), axis=0)
                elif self.pca_data[row, 1] == 3.0:
                    if self.al_pca_data is None:
                        self.al_pca_data = entry
                    else:
                        self.al_pca_data = np.concatenate((self.al_pca_data, entry), axis=0)
                elif self.pca_data[row, 1] == 5.0:
                    if self.mg_pca_data is None:
                        self.mg_pca_data = entry
                    else:
                        self.mg_pca_data = np.concatenate((self.mg_pca_data, entry), axis=0)

    def normalize_attribute_data(self):

        logger.info('Normalizing data...')

        for column in range(self.attribute_data.shape[1]):
            mean = np.mean(self.attribute_data[:, column])
            std = np.std(self.attribute_data[:, column])
            # self.attribute_data[:, column] = self.attribute_data[:, column] - mean
            self.attribute_data[:, column] = (self.attribute_data[:, column] - mean) / std
            self.normalization_params.append((column, mean, std))

    def export(self, filename):
        with open(filename, mode='w', newline='') as csv_file:
            field_names = self.keys
            writer = csv.DictWriter(csv_file, field_names, delimiter=',', extrasaction='ignore')
            writer.writeheader()
            for observation in self.data:
                writer.writerow(observation)

    def accumulate_data(self, **kwargs):

        logger.info('Accumulating data...')

        filter_ = {'exclude_edges': True,
                   'exclude_matrix': False,
                   'exclude_hidden': False,
                   'exclude_1': False,
                   'exclude_2': False,
                   'exclude_3': False,
                   'exclude_4': False}

        for key, value in kwargs.items():
            if key in filter_:
                filter_[key] = value

        if any(special_item in self.keys for special_item in ['theta_variance', 'theta_min', 'theta_max']):
            theta = True
        else:
            theta = False

        if any(special_item in self.keys for special_item in ['alpha_min', 'alpha_max']):
            alpha = True
        else:
            alpha = False

        for file_ in iter(self.files.splitlines()):
            instance = core.SuchSoftware.load(file_)

            for vertex in instance.graph.vertices:

                if (not (filter_['exclude_edges'] and vertex.is_edge_column)) and \
                        (not (filter_['exclude_matrix'] and not vertex.is_in_precipitate)) and \
                        (not (filter_['exclude_hidden'] and not vertex.show_in_overlay)) and \
                        (not (filter_['exclude_1'] and vertex.flag_1)) and \
                        (not (filter_['exclude_2'] and vertex.flag_2)) and \
                        (not (filter_['exclude_3'] and vertex.flag_3)) and \
                        (not (filter_['exclude_4'] and vertex.flag_4)):

                    if theta:
                        sub_graph = instance.graph.get_atomic_configuration(vertex.i)
                        angles = [a.angles[0] for a in sub_graph.meshes]
                        theta_variance = sub_graph.central_mesh_angle_variance
                        theta_min = min(angles)
                        theta_max = max(angles)
                    else:
                        theta_variance = None
                        theta_min = None
                        theta_max = None

                    if alpha:
                        alpha_max, alpha_min = graph_op.base_angle_score(instance.graph, vertex.i, apply=False)
                    else:
                        alpha_max, alpha_min = None, None

                    values = []
                    for key in self.keys:

                        if key == 'id':
                            value = self.num_vertices
                        elif key == 'index':
                            value = vertex.i
                        elif key == 'h_index':
                            value = vertex.h_index
                        elif key == 'peak_gamma':
                            value = vertex.peak_gamma
                        elif key == 'average_gamma':
                            value = vertex.avg_gamma
                        elif key == 'normalized_peak_gamma':
                            value = vertex.normalized_peak_gamma
                        elif key == 'normalized_average_gamma':
                            value = vertex.normalized_avg_gamma
                        elif key == 'theta_variance':
                            value = theta_variance
                        elif key == 'theta_min':
                            value = theta_min
                        elif key == 'theta_max':
                            value = theta_max
                        elif key == 'alpha_min':
                            value = alpha_min
                        elif key == 'alpha_max':
                            value = alpha_max
                        else:
                            value = None
                            logger.warning('Unexpected key: {}'.format(key))

                        values.append(value)

                    values = np.array(values)
                    if self.num_vertices == 0:
                        self.data = np.reshape(values, (1, len(self.keys)))
                    else:
                        self.data = np.concatenate((self.data, np.reshape(values, (1, len(self.keys)))), axis=0)
                    self.num_vertices += 1

            print('matrix: {}'.format(self.data.shape))
            self.num_files += 1

        self.attribute_keys = copy.deepcopy(self.keys)
        num_non_numeric = 0
        if 'id' in self.keys:
            num_non_numeric += 1
            self.attribute_keys.remove('id')
        if 'index' in self.keys:
            num_non_numeric += 1
            self.attribute_keys.remove('index')
        if 'h_index' in self.keys:
            num_non_numeric += 1
            self.attribute_keys.remove('h_index')

        self.attribute_data = copy.deepcopy(self.data[:, num_non_numeric:])


class VertexDictData:

    def __init__(self, files, keys):

        self.files = files
        self.num_files = 0
        self.num_vertices = 0

        self.data = []
        self.matrix_data = None
        self.normalized_data = None
        self.transformation_params = []
        self.keys = keys

    def __str__(self):
        if len(self.data) == 0:
            string = 'empty'
        else:
            string = ''
            for key in self.keys:
                string += key
            string += '\n------------------------------------\n'
            string += str(self.data[0])
            string += '\n    .\n    .\n    .'
        return string

    def plot(self):
        pass

    def principal_component_analysis(self, data=None):
        if data is not None:
            # Do pca on the provided data instead of self
            pass
        else:
            pass

    def normalize_data(self):
        starting_column = 0
        if 'id' in self.keys:
            starting_column += 1
        if 'index' in self.keys:
            starting_column += 1
        if 'h_index' in self.keys:
            starting_column += 1

        self.normalized_data = np.array(self.matrix_data)
        for column in range(starting_column, self.matrix_data.shape[1]):
            mean = np.mean(self.matrix_data[:, column])
            std = np.std(self.matrix_data[:, column])
            self.matrix_data[:, column] = (self.matrix_data[:, column] - mean) / std
            self.transformation_params.append((column, mean, std))

    def export(self, filename):
        with open(filename, mode='w', newline='') as csv_file:
            field_names = self.keys
            writer = csv.DictWriter(csv_file, field_names, delimiter=',', extrasaction='ignore')
            writer.writeheader()
            for observation in self.data:
                writer.writerow(observation)

    def accumulate_data(self, **kwargs):

        logger.info('Accumulating data...')

        filter_ = {'exclude_edges': True,
                   'exclude_matrix': False,
                   'exclude_hidden': False,
                   'exclude_1': False,
                   'exclude_2': False,
                   'exclude_3': False,
                   'exclude_4': False}

        for key, value in kwargs:
            if key in filter_:
                filter_[key] = value

        if any(special_item in self.keys for special_item in ['theta_variance', 'theta_min', 'theta_max']):
            theta = True
        else:
            theta = False

        if any(special_item in self.keys for special_item in ['alpha_min', 'alpha_max']):
            alpha = True
        else:
            alpha = False

        for file_ in iter(self.files.splitlines()):
            instance = core.SuchSoftware.load(file_)

            for vertex in instance.graph.vertices:

                if (not (filter_['exclude_edges'] and vertex.is_edge_column)) and \
                        (not (filter_['exclude_matrix'] and not vertex.is_in_precipitate)) and \
                        (not (filter_['exclude_hidden'] and not vertex.show_in_overlay)) and \
                        (not (filter_['exclude_1'] and vertex.flag_1)) and \
                        (not (filter_['exclude_2'] and vertex.flag_2)) and \
                        (not (filter_['exclude_3'] and vertex.flag_3)) and \
                        (not (filter_['exclude_4'] and vertex.flag_4)):

                    observation = {}

                    if theta:
                        sub_graph = instance.graph.get_atomic_configuration(vertex.i)
                        angles = [a.angles[0] for a in sub_graph.meshes]
                        theta_variance = sub_graph.central_mesh_angle_variance
                        theta_min = min(angles)
                        theta_max = max(angles)
                    else:
                        theta_variance = None
                        theta_min = None
                        theta_max = None

                    if alpha:
                        alpha_max, alpha_min = graph_op.base_angle_score(instance.graph, vertex.i, apply=False)
                    else:
                        alpha_max, alpha_min = None, None

                    values = []
                    for key in self.keys:

                        if key == 'id':
                            value = self.num_vertices
                        elif key == 'index':
                            value = vertex.i
                        elif key == 'h_index':
                            value = vertex.h_index
                        elif key == 'peak_gamma':
                            value = vertex.peak_gamma
                        elif key == 'average_gamma':
                            value = vertex.avg_gamma
                        elif key == 'theta_variance':
                            value = theta_variance
                        elif key == 'theta_min':
                            value = theta_min
                        elif key == 'theta_max':
                            value = theta_max
                        elif key == 'alpha_min':
                            value = alpha_min
                        elif key == 'alpha_max':
                            value = alpha_max
                        else:
                            value = None
                            logger.warning('Unexpected key!')

                        observation[key] = value
                        values.append(value)

                    values = np.array(values)
                    print('values: {}'.format(values.shape))
                    if self.num_vertices == 0:
                        self.matrix_data = values
                    else:
                        self.matrix_data = np.concatenate((self.matrix_data, values), axis=0)
                    self.data.append(observation)
                    self.num_vertices += 1

            print('matrix: {}'.format(self.matrix_data.shape))
            self.num_files += 1








