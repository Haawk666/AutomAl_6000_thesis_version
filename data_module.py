import core
from matplotlib import pyplot as plt
import graph_op
import numpy as np
import csv
import utils
from matplotlib.gridspec import GridSpec
from matplotlib import ticker as tick
import logging

# Instantiate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class VertexData:

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

    def principal_component_analysis(self):
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








