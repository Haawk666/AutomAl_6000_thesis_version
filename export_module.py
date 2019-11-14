"""Module for recieving the results of an export wizard!"""

# Program imports:
import core
# External imports:
import csv
import logging
# Instantiate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class InstanceExport:

    def __init__(self, files):

        self.files = files
        self.num_files = 0
        self.num_vertices = 0

        self.data = []


class VertexExport:

    def __init__(self, files, keys):

        self.files = files
        self.num_files = 0
        self.num_vertices = 0

        self.data = []
        self.keys = keys

    def accumulate_data(self, exclude_edges=True, exclude_matrix=False, exclude_hidden=False, exclude_1=False,
                        exclude_2=False, exclude_3=False, exclude_4=False):

        logger.info('Accumulating data...')

        for file_ in iter(self.files.splitlines()):
            instance = core.SuchSoftware.load(file_)
            self.num_files += 1

            for vertex in instance.graph.vertices:
                if not (exclude_edges and vertex.is_edge_column):
                    if not (exclude_matrix and not vertex.is_in_precipitate):
                        if not (exclude_hidden and not vertex.show_in_overlay):
                            if not (exclude_1 and vertex.flag_1):
                                if not (exclude_2 and vertex.flag_2):
                                    if not (exclude_3 and vertex.flag_3):
                                        if not (exclude_4 and vertex.flag_4):

                                            if vertex.level == 1:
                                                image_height = instance.al_lattice_const / (2 * instance.scale)
                                                spatial_height = instance.al_lattice_const / 2
                                            else:
                                                image_height = 0
                                                spatial_height = 0

                                            alpha = instance.graph.produce_alpha_angles(vertex.i)
                                            theta = instance.graph.produce_theta_angles(vertex.i)
                                            if theta:
                                                theta_min = min(theta)
                                                theta_max = max(theta)
                                                theta_avg = sum(theta) / len(theta)
                                            else:
                                                theta_min = 0
                                                theta_max = 0
                                                theta_avg = 0
                                            redshift = instance.graph.produce_blueshift_sum(vertex.i)

                                            dict_ = {'id': self.num_vertices,
                                                     'index': vertex.i,
                                                     'species': vertex.species(),
                                                     'peak gamma': vertex.peak_gamma,
                                                     'average gamma': vertex.avg_gamma,
                                                     'normalized peak gamma': vertex.normalized_peak_gamma,
                                                     'normalized average gamma': vertex.normalized_avg_gamma,
                                                     'real x': vertex.real_coor_x,
                                                     'real y': vertex.real_coor_y,
                                                     'real z': image_height,
                                                     'spatial x': vertex.spatial_coor_x,
                                                     'spatial y': vertex.spatial_coor_y,
                                                     'spatial z': spatial_height,
                                                     'image x': vertex.im_coor_x,
                                                     'image y': vertex.im_coor_y,
                                                     'level': vertex.level,
                                                     'alpha min': min(alpha),
                                                     'alpha max': max(alpha),
                                                     'theta min': theta_min,
                                                     'theta max': theta_max,
                                                     'theta average': theta_avg,
                                                     'red-shift': redshift}

                                            self.data.append(dict_)

                                            self.num_vertices += 1

    def export(self, filename):

        with open(filename, mode='w', newline='') as csv_file:
            field_names = self.keys
            print(field_names)
            writer = csv.DictWriter(csv_file, field_names, delimiter=',', extrasaction='ignore')
            writer.writeheader()

            for dict_ in self.data:

                writer.writerow(dict_)
                print(dict_)


class EdgeExport:

    def __init__(self, files):

        self.files = files
        self.num_files = 0
        self.num_edges = 0

        self.data = []




