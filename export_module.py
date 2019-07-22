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
                                            else:
                                                image_height = 0

                                            dict_ = {'id': self.num_vertices,
                                                     'index': vertex.i,
                                                     'species': vertex.species(),
                                                     'peak gamma': vertex.peak_gamma,
                                                     'average gamma': vertex.avg_gamma,
                                                     'real x': vertex.real_coor_x,
                                                     'real y': vertex.real_coor_y,
                                                     'spatial x': vertex.spatial_coor_x,
                                                     'spatial y': vertex.spatial_coor_y,
                                                     'image x': vertex.im_coor_x,
                                                     'image y': vertex.im_coor_y,
                                                     'level': vertex.level,
                                                     'image height': image_height,
                                                     'spatial height': image_height * instance.scale}

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




