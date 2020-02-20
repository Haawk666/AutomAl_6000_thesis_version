import utils
import numpy as np
import copy
import sys
import logging
# Instantiate logger:
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Vertex:

    def __init__(self, index, im_coor_x, im_coor_y, r, peak_gamma, avg_gamma, scale,
                 level=0, atomic_species='Un', species_index=6):

        # Index
        self.i = index

        # Some properties
        self.r = r
        self.scale = scale
        self.peak_gamma = peak_gamma
        self.avg_gamma = avg_gamma
        self.normalized_peak_gamma = peak_gamma
        self.normalized_avg_gamma = avg_gamma
        self.atomic_species = atomic_species
        self.species_index = species_index

        # Position
        self.im_coor_x = im_coor_x
        self.im_coor_y = im_coor_y
        self.im_coor_z = level
        self.spatial_coor_x = im_coor_x * scale
        self.spatial_coor_y = im_coor_y * scale
        self.spatial_coor_z = level * 0.5 * 404.95

        # Some settings
        self.is_in_precipitate = False
        self.is_edge_column = False
        self.is_set_by_user = False
        self.show_in_overlay = True
        self.flag_1 = False
        self.flag_2 = False
        self.flag_3 = False
        self.flag_4 = False
        self.flag_5 = False
        self.flag_6 = False
        self.flag_7 = False
        self.flag_8 = False
        self.flag_9 = False

        # Self-analysis
        self.probability_vector = [0, 0, 0, 0, 0, 0, 1]
        self.n = 3

        # Local graph mapping
        self.district = []
        self.out_neighbourhood = []
        self.in_neighbourhood = []
        self.neighbourhood = []
        self.anti_neighbourhood = []
        self.partners = []
        self.anti_partners = []

    def __str__(self):
        im_pos = self.im_pos()
        spatial_pos = self.spatial_pos()
        string = 'Vertex {}:\n'.format(self.i)
        string += '    real image position (x, y) = ({}, {})\n'.format(im_pos[0], im_pos[1])
        string += '    pixel image position (x, y) = ({}, {})\n'.format(np.floor(im_pos[0]), np.floor(im_pos[1]))
        string += '    spatial relative position in pm (x, y) = ({}, {})\n'.format(spatial_pos[0], spatial_pos[1])
        string += '    peak gamma = {}\n'.format(self.peak_gamma)
        string += '    average gamma = {}\n'.format(self.avg_gamma)
        string += '    species: {}'.format(self.atomic_species)
        return string

    def im_pos(self):
        return self.im_coor_x, self.im_coor_y, self.im_coor_z

    def spatial_pos(self):
        return self.spatial_coor_x, self.spatial_coor_y, self.spatial_coor_z



