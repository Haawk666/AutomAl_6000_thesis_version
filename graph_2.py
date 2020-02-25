import utils
import numpy as np
import copy
import sys
import logging
# Instantiate logger:
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Vertex:

    species_strings = ['Si', 'Cu', 'Zn', 'Al', 'Ag', 'Mg', 'Un']
    species_symmetry = [3, 3, 3, 4, 4, 5, 3]
    al_lattice_const = 404.95

    def __init__(self, index, im_coor_x, im_coor_y, r, peak_gamma, avg_gamma, scale,
                 level=0, atomic_species='Un', species_index=6):

        # Index
        self.i = index

        # Some properties
        self.r = r
        self.scale = scale
        self.peak_gamma = peak_gamma
        self.avg_gamma = avg_gamma
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
        self.analysis_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        self.n = 3
        self.h_index = 6

        # Graph parameters
        self.in_degree = 0
        self.out_degree = 0
        self.degree = 0
        self.alpha_angles = []
        self.theta_angles = []
        self.theta_angle_variance = []
        self.normalized_peak_gamma = peak_gamma
        self.normalized_avg_gamma = avg_gamma
        self.avg_redshift = 0

        # Local graph mapping
        self.district = []
        self.out_neighbourhood = []
        self.in_neighbourhood = []
        self.neighbourhood = []
        self.anti_neighbourhood = []
        self.partners = []
        self.anti_partners = []

        self.reset_probability_vector()

    def __str__(self):
        return self.report()

    def report(self):
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

    def normalize_probability_vector(self):
        self.probability_vector = [a / sum(self.probability_vector) for a in self.probability_vector]

    def reset_probability_vector(self, bias=-1):
        self.probability_vector = [1, 1, 1, 1, 1, 1, 1]
        if bias == -1:
            self.probability_vector[6] = 1.1
        elif bias in [0, 1, 2, 3, 4, 5]:
            self.probability_vector[bias] = 1.1
        self.normalize_probability_vector()
        self.determine_species_from_probability_vector()

    def determine_species_from_probability_vector(self):
        self.h_index = self.probability_vector.index(max(self.probability_vector))
        self.atomic_species = Vertex.species_strings[self.h_index]
        self.n = Vertex.species_symmetry[self.h_index]

    def determine_species_from_h_index(self):
        self.probability_vector[self.h_index] = max(self.probability_vector) + 0.3
        self.normalize_probability_vector()
        self.atomic_species = Vertex.species_strings[self.h_index]
        self.n = Vertex.species_symmetry[self.h_index]

    def increment_h(self):
        if self.h_index == 5 or self.h_index == 6:
            return False
        else:
            self.h_index += 1
            self.determine_species_from_h_index()
            return True

    def decrement_h(self):
        if self.h_index == 0:
            return False
        else:
            self.h_index -= 1
            self.determine_species_from_h_index()
            return True

    def set_species_by_h_index(self, h_index):
        self.h_index = h_index
        self.determine_species_from_h_index()

    def permute_j_k(self, j, k):
        if j == k:
            return False

        pos_j = -1
        pos_k = -1
        if j in self.district:
            pos_j = self.district.index(j)
        if k in self.district:
            pos_k = self.district.index(k)

        if pos_j == -1 and pos_k == -1:
            self.district[-1] = k
            return True
        elif not pos_j == -1 and not pos_k == -1:
            self.district[pos_j], self.district[pos_k] = self.district[pos_k], self.district[pos_j]
            return True
        elif pos_j == -1:
            return False
        else:
            self.district[-1] = k
            self.district[pos_j], self.district[pos_k] = self.district[pos_k], self.district[pos_j]
            return True


class AtomicGraph:

    species_strings = ['Si', 'Cu', 'Zn', 'Al', 'Ag', 'Mg', 'Un']
    species_symmetry = [3, 3, 3, 4, 4, 5, 3]
    al_lattice_const = 404.95

    def __init__(self, scale, district_size=8):

        self.vertices = []
        self.vertex_indices = []
        self.arcs = []
        self.arc_indices = []
        self.anti_arcs = []
        self.anti_arc_indices = []

        self.particle_boarder_indices = []

        self.scale = scale
        self.district_size = district_size

        # Stats
        self.chi = 0
        self.order = 0
        self.size = 0
        self.anti_size = 0
        self.avg_degree = 0
        self.matrix_redshift = 0
        self.particle_redshift = 0
        self.total_redshift = 0

    def __str__(self):
        return self.report()

    def report(self):
        self.summarize_stats()
        string = 'Atomic Graph summary:\n'
        string += '    Scale: {}\n'.format(self.scale)
        string += '    Order: {}\n'.format(self.order)
        string += '    Size: {}\n'.format(self.size)
        string += '    Anti-size: {}\n'.format(self.anti_size)
        string += '    Chi: {}\n'.format(self.chi)
        string += '    Average degree: {}\n'.format(self.avg_degree)
        string += '    Matrix redshift: {}\n'.format(self.matrix_redshift)
        string += '    Particle redshift: {}\n'.format(self.particle_redshift)
        string += '    Total redshift: {}\n'.format(self.total_redshift)
        return string

    def add_vertex(self, vertex):
        self.vertices.append(vertex)
        if not vertex.i == len(self.vertices) - 1:
            logger.error('Vertex index out of sync! Overwriting index! (This is indicative of something going awry!)')
            vertex.i = len(self.vertices) - 1
        self.vertex_indices.append(vertex.i)
        self.order += 1

    def remove_vertex(self, i):
        logger.info('Removing vertex {}'.format(i))

        # Find all vertices that have i in its district and re_write their districts:
        for vertex in self.vertices:
            if i in vertex.district:
                new_district = []

                all_distances = []
                sorted_indices = []
                sorted_distances = []

    def get_vertex_objects_from_indices(self, vertex_indices):
        vertices = []
        for index in vertex_indices:
            vertices.append(self.vertices[index])
        return vertices

    def get_alpha_angles(self, i):
        pass

    def get_theta_angles(self, i):
        pass

    def get_separation(self, i, j):
        pass

    def get_projected_separation(self, i, j):
        pass

    def map_district(self, i, search_extended_district=False):
        vertex = self.vertices[i]
        # Determine out-neighbourhood
        vertex.out_neighbourhood = vertex.district[:vertex.n]

        # determine in-neighbourhood
        vertex.in_neighbourhood = []
        if not search_extended_district:
            for co_citizen in self.get_vertex_objects_from_indices(vertex.district):
                if i in co_citizen.district[:co_citizen.n]:
                    vertex.in_neighbourhood.append(co_citizen.i)
        else:
            for co_citizen in self.vertices:
                if i in co_citizen.district[:co_citizen.n]:
                    vertex.in_neighbourhood.append(co_citizen.i)

        # Determine neighbourhood
        vertex.neighbourhood = vertex.out_neighbourhood
        for in_neighbour in vertex.in_neighbourhood:
            if in_neighbour not in vertex.neighbourhood:
                vertex.neighbourhood.append(in_neighbour)

        # Determine anti-neighbourhood
        vertex.anti_neighbourhood = []
        for co_citizen in vertex.district:
            if co_citizen not in vertex.neighbourhood:
                vertex.anti_neighbourhood.append(co_citizen)

        # Determine partners and anti-partners
        vertex.partners = []
        vertex.anti_partners = []
        for neighbour in vertex.neighbourhood:
            if neighbour in vertex.in_neighbourhood and neighbour in vertex.out_neighbourhood:
                vertex.partners.append(neighbour)
            else:
                vertex.anti_partners.append(neighbour)

        vertex.in_degree = len(vertex.in_neighbourhood)
        vertex.out_degree = len(vertex.out_neighbourhood)
        vertex.degree = len(vertex.neighbourhood)

    def map_districts(self, search_extended_district=False):
        for i in self.vertex_indices:
            self.map_district(i, search_extended_district=search_extended_district)

    def calc_redshift(self, i):
        pass

    def summarize_stats(self):

        self.map_districts()

        # Calc order
        self.order = len(self.vertices)

        # Calc size
        self.size = 0
        for vertex in self.vertices:
            self.size += len(vertex.out_neighbourhood)

        # Calc chi (# weak arcs / # num strong arcs)
        num_weak_arcs = 0
        for vertex in self.vertices:
            if not vertex.is_edge_column:
                num_weak_arcs += len(vertex.anti_neighbourhood)
        num_weak_arcs = num_weak_arcs / 2
        self.chi = num_weak_arcs / self.size

        # Calc average degree
        counted_columns = 0
        degrees = 0
        for vertex in self.vertices:
            if not vertex.is_edge_column:
                degrees += vertex.degree
                counted_columns += 1
        self.avg_degree = degrees / counted_columns

        # Calc redshifts

















