# This file contains the classes of graph objects

import numpy as np
import utils


class Vertex:

    def __init__(self, index, x, y, r, peak_gamma, avg_gamma, alloy_mat, num_selections=7, level=0, atomic_species='Un', h_index=6,
                 species_strings=None, certainty_threshold=0.8):

        self.i = index
        self.real_coor_x = x
        self.real_coor_y = y
        self.im_coor_x = int(np.floor(x))
        self.im_coor_y = int(np.floor(y))
        self.r = r
        self.peak_gamma = peak_gamma
        self.avg_gamma = avg_gamma
        self.num_selections = num_selections
        self.level = level
        self.atomic_species = atomic_species
        self.h_index = h_index
        self.alloy_mat = alloy_mat

        self.confidence = 0.0
        self.certainty_threshold = certainty_threshold
        self.is_in_precipitate = False
        self.set_by_user = False
        self.flag_1 = False
        self.flag_2 = False
        self.flag_3 = False
        self.flag_4 = False
        self.is_unpopular = False
        self.is_popular = False
        self.is_edge_column = False
        self.show_in_overlay = True
        self.prob_vector = np.ndarray([self.num_selections], dtype=np.float64)
        self.collapsed_prob_vector = np.zeros([self.num_selections], dtype=int)
        self.collapsed_prob_vector[self.num_selections - 1] = 1
        self.neighbour_indices = []
        self.partner_indices = []

        # The prob_vector is ordered to represent the elements in order of their radius:

        # Si
        # Cu
        # Zm
        # Al
        # Ag
        # Mg
        # Un
        self.species_strings = None
        if species_strings is None:
            self.species_strings = ['Si', 'Cu', 'Zn', 'Al', 'Ag', 'Mg', 'Un']
        else:
            self.species_strings = species_strings

        self.reset_prob_vector(bias=self.num_selections - 1)

    def n(self):

        n = 3

        if self.h_index == 0 or self.h_index == 1:
            n = 3
        elif self.h_index == 3:
            n = 4
        elif self.h_index == 5:
            n = 5

        return n

    def reset_prob_vector(self, bias=-1):
        self.prob_vector = np.ones([self.num_selections], dtype=np.float64)

        if not bias == -1:
            self.prob_vector[bias] = 1.1

        self.collapsed_prob_vector = np.zeros([self.num_selections], dtype=int)

        if not bias == -1:
            self.collapsed_prob_vector[bias] = 1
        else:
            self.collapsed_prob_vector[self.num_selections - 1] = 1

        for k in range(0, self.num_selections):
            self.prob_vector[k] *= self.alloy_mat[k]

        self.renorm_prob_vector()
        self.define_species()

    def collapse_prob_vector(self):
        self.define_species()
        self.prob_vector = self.collapsed_prob_vector

    def renorm_prob_vector(self):
        for k in range(0, self.num_selections):
            self.prob_vector[k] *= self.alloy_mat[k]
        vector_sum = np.sum(self.prob_vector)
        if vector_sum <= 0.00000001:
            pass
        else:
            correction_factor = 1 / vector_sum
            self.prob_vector = correction_factor * self.prob_vector

    def define_species(self):

        h_prob = 0.0
        h_index = 0

        for y in range(0, self.num_selections):

            if self.prob_vector[y] >= h_prob:
                h_prob = self.prob_vector[y]
                h_index = y

        if not h_prob > 0.0:
            self.atomic_species = self.species_strings[self.num_selections - 1]
            self.h_index = self.num_selections - 1
        else:
            self.atomic_species = self.species_strings[h_index]
            self.h_index = h_index

        self.collapsed_prob_vector = np.zeros([self.num_selections], dtype=int)
        self.collapsed_prob_vector[h_index] = 1

        self.analyse_prob_vector_confidence()

    def analyse_prob_vector_confidence(self):

        h_value = self.prob_vector.max()
        nh_value = 0.0
        h_index = self.prob_vector.argmax()
        is_certain = False

        for x in range(0, self.num_selections):
            if h_value > self.prob_vector[x] >= nh_value:
                nh_value = self.prob_vector[x]

        if h_value > 0.0:
            self.confidence = 1 - nh_value / h_value
            if self.confidence >= self.certainty_threshold:
                is_certain = True
            if self.confidence == 0:
                h_index = 6
        else:
            h_index = 6

        if h_value - nh_value < 0.00001:
            h_index = 6

        if self.set_by_user:
            self.confidence = 1.0
            is_certain = True

        return h_index, is_certain

    def force_species(self, h_index):
        self.h_index = h_index
        self.atomic_species = self.species_strings[h_index]
        self.confidence = 1.0
        self.set_by_user = True
        self.reset_prob_vector(bias=h_index)
        self.collapse_prob_vector()

    def anti_level(self):
        if self.level == 0:
            anti_level = 1
        elif self.level == 1:
            anti_level = 0
        elif self.level == 2:
            anti_level = None
        return anti_level

    def partners(self):
        if not len(self.neighbour_indices) == 0:
            self.partner_indices = []
            for i in range(0, self.n()):
                self.partner_indices.append(self.neighbour_indices[i])
            return self.partner_indices

    def print(self):
        print('\nVertex properties:\n----------')
        print('Index: {}\nImage pos: ({}, {})\nReal pos: ({}, {})'.format(self.i, self.im_coor_x, self.im_coor_y,
                                                                          self.real_coor_x, self.real_coor_y))
        print('Atomic Species: {}'.format(self.atomic_species))
        print('Probability vector: {}'.format(self.prob_vector))


class Edge:

    def __init__(self, vertex_a, vertex_b, index):

        # Initialize and edge with direction from vertex a -> vertex b.

        self.vertex_a = vertex_a
        self.vertex_b = vertex_b
        self.i = index
        self.vector = np.array([self.vertex_b.real_coor_x - self.vertex_a.real_coor_x,
                                self.vertex_b.real_coor_y - self.vertex_a.real_coor_y])
        self.magnitude = self.length()
        self.position_of_index_b_in_a = None
        self.position_of_index_a_in_b = None
        self.is_consistent_edge = True
        self.is_reciprocated = True
        self.is_legal_levels = True
        self.edge_category = 0

        self.find_self_map()
        self.is_consistent()
        self.determine_edge_category()

    def determine_edge_category(self):
        pass

    def find_self_map(self):
        pass

    def is_consistent(self):

        index_i = self.vertex_a.i
        index_j = self.vertex_b.i

        is_consistent = True
        is_illegal_levels = False
        is_reciprocated = True

        if self.vertex_a.level == self.vertex_b.level:
            is_illegal_levels = True
            is_consistent = False

        found_a = False

        for ind in self.vertex_a.partners():
            if ind == index_j:
                found_a = True

        if not found_a:
            print('Unexpected error in graph.Edge.is_consistent()')
            is_consistent = False
            is_reciprocated = False

        found_b = False

        for ind in self.vertex_b.partners():
            if ind == index_i:
                found_b = True

        if not found_b:
            is_consistent = False
            is_reciprocated = False

        self.is_consistent_edge = is_consistent
        self.is_reciprocated = is_reciprocated
        self.is_legal_levels = not is_illegal_levels

        return is_consistent

    def length(self):
        # Find the length of the vector
        delta_x = self.vertex_b.real_coor_x - self.vertex_a.real_coor_x
        delta_y = self.vertex_b.real_coor_y - self.vertex_a.real_coor_y
        arg = delta_x ** 2 + delta_y ** 2
        return np.sqrt(arg)

    def angle(self):
        # Find the angle between the edge and the horizontal defined on [0, 2pi)
        horizontal_unit_vector = np.array([0, 1])
        norm_factor = 1 / self.length()
        edge_unit_vector = norm_factor * self.vector
        angle = utils.find_angle(horizontal_unit_vector[0], edge_unit_vector[0], horizontal_unit_vector[1],
                                 edge_unit_vector[1])
        return angle


class AtomicGraph:

    def __init__(self, map_size=8):

        self.vertices = []
        self.vertex_indices = []
        self.edges = []
        self.particle_boarder = []

        self.map_size = map_size

        # Stats
        self.chi = 0
        self.num_vertices = len(self.vertices)
        self.num_particle_vertices = 0
        self.num_edges = 0
        self.num_inconsistencies = 0
        self.num_popular = 0
        self.num_unpopular = 0

    def summarize_stats(self):

        self.num_vertices = len(self.vertices)
        for x in range(0, self.num_vertices):
            if self.vertices[x].is_popular:
                self.num_popular += 1
            if self.vertices[x].is_unpopular:
                self.num_unpopular += 1
        self.calc_chi()

    def add_vertex(self, vertex):

        self.vertices.append(vertex)
        self.vertex_indices.append(vertex.i)
        self.num_vertices += 1
        if vertex.is_popular:
            self.num_popular += 1
        if vertex.is_unpopular:
            self.num_unpopular += 1

    def reset_vertex(self, i):
        self.vertices[i].level = 0
        self.vertices[i].reset_prob_vector(bias=self.vertices[i].num_selections - 1)
        self.vertices[i].is_in_precipitate = False
        self.vertices[i].is_unpopular = False
        self.vertices[i].is_popular = False
        self.vertices[i].is_edge_column = False
        self.vertices[i].show_in_overlay = True

    def remove_vertex(self, vertex_index):
        raise NotImplemented

    def add_edge(self, vertex_a, vertex_b, index):
        self.edges.append(Edge(vertex_a, vertex_b, index))
        self.num_edges += 1

    def remove_edge(self, edge_index):
        raise NotImplemented

    def redraw_edges(self):
        self.edges = []
        self.num_edges = 0
        for i in range(0, self.num_vertices):
            self.vertices[i].partners()
            for j in range(0, self.vertices[i].n()):
                self.add_edge(self.vertices[i], self.vertices[self.vertices[i].partner_indices[j]], self.num_edges)
        self.calc_chi()

    def calc_chi(self):
        self.chi = 0
        for i in range(0, self.num_edges):
            if self.edges[i].is_consistent():
                self.chi += 1
        if self.num_edges == 0:
            self.chi = 0
        else:
            self.chi = self.chi / self.num_edges
        return self.chi

    def reset_vertex_properties(self):
        self.edges = []
        self.chi = 0
        self.num_particle_vertices = 0
        self.num_inconsistencies = 0
        self.num_popular = 0
        self.num_unpopular = 0

        self.reset_all_flags()

        for x in range(0, self.num_vertices):
            if not self.vertices[x].set_by_user:
                self.vertices[x].level = 0
                self.vertices[x].reset_prob_vector(bias=self.vertices[x].num_selections - 1)
                self.vertices[x].is_in_precipitate = False
                self.vertices[x].is_unpopular = False
                self.vertices[x].is_popular = False
                self.vertices[x].is_edge_column = False
                self.vertices[x].show_in_overlay = True

        self.summarize_stats()

    def reset_all_flags(self):
        for x in range(0, self.num_vertices):
            self.vertices[x].flag_1 = False
            self.vertices[x].flag_2 = False
            self.vertices[x].flag_3 = False
            self.vertices[x].flag_4 = False

    def invert_levels(self):
        for x in range(0, self.num_vertices):
            if self.vertices[x].level == 0:
                self.vertices[x].level = 1
            elif self.vertices[x].level == 1:
                self.vertices[x].level = 0

    def find_mesh(self, i, j):
        # return list of indices of shape and number of edges
        return 5, 6

    def map_spatial_neighbours(self):

        for i in range(0, self.num_vertices):

            all_distances = []
            sorted_indices = []

            for j in range(0, self.num_vertices):

                if i == j:
                    all_distances.append(100000)
                else:
                    all_distances.append(self.spatial_distance(i, j))

            all_indices = np.array(all_distances)

            for k in range(0, self.map_size):

                index_of_min = all_indices.argmin()
                sorted_indices.append(index_of_min)
                all_indices[index_of_min] = all_indices.max() + 1

            self.vertices[i].neighbour_indices = sorted_indices

    def find_nearest(self, i, n):

        all_distances = []
        sorted_indices = []
        sorted_distances = []

        for j in range(0, self.num_vertices):

            if i == j:
                all_distances.append(100000)
            else:
                all_distances.append(self.spatial_distance(i, j))

        all_indices = np.array(all_distances)

        for k in range(0, n):
            index_of_min = all_indices.argmin()
            value_of_min = all_indices.min()
            sorted_indices.append(index_of_min)
            sorted_distances.append(value_of_min)
            all_indices[index_of_min] = all_indices.max() + 1

        return sorted_indices, sorted_distances, n

    def spatial_distance(self, i, j):
        delta_x = self.vertices[j].real_coor_x - self.vertices[i].real_coor_x
        delta_y = self.vertices[j].real_coor_y - self.vertices[i].real_coor_y
        arg = delta_x ** 2 + delta_y ** 2
        return np.sqrt(arg)

    def test_reciprocality(self, i, j):

        found = False

        for x in range(0, self.vertices[j].n()):

            if self.vertices[j].neighbour_indices[x] == i:

                found = True

        return found

    def set_level(self, i, level):
        self.vertices[i].level = level


