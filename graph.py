# This file contains the classes of graph objects

import numpy as np


class Vertex:

    def __init__(self, index, x, y, r, peak_gamma, avg_gamma, num_selections=7, level=0, atomic_species='Un', h_index=6,
                 species_strings=None, certainty_threshold=0.8):

        self.i = index
        self.real_coor_x = x
        self.real_coor_y = y
        self.im_coor_x = np.floor(x)
        self.im_coor_y = np.floor(y)
        self.r = r
        self.peak_gamma = peak_gamma
        self.avg_gamma = avg_gamma
        self.num_selections = num_selections
        self.level = level
        self.atomic_species = atomic_species
        self.h_index = h_index

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

        # The prob_vector is ordered to represent the elements in order of their radius:

        # Si
        # Cu
        # Zm
        # Al
        # Ag
        # Mg
        # Un

        if species_strings is None:
            self.species_strings = ['Si', 'Cu', 'Zn', 'Al', 'Ag', 'Mg', 'Un']
        else:
            self.species_strings = species_strings

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

        self.renorm_prob_vector()
        self.define_species()

    def renorm_prob_vector(self):

        vector_sum = np.sum(self.prob_vector)

        if vector_sum == 0.0:
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

        return h_index, is_certain


class Edge:

    def __init__(self, vertex_a, vertex_b):

        self.vertex_a = vertex_a
        self.vertex_b = vertex_b

    def is_consistent(self):
        pass


class AtomicGraph:

    def __init__(self):

        self.vertices = []
        self.vertex_indices = []
        self.edges = []
        self.chi = 0
        self.num_vertices = len(self.vertices)
        self.num_particle_vertices = 0
        self.num_inconsistencies = 0
        self.num_popular = 0
        self.num_unpopular = 0

    def add_vertex(self, vertex):

        self.vertices.append(vertex)
        self.vertex_indices.append(vertex.i)
        self.num_vertices += 1
        if vertex.is_popular:
            self.num_popular += 1
        if vertex.is_unpopular:
            self.num_unpopular += 1

    def remove_vertex(self, vertex_index):
        raise NotImplemented

