# This file contains the classes of graph objects

import numpy as np


class Vertex:

    def __init__(self, index, x, y, r, peak_gamma, avg_gamma, selections=7, level=0, atomic_species='Un', h_index=6):

        self.i = index
        self.x = x
        self.y = y
        self.r = r
        self.peak_gamma = peak_gamma
        self.avg_gamma = avg_gamma
        self.level = level
        self.atomic_species = atomic_species
        self.h_index = h_index

        self.confidence = 0.0
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
        self.prob_vector = np.ndarray([selections], dtype=np.float64)
        self.neighbour_indices = []

        # The prob_vector is ordered to represent the elements in order of their radius:

        # Si
        # Cu
        # Zm
        # Al
        # Ag
        # Mg
        # Un

    def n(self):

        n = 3

        if self.h_index == 0 or self.h_index == 1:
            n = 3
        elif self.h_index == 3:
            n = 4
        elif self.h_index == 5:
            n = 5

        return n


class Edge:

    def __init__(self, vertex_a, vertex_b):

        self.vertex_a = vertex_a
        self.vertex_b = vertex_b

    def is_consistent(self):
        pass


class AtomicGraph:

    def __init__(self):

        self.vertices = []
        self.edges = []
        self.chi = 0
        self.num_vertices = len(self.vertices)
        self.num_particle_vertices = 0
        self.num_inconsistencies = 0
        self.num_popular = 0
        self.num_unpopular = 0

