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

    def real_coor(self):

        real_coor = (self.real_coor_x, self.real_coor_y)

        return real_coor

    def im_coor(self):

        im_coor = (self.im_coor_x, self.im_coor_y)

        return im_coor

    def increase_h_value(self):

        changed = False
        h = self.h_index

        if h == 0 or h == 1:

            self.reset_prob_vector(bias=3)
            changed = True

        elif h == 3:

            self.reset_prob_vector(bias=5)
            changed = True

        return changed

    def decrease_h_value(self):

        changed = False
        h = self.h_index

        if h == 5:

            self.reset_prob_vector(bias=3)
            changed = True

        elif h == 3:

            self.reset_prob_vector()
            self.prob_vector[0] += 0.1
            self.prob_vector[1] += 0.1
            self.renorm_prob_vector()
            self.define_species()
            changed = True

        return changed

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
        else:
            anti_level = None
        return anti_level

    def partners(self):
        if not len(self.neighbour_indices) == 0:
            self.partner_indices = self.neighbour_indices[0:self.n()]
            return self.partner_indices

    def partner_query(self, j):
        found = False
        for i in self.partners():
            if i == j:
                found = True
        return found

    def print(self):
        print('\nVertex properties:\n----------')
        print('Index: {}\nImage pos: ({}, {})\nReal pos: ({}, {})'.format(self.i, self.im_coor_x, self.im_coor_y,
                                                                          self.real_coor_x, self.real_coor_y))
        print('Atomic Species: {}'.format(self.atomic_species))
        print('Probability vector: {}'.format(self.prob_vector))


class Edge:

    def __init__(self, vertex_a, vertex_b, index):

        # Initialize an edge with direction from vertex a -> vertex b.

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

        for ind in self.vertex_b.partners():
            if ind == index_i:
                found_a = True

        if not found_a:
            is_consistent = False
            is_reciprocated = False

        found_b = False

        for ind in self.vertex_a.partners():
            if ind == index_j:
                found_b = True

        if not found_b:
            print('Unexpected error in graph.Edge.is_consistent()')
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
        self.num_popular = 0
        self.num_unpopular = 0
        for vertex in self.vertices:
            if vertex.is_popular:
                self.num_popular += 1
            if vertex.is_unpopular:
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

    def increase_h(self, i):
        changed = self.vertices[i].increase_h_value()
        return changed

    def decrease_h(self, i):
        changed = self.vertices[i].decrease_h_value()
        return changed

    def add_edge(self, vertex_a, vertex_b, index):
        self.edges.append(Edge(vertex_a, vertex_b, index))
        self.num_edges += 1

    def remove_edge(self, i, j):
        if not self.weak_remove_edge(i, j):
            if not self.strong_remove_edge(i, j):
                return False
            else:
                return True
        else:
            return True

    def weak_remove_edge(self, i, j):
        raise NotImplemented

    def strong_remove_edge(self, i, j):

        if self.perturb_j_to_last_partner(i, j):
            if self.decrease_h(i):
                return True
            else:
                return False
        else:
            return True

    def perturb_j_to_last_partner(self, i, j):

        for pos, k in enumerate(self.vertices[i].partners()):
            if k == j:
                pos_n = self.vertices[i].n() - 1
                if not pos == pos_n:
                    self.vertices[i].neighbour_indices[pos], self.vertices[i].neighbour_indices[pos_n] =\
                        self.vertices[i].neighbour_indices[pos_n], self.vertices[i].neighbour_indices[pos]
                break
        else:
            return False
        return True

    def redraw_edges(self):
        self.edges = []
        self.num_edges = 0
        for vertex in self.vertices:
            for partner in vertex.partners():
                self.add_edge(vertex, self.vertices[partner], self.num_edges)
        self.calc_chi()

    def calc_chi(self):
        self.chi = 0
        if not len(self.edges) <= 0 and self.edges is not None:
            for edge in self.edges:
                if edge.is_consistent():
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
        for vertex in self.vertices:
            vertex.flag_1 = False
            vertex.flag_2 = False
            vertex.flag_3 = False
            vertex.flag_4 = False

    def invert_levels(self):
        for vertex in self.vertices:
            vertex.level = vertex.anti_level()

    def angle_sort(self, i, j):

        min_angle = 100
        next_index = -1
        p1 = (self.vertices[i].real_coor_x, self.vertices[i].real_coor_y)
        pivot = (self.vertices[j].real_coor_x, self.vertices[j].real_coor_y)

        for k in self.vertices[j].partners():
            if not k == i:
                p2 = (self.vertices[k].real_coor_x, self.vertices[k].real_coor_y)
                alpha = utils.find_angle_from_points(p1, p2, pivot)
                if alpha < min_angle:
                    min_angle = alpha
                    next_index = k

        return min_angle, next_index

    def find_mesh(self, i, j, clockwise=True):

        if not clockwise:
            i, j = j, i

        # Check that j is partner to i
        if not self.vertices[i].partner_query(j):
            raise NotImplementedError

        corners = [i, j]
        angles = [0]
        vectors = [0]

        counter = 0
        stop = False

        while not stop:

            angle, next_index = self.angle_sort(i, j)
            angles.append(angle)

            p1 = (self.vertices[i].real_coor_x, self.vertices[i].real_coor_y)
            pivot = (self.vertices[j].real_coor_x, self.vertices[j].real_coor_y)

            theta = 0.5 * angle
            vector = (p1[0] - pivot[0], p1[1] - pivot[1])
            length = utils.vector_magnitude(vector)
            vector = (vector[0] / length, vector[1] / length)
            vector = (vector[0] * np.cos(theta) - vector[1] * np.sin(theta),
                      vector[0] * np.sin(theta) + vector[1] * np.cos(theta))
            vectors.append(vector)

            if next_index == corners[0] or counter > 6:
                stop = True
                p1 = (self.vertices[j].real_coor_x, self.vertices[j].real_coor_y)
                pivot = (self.vertices[corners[0]].real_coor_x, self.vertices[corners[0]].real_coor_y)
                p2 = (self.vertices[corners[1]].real_coor_x, self.vertices[corners[1]].real_coor_y)
                angles[0] = utils.find_angle_from_points(p1, p2, pivot)

                theta = 0.5 * angles[0]
                vector = (p1[0] - pivot[0], p1[1] - pivot[1])
                length = utils.vector_magnitude(vector)
                vector = (vector[0] / length, vector[1] / length)
                vector = (vector[0] * np.cos(theta) - vector[1] * np.sin(theta),
                          vector[0] * np.sin(theta) + vector[1] * np.cos(theta))
                vectors[0] = vector
            else:
                corners.append(next_index)
                counter += 1
                i, j = j, next_index

        return corners, angles, vectors

    def get_atomic_configuration(self, i):

        sub_graph = SubGraph(self.map_size)
        sub_graph.add_vertex(self.vertices[i])
        meshes = []

        for partner in self.vertices[i].partners():

            sub_graph.add_vertex(self.vertices[partner])
            corners, ang, vec = self.find_mesh(i, partner)
            mesh = Mesh()
            for k, corner in enumerate(corners):
                mesh.add_vertex(self.vertices[corner])
                mesh.angles.append(ang[k])
                mesh.angle_vectors.append(vec[k])
            mesh.redraw_edges()
            meshes.append(mesh)

            for j in corners:

                if j not in sub_graph.vertex_indices:

                    sub_graph.add_vertex(self.vertices[j])

        sub_graph.redraw_edges()
        sub_graph.summarize_stats()

        return sub_graph, meshes

    def find_intersects(self):

        # Extend?

        intersecting_segments = []

        for a in self.vertices:
            for b in [self.vertices[index] for index in a.partners()]:
                if not a.is_edge_column and not b.is_edge_column:
                    for c in [self.vertices[index] for index in a.partners()]:
                        if not c.i == b.i:
                            for d in [self.vertices[index] for index in c.partners()]:
                                intersects = utils.closed_segment_intersect(a.real_coor(), b.real_coor(),
                                                                            c.real_coor(), d.real_coor())
                                if intersects and (a.i, b.i, c.i, d.i) not in intersecting_segments and\
                                        (c.i, d.i, a.i, b.i) not in intersecting_segments:
                                    intersecting_segments.append((a.i, b.i, c.i, d.i))
                    for c in [self.vertices[index] for index in a.partners()]:
                        for d in [self.vertices[index] for index in c.partners()]:
                            for e in [self.vertices[index] for index in d.partners()]:
                                intersects = utils.closed_segment_intersect(a.real_coor(), b.real_coor(),
                                                                            d.real_coor(), e.real_coor())
                                if intersects and (a.i, b.i, d.i, e.i) not in intersecting_segments and\
                                        (d.i, e.i, a.i, b.i) not in intersecting_segments:
                                    intersecting_segments.append((a.i, b.i, d.i, e.i))

        return intersecting_segments

    def map_spatial_neighbours(self):

        # This function is unefficient and stupid. Use column_centre_mat of the core module for this task. This function
        # is only a back-up

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

        # This function is unefficient and stupid. Use column_centre_mat of the core module for this task. This function
        # is only a back-up

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
        # Is i in j.partners?
        found = self.vertices[j].partner_query(i)
        return found

    def set_level(self, i, level):
        self.vertices[i].level = level


class SubGraph:

    def __init__(self, map_size=8):

        self.vertices = []
        self.vertex_indices = []
        self.edges = []

        self.map_size = map_size

        self.chi = 0
        self.num_vertices = len(self.vertices)
        self.num_edges = 0
        self.num_inconsistencies = 0
        self.num_popular = 0
        self.num_unpopular = 0

    def summarize_stats(self):

        self.num_vertices = len(self.vertices)
        self.num_popular = 0
        self.num_unpopular = 0
        for vertex in self.vertices:
            if vertex.is_popular:
                self.num_popular += 1
            if vertex.is_unpopular:
                self.num_unpopular += 1

    def add_vertex(self, vertex):

        self.vertices.append(vertex)
        self.vertex_indices.append(vertex.i)
        self.num_vertices += 1
        if vertex.is_popular:
            self.num_popular += 1
        if vertex.is_unpopular:
            self.num_unpopular += 1

    def get_ind_from_mother(self, i):

        for index, mother_index in enumerate(self.vertex_indices):
            if mother_index == i:
                sub_index = index
                break
        else:
            sub_index = -1
        return sub_index

    def reset_vertex(self, i):
        i = self.get_ind_from_mother(i)
        if not i == -1:
            self.vertices[i].level = 0
            self.vertices[i].reset_prob_vector(bias=self.vertices[i].num_selections - 1)
            self.vertices[i].is_in_precipitate = False
            self.vertices[i].is_unpopular = False
            self.vertices[i].is_popular = False
            self.vertices[i].is_edge_column = False
            self.vertices[i].show_in_overlay = True

    def remove_vertex(self, vertex_index):
        raise NotImplemented

    def increase_h(self, i):
        i = self.get_ind_from_mother(i)
        if not i == -1:
            changed = self.vertices[i].increase_h_value()
        else:
            changed = False
        return changed

    def decrease_h(self, i):
        i = self.get_ind_from_mother(i)
        if not i == -1:
            changed = self.vertices[i].decrease_h_value()
        else:
            changed = False
        return changed

    def add_edge(self, vertex_a, vertex_b, index):
        self.edges.append(Edge(vertex_a, vertex_b, index))
        self.num_edges += 1

    def remove_edge(self, edge_index):
        raise NotImplemented

    def redraw_edges(self):
        self.edges = []
        self.num_edges = 0
        for vertex in self.vertices:
            for partner in vertex.partners():
                if partner in self.vertex_indices:
                    self.add_edge(vertex, self.vertices[self.get_ind_from_mother(partner)], self.num_edges)


class Mesh:

    def __init__(self):

        self.vertices = []
        self.vertex_indices = []
        self.edges = []
        self.angles = []
        self.angle_vectors = []

        self.is_enclosed = True
        self.is_consistent = True
        self.num_corners = 0
        self.num_edges = 0

    def test_consistency(self):

        self.is_consistent = True
        for edge in self.edges:
            if not edge.is_consistent():
                self.is_consistent = False
        if not self.num_corners == 4:
            self.is_consistent = False
        return self.is_consistent

    def add_vertex(self, vertex):

        self.vertices.append(vertex)
        self.vertex_indices.append(vertex.i)
        self.num_corners += 1

    def redraw_edges(self):

        self.edges = []
        self.num_edges = 0
        for vertex in self.vertices:
            for partner in vertex.partners():
                if partner in self.vertex_indices:
                    self.add_edge(vertex, self.vertices[self.get_ind_from_mother(partner)], self.num_edges)

    def add_edge(self, vertex_a, vertex_b, index):
        self.edges.append(Edge(vertex_a, vertex_b, index))
        self.num_edges += 1

    def get_ind_from_mother(self, i):

        for index, mother_index in enumerate(self.vertex_indices):
            if mother_index == i:
                sub_index = index
                break
        else:
            sub_index = -1
        return sub_index






