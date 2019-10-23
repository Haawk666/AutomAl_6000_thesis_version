"""Module that contains the information extracted from HAADF-STEM images in a graph-oriented structure. The main
structure will be a **Graph** object that holds a list of **Vertex** objects as well as a list of **Edge** objects.
There are also some functionality for generating **SubGraph** objects, as well as **Mesh** objects."""

# Program imports:
import utils
# External imports:
import numpy as np
import copy
import logging
import sys
# Instantiate logger:
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Vertex:
    """A Vertex is a base building-block of a graph structure.

        In addition to the initialization-parameters listed here, it also has a wide range of fields that are
        manipulated by various other methods and algorithms. They can also be set manually, see the source code.

        :param index: A unique index which reflects its position in the vertex-list of a **Graph** object.
        :param x: The x-position of the atomic column that the vertex represents in coordinates relative to the HAADF-STEM image.
        :param y: The x-position of the atomic column that the vertex represents in coordinates relative to the HAADF-STEM image.
        :param r: The approximated atomic radii in pixels relative to the original HAADF-STEM image.
        :param peak_gamma: The peak intensity (the brightest pixel) of the atomic column.
        :param avg_gamma: The average pixel intensity over the area defined by the circle centered at (x, y) with a radius of r.
        :param alloy_mat: A copy of the alloy-matrix of the **SuchSoftware** instance that the graph of the vertex is a part of.

        :type index: int
        :type x: float
        :type y: float
        :type r: int
        :type peak_gamma: float
        :type avg_gamma: float
        :type alloy_mat: list(<int>)

        .. code-block:: python
            :caption: Example

            >>> import graph
            >>> my_vertex = graph.Vertex(3, 20.3, 39.0004, 5, 0.9, 0.87, [1, 1, 0, 1, 0, 1, 0])
            >>> print(my_vertex)
            Vertex 3:
                real image position (x, y) = (20.3, 39.0004)
                pixel image position (x, y) = (20, 39)
                spatial relative position in pm (x, y) = (20.3, 39.0004)
                peak gamma = 0.9
                average gamma = 0.87
                species: Mg

    """

    def __init__(self, index, x, y, r, peak_gamma, avg_gamma, alloy_mat, num_selections=7, level=0, atomic_species='Un', h_index=6,
                 species_strings=None, certainty_threshold=0.8, scale=1):

        self.i = index
        self.real_coor_x = x
        self.real_coor_y = y
        self.spatial_coor_x = x * scale
        self.spatial_coor_y = y * scale
        self.im_coor_x = int(np.floor(x))
        self.im_coor_y = int(np.floor(y))
        self.r = r
        self.peak_gamma = peak_gamma
        self.normalized_peak_gamma = peak_gamma
        self.avg_gamma = avg_gamma
        self.normalized_avg_gamma = avg_gamma
        self.num_selections = num_selections
        self.level = level
        self.atomic_species = atomic_species
        self.h_index = h_index
        self.alloy_mat = alloy_mat

        self.confidence = 0.0
        self.level_confidence = 0.0
        self.symmetry_confidence = 0.0
        self.certainty_threshold = certainty_threshold
        self.central_angle_variance = 0.0
        self.is_in_precipitate = False
        self.set_by_user = False
        self.flag_1 = False
        self.flag_2 = False
        self.flag_3 = False
        self.flag_4 = False
        self.flag_5 = False
        self.flag_6 = False
        self.flag_7 = False
        self.flag_8 = False
        self.is_unpopular = False
        self.is_popular = False
        self.is_edge_column = False
        self.show_in_overlay = True
        self.symmetry_vector = [1/3, 1/3, 1/3]
        self.level_vector = [1/2, 1/2]
        self.prob_vector = np.ndarray([self.num_selections], dtype=np.float64)
        self.collapsed_prob_vector = np.zeros([self.num_selections], dtype=int)
        self.collapsed_prob_vector[self.num_selections - 1] = 1

        # Relational sets
        self.neighbour_indices = []

        self.partner_indices = []
        self.anti_partner_indices = []

        self.true_partner_indices = []
        self.unfriendly_partner_indices = []
        self.true_anti_partner_indices = []
        self.friendly_anti_partner_indices = []

        self.anti_friend_indices = []
        self.friend_indices = []

        self.friendly_indices = []
        self.outsider_indices = []

        # The following params are reserved for future use, whilst still maintaining backwards compatibility:
        self.ad_hoc_list_1 = []
        self.ad_hoc_value_1 = 0
        self.ad_hoc_value_2 = 0

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
            self.species_strings = ['Cu', 'Si', 'Zn', 'Al', 'Ag', 'Mg', 'Un']
        else:
            self.species_strings = species_strings

        self.reset_prob_vector(bias=self.num_selections - 1)

    def __str__(self):
        string = 'Vertex {}:\n'.format(self.i)
        string += '    real image position (x, y) = ({}, {})\n'.format(self.real_coor_x, self.real_coor_y)
        string += '    pixel image position (x, y) = ({}, {})\n'.format(self.im_coor_x, self.im_coor_y)
        string += '    spatial relative position in pm (x, y) = ({}, {})\n'.format(self.spatial_coor_x, self.spatial_coor_y)
        string += '    peak gamma = {}\n'.format(self.peak_gamma)
        string += '    average gamma = {}\n'.format(self.avg_gamma)
        string += '    species: {}'.format(self.species())
        return string

    def n(self):
        """Return the number of closest neighbours, or its symmetry in a sense.

        :return: The number of closest neighbours *n*.
        :rtype: int

        """
        n = 3
        if self.h_index == 0 or self.h_index == 1:
            n = 3
        elif self.h_index == 3:
            n = 4
        elif self.h_index == 5:
            n = 5
        return n

    def real_coor(self):
        """Return a tuple of the vertex image coordinates in floats.

        :return: tuple of floats (x, y) that is the vertex' image position.
        :rtype: tuple(<float>, <float>)

        """
        real_coor = (self.real_coor_x, self.real_coor_y)
        return real_coor

    def spatial_coor(self):
        """Return a tuple of the vertex spatial coordinates in floats.

        :return: tuple of floats (x, y) that is the vertex' real spatial relative position.
        :rtype: tuple(<float>, <float>)

        """
        spatial_coor = (self.spatial_coor_x, self.spatial_coor_y)
        return spatial_coor

    def im_coor(self):
        """Return a tuple of the vertex image coordinates in pixel-discrete ints.

        :return: tuple of ints (x, y) that is the vertex' image pixel position.
        :rtype: tuple(<int>, <int>)

        """
        im_coor = (self.im_coor_x, self.im_coor_y)
        return im_coor

    def increase_h_value(self):
        """Increase the :code:`self.h_index` value to the next valid value, if any.

        This method will forcefully try to change the atomic species by incrementing the h_index. If the current species
        is Si, it will set the species to Cu etc:

        ======================= ===============
        current species         New species
        ======================= ===============
        Si                      Cu
        Cu                      Al
        Al                      Mg
        Mg                      No change
        ======================= ===============

        :return: A boolean to indicate whether the species was successfully changed or not.
        :rtype: bool

        """
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
        """Decrease the :code:`self.h_index` value to the next valid value, if any.

        This method will forcefully try to change the atomic species by decrementing the h_index. If the current species
        is Cu, it will set the species to Si etc:

        ======================= ===============
        current species         New species
        ======================= ===============
        Si                      No change
        Cu                      Si
        Al                      Cu
        Mg                      Al
        ======================= ===============

        :return: A boolean to indicate whether the species was successfully changed or not.
        :rtype: bool

        """
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

    def reset_level_vector(self):
        """Reset :code:`self.level_vector`.

        Will reset the *probability vector* :code:`self.level_vector` to a uniform vector: [0.5, 0.5].

        """
        self.level_vector = [1/2, 1/2]

    def reset_symmetry_vector(self, bias=-1):
        """Reset :code:`self.symmetry_vector`.

        Will reset the *probability vector* :code:`self.symmetry_vector` to a uniform vector: [0.33, 0.33, 033]

        :param bias: (Optional, default=-1) Introduce a slight bias for a certain symmetry, bias=-1 equals no bias.
        :type bias: int

        """
        self.symmetry_vector = [1.0, 1.0, 1.0]

        if not bias == -1:
            self.symmetry_vector[bias - 3] = 1.1

        self.renorm_symmetry_vector()

    def reset_prob_vector(self, bias=-1):
        """Reset the atomic species probabillity vector to a uniform distribution.

        :param bias: (Optional, default=-1) Introduce a slight bias for a certain symmetry, bias=-1 equals no bias.
        :type bias: int

        """
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
        """Collapse the probability vector.

        .. code-block:: python
            :caption: Example

            >>> import numpy as np
            >>> my_vertex.prob_vector = np.array([0.2, 0.3, 0.0, 0.1, 0.0, 0.4, 0.0])
            >>> my_vertex.collapse_prob_vector()
            >>> my_vertex.prob_vector
            array([0, 0, 0, 0, 0, 1, 0])

        """
        self.define_species()
        self.prob_vector = self.collapsed_prob_vector

    def multiply_symmetry(self):
        """Temporary function.

        """
        self.reset_prob_vector()
        self.prob_vector[0] *= self.symmetry_vector[0]
        self.prob_vector[1] *= self.symmetry_vector[0]
        self.prob_vector[2] = 0
        self.prob_vector[3] *= self.symmetry_vector[1]
        self.prob_vector[4] = 0
        self.prob_vector[5] *= self.symmetry_vector[2]
        self.prob_vector[6] = 0
        self.renorm_prob_vector()
        self.define_species()

    def renorm_level_vector(self):
        """Re-normalize the level vector to sum to 1.

        """
        self.level_vector = utils.normalize_list(self.level_vector)

    def renorm_symmetry_vector(self):
        """Re-normalize the symmetry vector to sum to 1.

        """
        self.symmetry_vector = utils.normalize_list(self.symmetry_vector)

    def renorm_prob_vector(self):
        """Re-normalize the probability vector to sum to 1.

        """
        for k in range(0, self.num_selections):
            self.prob_vector[k] *= self.alloy_mat[k]
        vector_sum = np.sum(self.prob_vector)
        if vector_sum <= 0.00000001:
            pass
        else:
            correction_factor = 1 / vector_sum
            self.prob_vector = correction_factor * self.prob_vector

    def define_species(self):
        """Determine :code:`self.h_index` by analysing :code:`self.prob_vector`.

        """

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
        """Determine the confidence from the probability vector. The *confidence* is here defined as the difference
        between the highest and next-highest probabilities.

        """

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

    def analyse_symmetry_vector_confidence(self):

        h_value = max(self.symmetry_vector)
        nh_value = 0
        for symmetry in self.symmetry_vector:
            if h_value > symmetry >= nh_value:
                nh_value = symmetry

        confidence = 1 - nh_value / h_value
        is_certain = False
        if confidence >= self.certainty_threshold:
            is_certain = True

        return h_value, is_certain

    def analyse_level_vector_confidence(self):
        self.renorm_level_vector()
        confidence = abs(self.level_vector[0] - self.level_vector[1])
        return confidence

    def set_level_from_vector(self):
        self.level_confidence = self.analyse_level_vector_confidence()
        if self.level_vector[0] > self.level_vector[1]:
            self.level = 0
        elif self.level_vector[0] < self.level_vector[1]:
            self.level = 1
        else:
            self.level = 0
        return self.level

    def force_species(self, h_index):
        """Force the species of the vertex by h_index -value.

        Force the species of the vertex by h_index. The h-index relates to atomic species as:

        ========    ================
        h-index     Atomic species
        ========    ================
        0           Si
        1           Cu
        2           Zn
        3           Al
        4           Ag
        5           Mg
        6           Un
        ========    ================

        :param h_index: Species index
        :type h_index: int

        """
        self.h_index = h_index
        self.atomic_species = self.species_strings[h_index]
        self.confidence = 1.0
        self.set_by_user = True
        self.reset_prob_vector(bias=h_index)
        self.collapse_prob_vector()

    def anti_level(self):
        """Get the opposite level of :code:`self.level`.

        :return: If :code:`self.level` is 0, return 1. If :code:`self.level` is 1, return 0
        :rtype: int

        """
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
        """Update and return a list of indices of the partners of the vertex.

        :return: List of partner indices
        :rtype: list(<int>)

        """
        if not len(self.neighbour_indices) == 0:
            self.partner_indices = self.neighbour_indices[0:self.n()]
            return self.partner_indices

    def anti_partners(self):
        """Return a list of indices, that are neighbours, but not partners to vertex.

        :return: List of anti-partner indices
        :rtype: list(<int>)

        """

        if not len(self.neighbour_indices) == 0:
            self.anti_partner_indices = self.neighbour_indices[self.n():]
            return self.anti_partner_indices

    def partner_query(self, j):
        """Test if index j is among the partners of the vertex.

        :return: True if j is a partner to :code:`self`, False otherwise.
        :rtype: bool

        """
        found = False
        for i in self.partners():
            if i == j:
                found = True
        return found

    def perturb_j_k(self, j, k):
        """Perturb the neighbour positions.

        The neighbour with index *j*, will switch positions with the neighbour with index *k*. This change will also be
        reflected in the partner indices. If neither *j* or *k* is found, the last position of the neighbour_indices
        will be overwritten by *k*. If *j* is found, but not *k*, the last position of the neighbour_indices will be
        overwritten by *k*, and then perturbed with *j*. If *k* is found, but not *j*, nothing will be done. In essence,
        this is used to downgrade the importance of *j*, and upgrade *k*. It is not a symmetric function!

        :param j: Index of neighbour to downgrade
        :type j: int
        :param k: Index of neighbour to upgrade
        :type k: int

        """
        if j == k:
            pass
        else:
            pos_j = -1
            pos_k = -1
            for m, neighbour in enumerate(self.neighbour_indices):
                if neighbour == j:
                    pos_j = m
                if neighbour == k:
                    pos_k = m
            if not pos_j == -1 and not pos_k == -1:
                self.neighbour_indices[pos_j], self.neighbour_indices[pos_k] =\
                    self.neighbour_indices[pos_k], self.neighbour_indices[pos_j]
            elif pos_j == -1 and pos_k == -1:
                self.neighbour_indices[-1] = k
            else:
                if pos_j == -1:
                    logger.warning('Was not able to perturb! index j not present!')
                elif pos_k == -1:
                    self.neighbour_indices[-1] = k
                    self.neighbour_indices[pos_j], self.neighbour_indices[-1] = \
                        self.neighbour_indices[-1], self.neighbour_indices[pos_j]

    def species(self):
        """Get a string representation of the vertex´ atomic species.

        :return: Atomic species.
        :rtype: string

        """
        return self.species_strings[self.h_index]


class Edge:
    """Edge objets where intended to represent connections, but in the end, it was better to just use the internal
    mapping by the vertices. To be DEPRECATED"""

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

        if not self.vertex_b.partner_query(index_i):
            is_consistent = False
            is_reciprocated = False

        if not self.vertex_a.partner_query(index_j):
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
    """An atomic graph is the central concept for the data-structure for AutoAtom.

    An *AtomicGraph* holds a list of references to every vertex, and has mutator-functions to operate on their
    relations.

    :param map_size: (Optional, default=8). The number of possible atomic species to be considered. For now, only 8 specific species are supported.
    :type map_size: int

    .. code-block:: python
        :caption: Example

        >>> import graph
        >>> my_vertex = graph.Vertex(0, 20.3, 39.0004, 5, 0.9, 0.87, [1, 1, 0, 1, 0, 1, 0])
        >>> my_graph = graph.AtomicGraph()
        >>> my_graph.add_vertex(my_vertex)
        >>> print(my_graph)
        Graph summary:----------
            Number of vertices: 1
            Chi: 0

    """

    def __init__(self, map_size=8):

        self.vertices = []
        self.vertex_indices = []
        self.edges = []
        self.particle_boarder = []

        self.meshes = []
        self.mesh_indices = []

        self.map_size = map_size

        # Stats
        self.chi = 0
        self.num_vertices = len(self.vertices)
        self.num_particle_vertices = 0
        self.num_edges = 0
        self.num_inconsistencies = 0
        self.num_popular = 0
        self.num_unpopular = 0
        self.avg_species_confidence = 0.0
        self.avg_symmetry_confidence = 0.0
        self.avg_level_confidence = 0.0
        self.avg_central_variance = 0.0

    def __str__(self):
        self.summarize_stats()
        string = 'Graph summary:----------\n'
        string += '    Number of vertices: {}\n'.format(self.num_vertices)
        string += '    Chi: {}'.format(self.chi)
        return string

    def __len__(self):
        return len(self.vertices)

    def sort_subsets_by_distance(self):
        self.map_all_subsets()
        for vertex in self.vertices:

            true_partner_distances = []
            unfriendly_partner_distances = []
            true_anti_partner_distnaces = []
            friendly_anti_partner_distances = []
            outsider_distances = []

            for tp in vertex.true_partner_indices:
                true_partner_distances.append(self.projected_distance(vertex.i, tp))
            for up in vertex.unfriendly_partner_indices:
                unfriendly_partner_distances.append(self.projected_distance(vertex.i, up))
            for ta in vertex.true_anti_partner_indices:
                true_anti_partner_distnaces.append(self.projected_distance(vertex.i, ta))
            for fa in vertex.friendly_anti_partner_indices:
                friendly_anti_partner_distances.append(self.projected_distance(vertex.i, fa))
            for out in vertex.outsider_indices:
                outsider_distances.append(self.projected_distance(vertex.i, out))

            vertex.true_partner_indices = [y for x, y in sorted(zip(true_partner_distances, vertex.true_partner_indices))]
            vertex.unfriendly_partner_indices = [y for x, y in sorted(zip(unfriendly_partner_distances, vertex.unfriendly_partner_indices))]
            vertex.true_anti_partner_indices = [y for x, y in sorted(zip(true_partner_distances, vertex.true_anti_partner_indices))]
            vertex.friendly_anti_partner_indices = [y for x, y in sorted(zip(friendly_anti_partner_distances, vertex.friendly_anti_partner_indices))]
            vertex.outsider_indices = [y for x, y in sorted(zip(outsider_distances, vertex.outsider_indices))]

            vertex.partners = vertex.true_partner_indices + vertex.unfriendly_partner_indices
            vertex.anti_partner_indices = vertex.true_anti_partner_indices + vertex.friendly_anti_partner_indices

            vertex.neighbour_indices = vertex.partner_indices + vertex.anti_partner_indices

    def map_friends(self):
        logger.info('Mapping friends..')
        for vertex in self.vertices:
            vertex.friendly_indices = []
        for vertex in self.vertices:
            for partner in vertex.partners():
                if vertex.i not in self.vertices[partner].friendly_indices:
                    self.vertices[partner].friendly_indices.append(vertex.i)
        logger.info('friends mapped!')

    def produce_vertex_objects_from_indices(self, indices):
        vertices = []
        for i in indices:
            vertices.append(self.vertices[i])
        return vertices

    def map_all_subsets(self):
        self.map_friends()
        for i in self.vertex_indices:
            self.determine_subsets(i)

    def determine_subsets(self, i):
        partner_indices = self.vertices[i].partners()
        anti_partner_indices = self.vertices[i].anti_partners()

        self.vertices[i].true_partner_indices = []
        self.vertices[i].unfriendly_partner_indices = []
        self.vertices[i].true_anti_partner_indices = []
        self.vertices[i].friendly_anti_partner_indices = []

        for j in partner_indices:
            if i in self.vertices[j].partners():
                self.vertices[i].true_partner_indices.append(j)
            else:
                self.vertices[i].unfriendly_partner_indices.append(j)
        for k in anti_partner_indices:
            if i in self.vertices[k].partners():
                self.vertices[i].friendly_anti_partner_indices.append(k)
            else:
                self.vertices[i].true_anti_partner_indices.append(k)

        self.vertices[i].anti_friend_indices = self.vertices[i].unfriendly_partner_indices + \
                                               self.vertices[i].true_anti_partner_indices
        self.vertices[i].friend_indices = self.vertices[i].true_partner_indices + \
                                          self.vertices[i].friendly_anti_partner_indices

        self.vertices[i].outsider_indices = []
        for friendly in self.vertices[i].friendly_indices:
            if friendly not in self.vertices[i].friend_indices:
                self.vertices[i].outsider_indices.append(friendly)

    def map_meshes(self, i):
        """Automatically generate a connected relational map of all meshes in graph.

        The index of a mesh is temporarily indexed during the mapping by the following algorithm: Take the indices of
        its corners, and circularly pertub them such that the lowest index comes first. After the mapping is complete,
        these indices are replaced by the integers 0 to the number of meshes.

        :param i: Index of starting vertex.
        :type i: int

        """

        logger.info('Mapping meshes..')
        self.meshes = []
        self.mesh_indices = []
        self.map_friends()
        sub_graph_0 = self.get_atomic_configuration(i)
        mesh_0 = sub_graph_0.meshes[0]
        mesh_0.mesh_index = self.determine_temp_index(mesh_0)
        mesh_0.calc_cm()

        self.meshes.append(mesh_0)
        self.mesh_indices.append(mesh_0.mesh_index)

        sys.setrecursionlimit(5000)
        self.walk_mesh_edges(mesh_0)

        new_indices = [i for i in range(0, len(self.mesh_indices))]

        for k, mesh in enumerate(self.meshes):
            for j, neighbour in enumerate(mesh.surrounding_meshes):
                mesh.surrounding_meshes[j] = self.mesh_indices.index(neighbour)
            mesh.mesh_index = self.mesh_indices.index(mesh.mesh_index)

        self.mesh_indices = new_indices
        logger.info('Meshes mapped!')

    def walk_mesh_edges(self, mesh):

        for k, corner in enumerate(vertex.i for vertex in mesh.vertices):
            new_mesh = self.find_mesh(corner, mesh.vertices[k - 1].i, return_mesh=True, use_friends=True)
            has_edge_columns = False
            for vertex in new_mesh.vertices:
                if vertex.is_edge_column:
                    has_edge_columns = True
            if not has_edge_columns:
                tmp_index = self.determine_temp_index(new_mesh)
                mesh.surrounding_meshes.append(tmp_index)
                if tmp_index not in self.mesh_indices:
                    new_mesh.mesh_index = tmp_index
                    new_mesh.calc_cm()
                    self.meshes.append(new_mesh)
                    self.mesh_indices.append(tmp_index)
                    self.walk_mesh_edges(new_mesh)

    @staticmethod
    def determine_temp_index(mesh):
        return utils.make_int_from_list(utils.cyclic_sort(mesh.vertex_indices))

    def calc_avg_species_confidence(self):
        """Calculate the average species confidence of the graph.

        :return: Average species confidence
        :rtype: float

        """
        sum_ = 0
        for vertex in self.vertices:
            vertex.analyse_prob_vector_confidence()
            sum_ += vertex.confidence
        if self.num_vertices == 0:
            result = 0
        else:
            result = sum_ / self.num_vertices
        self.avg_species_confidence = result
        return result

    def calc_avg_symmetry_confidence(self):
        """Calculate average confidence based on symmetry vectors.

        :return: Average symmetry confidence
        :rtype: float

        """
        sum_ = 0
        for vertex in self.vertices:
            vertex.analyse_symmetry_vector_confidence()
            sum_ += vertex.symmetry_confidence
        if self.num_vertices == 0:
            result = 0
        else:
            result = sum_ / self.num_vertices
        self.avg_symmetry_confidence = result
        return result

    def calc_avg_level_confidence(self):
        """Calculate level confidence based on level vectors.

        :return: Average level confidence
        :rtype: float

        """
        sum_ = 0
        for vertex in self.vertices:
            vertex.analyse_level_vector_confidence()
            sum_ += vertex.level_confidence
        if self.num_vertices == 0:
            result = 0
        else:
            result = sum_ / self.num_vertices
        self.avg_level_confidence = result
        return result

    def summarize_stats(self):
        """Summarize some graph stats.

        """

        self.num_vertices = len(self.vertices)
        self.num_popular = 0
        self.num_unpopular = 0
        for vertex in self.vertices:
            if vertex.is_popular:
                self.num_popular += 1
            if vertex.is_unpopular:
                self.num_unpopular += 1
        self.calc_chi()
        self.calc_avg_species_confidence()
        self.calc_avg_symmetry_confidence()
        self.calc_avg_level_confidence()
        self.calc_avg_central_angle_variance()

    def add_vertex(self, vertex):
        """Add a new vertex to the graph.

        """

        self.vertices.append(vertex)
        self.vertex_indices.append(vertex.i)
        self.num_vertices += 1
        if vertex.is_popular:
            self.num_popular += 1
        if vertex.is_unpopular:
            self.num_unpopular += 1
        logger.debug('Added vertex with index {}'.format(vertex.i))

    def reset_vertex(self, i):
        """Reset the properties of vertex *i*.

        :param i: index of vertex to effect.
        :type i: int

        """
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
        """Increase the species index by symmetry.

        """
        changed = self.vertices[i].increase_h_value()
        return changed

    def decrease_h(self, i):
        """Decrease the species index by symmetry.

        """
        changed = self.vertices[i].decrease_h_value()
        return changed

    def add_edge(self, vertex_a, vertex_b, index):
        """Add new edge to the graph.

        """
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

    def weak_remove_edge(self, i, j, aggresive=True):

        for anti_partner in self.vertices[i].anti_partners():

            if self.vertices[anti_partner].partner_query(i):
                self.perturb_j_k(i, j, anti_partner)
                break

        else:

            for anti_partner_2 in self.vertices[i].anti_partners():
                found = False
                for anti_partner_partner in self.vertices[anti_partner_2].partners():

                    if aggresive and not self.vertices[anti_partner_partner].partner_query(anti_partner_2) and\
                            self.vertices[i].level == self.vertices[anti_partner_2].anti_level():

                        self.perturb_j_k(i, j, anti_partner_2)
                        found = True
                        break

                if found:
                    break

            else:

                return False

            return True

        return True

    def strong_remove_edge(self, i, j):

        if self.perturb_j_to_last_partner(i, j):
            if self.decrease_h(i):
                return True
            else:
                return False
        else:
            return True

    def strong_enforce_edge(self, i, j):

        k = self.vertices[j].anti_partners()[0]
        self.vertices[j].perturb_j_k(k, i)
        if self.increase_h(j):
            return True
        else:
            return False

    def perturb_j_k(self, i, j, k):

        if not j == k:

            pos_j = -1
            pos_k = -1

            for m, neighbour in enumerate(self.vertices[i].neighbour_indices):
                if neighbour == j:
                    pos_j = m
                if neighbour == k:
                    pos_k = m

            if pos_k == -1:
                last_index = self.vertices[i].neighbour_indices[self.map_size - 1]
                self.substitute_neighbour(i, last_index, k)
                pos_k = self.map_size - 1

            self.perturb_pos_j_pos_k(i, pos_j, pos_k)

    def perturb_pos_j_pos_k(self, i, pos_j, pos_k):

        self.vertices[i].neighbour_indices[pos_j], self.vertices[i].neighbour_indices[pos_k] =\
            self.vertices[i].neighbour_indices[pos_k], self.vertices[i].neighbour_indices[pos_j]

    def substitute_neighbour(self, i, j, k):

        for pos, neighbour in enumerate(self.vertices[i].neighbour_indices):
            if neighbour == j:
                self.vertices[i].neighbour_indices[pos] = k
                self.vertices[i].partners()
                break
        else:
            return False
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

    def perturb_j_to_first_antipartner(self, i, j):
        if j in self.vertices[i].neighbour_indices:
            pass
        else:
            self.vertices[i].neighbour_indices[-1] = j
        k = self.vertices[i].anti_partners()[0]
        self.perturb_j_k(i, j, k)

    def redraw_edges(self):
        self.edges = []
        self.num_edges = 0
        for vertex in self.vertices:
            if vertex.partners() is not None:
                for partner in vertex.partner_indices:
                    self.add_edge(vertex, self.vertices[partner], self.num_edges)
        self.calc_chi()

    def calc_chi(self):
        self.chi = 0
        if not len(self.edges) <= 0 and self.edges is not None:
            for edge in self.edges:
                if not edge.is_consistent():
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
            vertex.flag_5 = False
            vertex.flag_6 = False
            vertex.flag_7 = False
            vertex.flag_8 = False
        logger.info('All flags reset!')

    def invert_levels(self):
        """Switch the z-position of all vertices.

        """
        for vertex in self.vertices:
            vertex.level = vertex.anti_level()

    def angle_sort(self, i, j, strict=False, use_friends=False):

        logger.debug('Finding next angle from {} -> {}'.format(i, j))

        min_angle = 100
        next_index = -1
        p1 = self.vertices[i].real_coor()
        pivot = self.vertices[j].real_coor()

        search_list = self.vertices[j].partners()
        if use_friends:
            for l in self.vertices[j].friendly_indices:
                if l not in search_list:
                    search_list.append(l)

        for k in search_list:
            if not k == i:
                p2 = self.vertices[k].real_coor()
                alpha = utils.find_angle_from_points(p1, p2, pivot)
                if alpha < min_angle:
                    if not strict:
                        min_angle = alpha
                        next_index = k
                    else:
                        if self.vertices[k].partner_query():
                            min_angle = alpha
                            next_index = k

        logger.debug('Found next: {}'.format(next_index))

        return min_angle, next_index

    def find_mesh(self, i, j, clockwise=True, strict=False, return_mesh=False, use_friends=False):

        logger.debug('Finding mesh from {} -> {}'.format(i, j))

        if not clockwise:
            i, j = j, i

        corners = [i, j]
        counter = 0
        backup_counter = 0
        stop = False

        while not stop:

            angle, next_index = self.angle_sort(i, j, strict=strict, use_friends=use_friends)

            if next_index == corners[0] or counter > 14:
                _, nextnext = self.angle_sort(j, next_index, strict=strict, use_friends=use_friends)
                if not nextnext == corners[1]:
                    corners, i, j = self.rebase(corners, nextnext, next_index, append=False)
                stop = True

            elif next_index in corners:
                corners, i, j = self.rebase(corners, next_index, j)
                counter = len(corners) - 2

            else:
                corners.append(next_index)
                counter += 1
                i, j = j, next_index

            backup_counter += 1

            if backup_counter > 25:

                logger.warning('Emergency stop!')
                stop = True

        angles = []
        vectors = []

        for m, corner in enumerate(corners):

            pivot = self.vertices[corner].real_coor()
            if m == 0:
                p1 = self.vertices[corners[len(corners) - 1]].real_coor()
                p2 = self.vertices[corners[m + 1]].real_coor()
            elif m == len(corners) - 1:
                p1 = self.vertices[corners[m - 1]].real_coor()
                p2 = self.vertices[corners[0]].real_coor()
            else:
                p1 = self.vertices[corners[m - 1]].real_coor()
                p2 = self.vertices[corners[m + 1]].real_coor()

            angle = utils.find_angle_from_points(p1, p2, pivot)
            theta = angle / 2
            vector = (p1[0] - pivot[0], p1[1] - pivot[1])
            length = utils.vector_magnitude(vector)
            vector = (vector[0] / length, vector[1] / length)
            vector = (vector[0] * np.cos(theta) + vector[1] * np.sin(theta),
                      -vector[0] * np.sin(theta) + vector[1] * np.cos(theta))

            angles.append(angle)
            vectors.append(vector)

        if return_mesh:
            mesh = Mesh()
            for k, corner in enumerate(corners):
                mesh.add_vertex(self.vertices[corner])
                mesh.angles.append(angles[k])
                mesh.angle_vectors.append(vectors[k])
            mesh.redraw_edges()
            return mesh
        else:
            return corners, angles, vectors

    @staticmethod
    def rebase(corners, next_, j, append=True):

        logger.debug('Rebasing!')

        for k, corner in enumerate(corners):
            if corner == next_:
                del corners[k + 1:]
                if append:
                    corners.append(j)
                break
        return corners, next_, j

    def get_atomic_configuration(self, i, strict=False, use_friends=False):

        sub_graph = SubGraph(self.map_size)
        sub_graph.add_vertex(self.vertices[i])

        for partner in self.vertices[i].partners():
            sub_graph.add_vertex(self.vertices[partner])
            corners, ang, vec = self.find_mesh(i, partner, strict=strict, use_friends=use_friends)
            mesh = Mesh()
            for k, corner in enumerate(corners):
                mesh.add_vertex(self.vertices[corner])
                mesh.angles.append(ang[k])
                mesh.angle_vectors.append(vec[k])
            mesh.redraw_edges()
            sub_graph.add_mesh(mesh)

            for j in corners:
                if j not in sub_graph.vertex_indices:
                    sub_graph.add_vertex(self.vertices[j])

            closed = False
            while not closed:

                if corners[-1] not in self.vertices[i].partners():
                    corners, ang, vec = self.find_mesh(i, corners[len(corners) - 1], strict=strict, use_friends=use_friends)
                    mesh = Mesh()
                    for k, corner in enumerate(corners):
                        mesh.add_vertex(self.vertices[corner])
                        mesh.angles.append(ang[k])
                        mesh.angle_vectors.append(vec[k])
                    mesh.redraw_edges()
                    sub_graph.add_mesh(mesh)

                    for j in corners:
                        if j not in sub_graph.vertex_indices:
                            sub_graph.add_vertex(self.vertices[j])
                else:
                    closed = True

        sub_graph.finalize_init()

        return sub_graph

    def get_anti_graph(self):
        logger.info('Trying to build anti-graph...')
        return AntiGraph(self).graph

    def get_neighbours(self, i):
        """Get the actual neighbour vertices of vertex *i* as instances.

        The indices of the neighbours of a vertex are easily obtained from their internal list of indices, but sometimes
        it is more convenient to retrieve the actual vertex instances for certain loops etc, which is what this method
        does.

        :param i: Index of vertex to fetch the neighbours of.
        :type i: int

        :return: A list of vertex instances that are neighbours of vertex *i*.
        :rtype: list(<graph.Vertex>)

        """

        neighbours = []
        for index in self.vertices[i].neighbour_indices:
            neighbours.append(self.vertices[index])

        return neighbours

    def get_partners(self, i):
        """Get the actual partner vertices of vertex *i* as instances.

        The indices of the partners of a vertex are easily obtained from their internal list of indices, but sometimes
        it is more convenient to retrieve the actual vertex instances for certain loops etc, which is what this method
        does.

        :param i: Index of vertex to fetch the partners of.
        :type i: int

        :return: A list of vertex instances that are partners of vertex *i*.
        :rtype: list(<graph.Vertex>)

        """

        partners = []
        for index in self.vertices[i].partners():
            partners.append(self.vertices[index])

        return partners

    def find_intersects(self):

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
                    all_distances.append(self.projected_distance(i, j))

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
                all_distances.append(self.projected_distance(i, j))

        all_indices = np.array(all_distances)

        for k in range(0, n):
            index_of_min = all_indices.argmin()
            value_of_min = all_indices.min()
            sorted_indices.append(index_of_min)
            sorted_distances.append(value_of_min)
            all_indices[index_of_min] = all_indices.max() + 1

        return sorted_indices, sorted_distances, n

    def projected_distance(self, i, j):
        delta_x = self.vertices[j].real_coor_x - self.vertices[i].real_coor_x
        delta_y = self.vertices[j].real_coor_y - self.vertices[i].real_coor_y
        arg = delta_x ** 2 + delta_y ** 2
        return np.sqrt(arg)

    def test_reciprocality(self, i, j):
        # Is i in j.partners?
        found = self.vertices[j].partner_query(i)
        return found

    def set_level(self, i, level):
        """Set the z-height level of vertex *i* to *level*.

        """
        self.vertices[i].level = level

    def calc_central_angle_variance(self, i):
        """Calculate the variance of the central *theta*-angles of vertex *i*.

        :param i: index of vertex
        :type i: int

        :return: 3-tuple consisting of list of partners sorted by rotation, list of angles and the variance, or tuple of None, None, None if graph is in a pre-spatial-map state.
        :rtype: tuple(list(<int>), list(<float>), float)

        """

        # Generator?
        partners = self.vertices[i].partners()
        if partners is not None:
            rotation_sorted_partners = []
            rotation_sorted_partners.append(partners[0])
            angles = []

            for j in range(1, len(partners) + 1):
                angle, partner = self.angle_sort(rotation_sorted_partners[j - 1], i, strict=False)
                angles.append(angle)
                rotation_sorted_partners.append(partner)

            variance = utils.variance(angles)
            self.vertices[i].central_angle_variance = variance

            return rotation_sorted_partners, angles, variance
        else:
            return None, None, None

    def calc_avg_central_angle_variance(self):
        """Calculate the average central *theta*-angle variance of the entire graph, exluding edge-columns.

        :return: Average variance of *theta*-angles.
        :rtype: float

        """
        sum_ = 0
        for vertex in self.vertices:
            *_, var = self.calc_central_angle_variance(vertex.i)
            if var is not None and not vertex.is_edge_column:
                sum_ += var
        if self.num_vertices == 0:
            variance = 0
        else:
            variance = sum_ / self.num_vertices
        self.avg_central_variance = variance
        return variance


class AntiGraph:

    def __init__(self, graph):

        self.graph = graph
        self.vertices = copy.deepcopy(graph.vertices)
        self.vertex_indices = copy.deepcopy(graph.vertex_indices)

        self.build()

    def build(self):
        for i, vertex in enumerate(self.graph.vertices):
            if not vertex.is_edge_column:
                sub_graph = self.graph.get_atomic_configuration(vertex.i)
                for mesh in sub_graph.meshes:
                    self.vertices[i].perturb_j_k(mesh.vertex_indices[1], mesh.vertex_indices[2])
                self.vertices[i].partners()
        self.graph = AtomicGraph()
        for vertex in self.vertices:
            self.graph.add_vertex(vertex)
        self.graph.redraw_edges()
        self.graph.summarize_stats()


class SubGraph:

    def __init__(self, map_size=8):

        self.vertices = []
        self.vertex_indices = []
        self.edges = []
        self.meshes = []

        self.map_size = map_size

        self.chi = 0
        self.num_vertices = 0
        self.num_edges = 0
        self.num_meshes = 0
        self.num_inconsistencies = 0
        self.num_popular = 0
        self.num_unpopular = 0

        self.configuration_partner_indices = []
        self.central_mesh_angle_variance = 0

    def finalize_init(self):

        self.configuration_partners()
        self.redraw_edges()
        self.sort_meshes()
        self.summarize_stats()

    def configuration_partners(self):

        for mesh in self.meshes:

            self.configuration_partner_indices.append(mesh.vertex_indices[1])

        return self.configuration_partner_indices

    def calc_central_mesh_angle_variance(self):

        angles = []

        for mesh in self.meshes:
            angles.append(mesh.angles[0])

        var = utils.variance(angles)

        self.central_mesh_angle_variance = var

        return var

    def summarize_stats(self):

        self.num_vertices = len(self.vertices)
        self.num_popular = 0
        self.num_unpopular = 0
        for vertex in self.vertices:
            if vertex.is_popular:
                self.num_popular += 1
            if vertex.is_unpopular:
                self.num_unpopular += 1
        self.num_edges = len(self.edges)
        self.num_meshes = len(self.meshes)

        self.calc_central_mesh_angle_variance()

    def sort_meshes(self):

        new_list = []

        for mesh in self.meshes:

            if mesh.vertex_indices[1] == self.vertices[0].partner_indices[0]:

                new_list.append(mesh)
                break

        closed = False
        if len(self.meshes) == 0:
            closed = True
            print('Error in sub_graph.sort_meshes()')

        while not closed:

            for mesh in self.meshes:

                if mesh.vertex_indices[1] ==\
                        new_list[-1].vertex_indices[-1]:

                    new_list.append(mesh)

                    if new_list[-1].vertex_indices[-1] ==\
                            new_list[0].vertex_indices[1]:

                        closed = True

                    break

        if len(new_list) == len(self.meshes):
            self.meshes = new_list
        else:
            print('Error in sub_graph.sort_meshes()')

    def add_vertex(self, vertex):

        self.vertices.append(vertex)
        self.vertex_indices.append(vertex.i)
        self.num_vertices += 1
        if vertex.is_popular:
            self.num_popular += 1
        if vertex.is_unpopular:
            self.num_unpopular += 1

    def add_mesh(self, mesh):

        self.meshes.append(mesh)
        self.num_meshes += 1

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

    def __init__(self, mesh_index=0):

        self.mesh_index = mesh_index
        self.vertices = []
        self.vertex_indices = []
        self.edges = []
        self.angles = []
        self.angle_vectors = []
        self.surrounding_meshes = []

        self.is_enclosed = True
        self.is_consistent = True
        self.num_corners = 0
        self.num_edges = 0
        self.cm = (0, 0)

    def __str__(self):

        string = ''

        for k, index in enumerate(self.vertex_indices):

            if self.vertices[utils.circularize_next_index(k + 1, len(self.vertices) - 1)].partner_query(k):
                end_left = '<'
            else:
                end_left = ''

            if self.vertices[k].partner_query(utils.circularize_next_index(k + 1, len(self.vertices) - 1)):
                end_right = '>'
            else:
                end_right = ''

            string += '{} {}-{} '.format(index, end_left, end_right)

        return string

    def __eq__(self, other):
        return utils.is_circularly_identical(self.vertex_indices, other.vertex_indices)

    def calc_cm(self):
        x_fit = 0
        y_fit = 0
        mass = len(self.vertices)
        for corner in self.vertices:
            x_fit += corner.real_coor_x
            y_fit += corner.real_coor_y
        x_fit = x_fit / mass
        y_fit = y_fit / mass
        self.cm = (x_fit, y_fit)

    def test_consistency(self):

        self.is_consistent = True
        for edge in self.edges:
            if not edge.is_consistent():
                self.is_consistent = False
        if not self.num_corners == 4:
            self.is_consistent = False
        return self.is_consistent

    def test_sidedness(self):
        if not self.num_corners == 4:
            return False
        else:
            return True

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

    @staticmethod
    def neighbour_test(mesh_1, mesh_2):

        found = 0
        edge = []

        for corner in mesh_1.vertex_indices:

            if corner in mesh_2.vertex_indices:

                found += 1
                edge.append(corner)

            if found == 2:

                break

        else:

            return False

        assure_edge = True

        for k, corner in enumerate(mesh_1.vertex_indices):
            k_max = len(mesh_1.vertex_indices) - 1

            if corner == edge[0]:

                if mesh_1.vertex_indices[utils.circularize_next_index(k - 1, k_max)] == edge[1] or\
                        mesh_1.vertex_indices[utils.circularize_next_index(k + 1, k_max) == edge[1]]:
                    pass
                else:
                    assure_edge = False

            elif corner == edge[1]:

                if mesh_1.vertex_indices[utils.circularize_next_index(k - 1, k_max)] == edge[0] or\
                        mesh_1.vertex_indices[utils.circularize_next_index(k + 1, k_max) == edge[0]]:
                    pass
                else:
                    assure_edge = False

        for k, corner in enumerate(mesh_2.vertex_indices):
            k_max = len(mesh_2.vertex_indices) - 1

            if corner == edge[0]:

                if mesh_2.vertex_indices[utils.circularize_next_index(k - 1, k_max)] == edge[1] or \
                        mesh_2.vertex_indices[utils.circularize_next_index(k + 1, k_max) == edge[1]]:
                    pass
                else:
                    assure_edge = False

            elif corner == edge[1]:

                if mesh_2.vertex_indices[utils.circularize_next_index(k - 1, k_max)] == edge[0] or \
                        mesh_2.vertex_indices[utils.circularize_next_index(k + 1, k_max) == edge[0]]:
                    pass
                else:
                    assure_edge = False

        if assure_edge:
            return True, edge
        else:
            return False, edge

