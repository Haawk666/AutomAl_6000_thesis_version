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
                 level=0, atomic_species='Un', species_index=6, void=False):

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
        self.void = void
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
        self.redshift_sum = 0
        self.redshift_variance = 0

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
        elif bias in [0, 1, 2, 3, 4, 5, 6]:
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

    def permute_pos_j_pos_k(self, pos_j, pos_k):
        self.district[pos_j], self.district[pos_k] = self.district[pos_k], self.district[pos_j]
        return True

    def shift_pos_j_pos_k(self, pos_j, pos_k):
        if pos_k == len(self.district) - 1 or pos_k == -1:
            new_district = self.district[:pos_j - 1] + self.district[pos_j + 1:pos_k] +\
                           self.district[pos_j]
        else:
            new_district = self.district[:pos_j - 1] + self.district[pos_j + 1:pos_k] + \
                           self.district[pos_j] + self.district[pos_k + 1:-1]
        self.district = new_district
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
        self.num_void_vertices = 0
        self.order = 0
        self.reduced_order = 0
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
                vertex.shift_pos_j_pos_k(vertex.district.index(i), -1)
                new_district = self.get_spatial_district(vertex.i, exclude=[i])
                for new_citizen in new_district:
                    if new_citizen not in vertex.district:
                        vertex.district[-1] = new_citizen
                        break

        # Replace vertex with void vertex
        new_vertex = Vertex(i, 0, 0, self.vertices[i].r, self.vertices[i].peak_gamma, self.vertices[i].avg_gamma, self.scale, void=True)
        new_vertex.is_edge_column = True
        new_vertex.show_in_overlay = False
        new_vertex.is_set_by_user = True
        self.vertices[i] = new_vertex

        # Remap district sets all over the graph
        self.map_districts()
        self.summarize_stats()

    def get_vertex_objects_from_indices(self, vertex_indices):
        vertices = []
        for index in vertex_indices:
            vertices.append(self.vertices[index])
        return vertices

    def get_only_non_void_vertex_indices(self):
        non_void = []
        for vertex in self.vertices:
            if not vertex.void:
                non_void.append(vertex.i)
        return non_void

    def get_alpha_angles(self, i):
        pass

    def get_theta_angles(self, i):
        pass

    def get_redshift_sum(self, i):
        pass

    def get_separation(self, i, j):
        pos_i = self.vertices[i].spatial_pos()
        pos_j = self.vertices[j].spatial_pos()

        separation = np.sqrt((pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2 + (pos_i[2] - pos_j[2]) ** 2)

        return separation

    def get_projected_separation(self, i, j):
        pos_i = self.vertices[i].spatial_pos()
        pos_j = self.vertices[j].spatial_pos()

        projected_separation = np.sqrt((pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2)

        return projected_separation

    def get_adjacency_matrix(self):
        self.summarize_stats()
        M = np.zeros(self.order - 1, self.order - 1)
        for x in range(0, self.order):
            for y in range(0, self.order):
                if x in self.vertices[y].out_neighbourhood:
                    M[y, x] = 1
                else:
                    M[y, x] = 0

        return M

    def get_reduced_adjacency_matrix(self):
        self.summarize_stats()
        M = np.zeros(self.reduced_order - 1, self.reduced_order - 1)
        for x in range(0, self.reduced_order):
            if not self.vertices[x].void:
                for y in range(0, self.reduced_order):
                    if not self.vertices[y].void:
                        if x in self.vertices[y].out_neighbourhood:
                            M[y, x] = 1
                        else:
                            M[y, x] = 0

        return M

    def get_spatial_district(self, i, n=8, exclude=None):
        projected_separations = []
        indices = []
        for vertex in self.vertices:
            if not vertex.i == i and not vertex.void:
                if exclude:
                    if vertex.i not in exclude:
                        projected_separation = self.get_projected_separation(i, vertex.i)
                        projected_separations.append(projected_separation)
                        indices.append(vertex.i)
                else:
                    projected_separation = self.get_projected_separation(i, vertex.i)
                    projected_separations.append(projected_separation)
                    indices.append(vertex.i)

        district = []

        for k in range(0, n):
            min_index = projected_separations.index(min(projected_separations))
            district.append(indices[min_index])
            projected_separations[min_index] = max(projected_separations) + 5

        return district

    def get_column_centered_subgraph(self, i, order=1):
        subgraph = []
        for neighbour in self.vertices[i].neighbours:
            corners, angles, vectors = self.get_mesh_centered_subgraph(i, neighbour)
            mesh = [corners, angles, vectors]
            subgraph.append(mesh)

        return subgraph

    def get_arc_centered_subgraph(self, i, j, order=1):
        mesh_1_corners, mesh_1_angles, mesh_1_vectors = self.get_mesh_centered_subgraph(i, j)
        mesh_2_corners, mesh_2_angles, mesh_2_vectors = self.get_mesh_centered_subgraph(j, i)

        return mesh_1_corners, mesh_2_corners, mesh_1_angles, mesh_2_angles, mesh_1_vectors, mesh_2_vectors

    def get_mesh_centered_subgraph(self, i, j, order=0):

        corners = [i, j]
        counter = 0
        backup_counter = 0
        stop = False

        while not stop:

            angle, next_index = self.angle_sort(i, j)

            if next_index == corners[0] or counter > 14:
                _, nextnext = self.angle_sort(j, next_index)
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

        return corners, angles, vectors

    @staticmethod
    def rebase(corners, next_, j, append=True):
        for k, corner in enumerate(corners):
            if corner == next_:
                del corners[k + 1:]
                if append:
                    corners.append(j)
                break
        return corners, next_, j

    def angle_sort(self, i, j):

        min_angle = 300
        next_index = -1
        p1 = self.vertices[i].im_pos()
        pivot = self.vertices[j].im_pos()

        search_list = self.vertices[j].neighbourhood

        for k in search_list:
            if not k == i:
                p2 = self.vertices[k].im_pos()
                alpha = utils.find_angle_from_points(p1[:1], p2[:1], pivot[:1])
                if alpha < min_angle:
                    min_angle = alpha
                    next_index = k

        logger.debug('Found next: {}'.format(next_index))

        return min_angle, next_index

    def map_district(self, i, search_extended_district=False):
        vertex = self.vertices[i]

        if not vertex.void:
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

    def summarize_stats(self):

        self.map_districts()

        # Calc order
        self.num_void_vertices = 0
        for vertex in self.vertices:
            if vertex.void:
                self.num_void_vertices += 1
        self.order = len(self.vertices)
        self.reduced_order = self.order - self.num_void_vertices

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

















