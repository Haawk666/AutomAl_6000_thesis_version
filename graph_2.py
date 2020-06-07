

# Internal imports:
import utils
import statistics
# External imports:
import numpy as np
import copy
import sys
import time
import logging
# Instantiate logger:
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Vertex:

    al_lattice_const = 404.95

    def __init__(self, index, im_coor_x, im_coor_y, r, scale, zeta=0, advanced_species='Un_1', n=3, atomic_species='Un', void=False, parent_graph=None):

        # parent
        self.parent_graph = parent_graph

        # Index
        self.i = index

        # Some properties
        self.r = r
        self.zeta = zeta
        self.peak_gamma = 0
        self.avg_gamma = 0
        self.advanced_species = advanced_species
        self.atomic_species = atomic_species

        # Position
        self.scale = scale
        self.zeta = zeta
        self.im_coor_x = im_coor_x
        self.im_coor_y = im_coor_y
        self.im_coor_z = zeta
        self.spatial_coor_x = im_coor_x * scale
        self.spatial_coor_y = im_coor_y * scale
        self.spatial_coor_z = zeta * 0.5 * self.al_lattice_const

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

        # model-analysis
        self.probability_vector = {}
        self.alpha_probability_vector = {}
        self.composite_probability_vector = {}
        self.weighted_probability_vector = {}

        self.confidence = 0
        self.alpha_confidence = 0
        self.composite_confidence = 0
        self.weighted_confidence = 0

        self.advanced_probability_vector = {}
        self.advanced_alpha_probability_vector = {}
        self.advanced_composite_probability_vector = {}
        self.advanced_weighted_probability_vector = {}

        # Model variables
        self.alpha_angles = []
        self.alpha_max = 0
        self.alpha_min = 0
        self.theta_angles = []
        self.theta_max = 0
        self.theta_min = 0
        self.theta_angle_variance = 0
        self.theta_angle_mean = 0
        self.normalized_peak_gamma = 0
        self.normalized_avg_gamma = 0
        self.redshift = 0
        self.redshift_variance = 0
        self.avg_redshift = 0
        self.avg_central_separation = 0

        # Local graph mapping
        self.district = []
        self.out_neighbourhood = set()
        self.in_neighbourhood = set()
        self.neighbourhood = set()
        self.anti_neighbourhood = set()
        self.partners = set()
        self.semi_partners = set()
        self.out_semi_partners = set()
        self.in_semi_partners = set()

        # Graph parameters
        self.n = n
        self.in_degree = 0
        self.out_degree = 0
        self.degree = 0

        self.reset_probability_vector_from_atomic_species()

    def __str__(self):
        return self.report()

    def report(self):
        im_pos = self.im_pos()
        spatial_pos = self.spatial_pos()
        string = 'Vertex {}:\n'.format(self.i)
        string += '    General:\n'
        string += '        Image position (x, y) = ({:.3f}, {:.3f})\n'.format(im_pos[0], im_pos[1])
        string += '        Pixel position (x, y) = ({:.0f}, {:.0f})\n'.format(np.floor(im_pos[0]), np.floor(im_pos[1]))
        string += '        Spatial relative position in pm (x, y, z) = ({:.3f}, {:.3f}, {:.3f})\n'.format(spatial_pos[0], spatial_pos[1], spatial_pos[2])
        string += '        Peak gamma = {:.4f}\n'.format(self.peak_gamma)
        string += '        Average gamma = {:.4f}\n'.format(self.avg_gamma)
        string += '        Atomic species: {}\n'.format(self.atomic_species)
        string += '        Advanced species: {}\n'.format(self.advanced_species)
        string += '        Symmetry: {}\n'.format(self.n)
        string += '    Analysis:\n'
        string += '        Probability vector: {\n'
        for key, prob in self.probability_vector.items():
            string += '            {}: {:.3f}\n'.format(key, prob)
        string += '        }\n'
        string += '            Prediction: {}\n'.format(self.atomic_species)
        string += '            Confidence: {}\n'.format(self.confidence)
        string += '    Graph parameters:\n'
        string += '        In-degree: {}\n'.format(self.in_degree)
        string += '        Out-degree: {}\n'.format(self.out_degree)
        string += '        Degree: {}\n'.format(self.degree)
        string += '        Alpha angles: ['
        for alpha in self.alpha_angles:
            string += ' {:.3f}'.format(alpha)
        string += ' ]\n'
        string += '            Max: {:.3f}\n'.format(self.alpha_max)
        string += '            Min: {:.3f}\n'.format(self.alpha_min)
        string += '        Theta angles: ['
        for theta in self.theta_angles:
            string += ' {:.3f}'.format(theta)
        string += ' ]\n'
        string += '            Max: {:.3f}\n'.format(self.theta_max)
        string += '            Min: {:.3f}\n'.format(self.theta_min)
        string += '            Variance: {:.3f}\n'.format(self.theta_angle_variance)
        string += '            Mean: {:.3f}\n'.format(self.theta_angle_mean)
        string += '        Normalized peak gamma: {:.3f}\n'.format(self.normalized_peak_gamma)
        string += '        Normalized average gamma: {:.3f}\n'.format(self.normalized_avg_gamma)
        string += '        Redshift: {:.3f}\n'.format(self.redshift)
        string += '        Average central separation: {}\n'.format(self.avg_central_separation)
        string += '    Local graph mapping:\n'
        string += '        District: {}\n'.format(self.district)
        string += '        In-neighbourhood: {}\n'.format(self.in_neighbourhood)
        string += '        Out-neighbourhood: {}\n'.format(self.out_neighbourhood)
        string += '        Neighbourhood: {}\n'.format(self.neighbourhood)
        string += '        Anti-Neighbourhood: {}\n'.format(self.anti_neighbourhood)
        string += '        Partners: {}\n'.format(self.partners)
        string += '        Semi-partners: {}\n'.format(self.semi_partners)
        string += '        Out-semi-partners: {}\n'.format(self.out_semi_partners)
        string += '        In-semi-partners: {}\n'.format(self.in_semi_partners)
        string += '    Settings:\n'
        string += '        Is in precipitate: {}\n'.format(str(self.is_in_precipitate))
        string += '        Is edge column: {}\n'.format(str(self.is_edge_column))
        string += '        Is set by user: {}\n'.format(str(self.is_set_by_user))
        string += '        Show in overlay: {}\n'.format(str(self.show_in_overlay))
        string += '        Is void: {}\n'.format(str(self.void))
        string += '        Flag 1: {}\n'.format(str(self.flag_1))
        string += '        Flag 2: {}\n'.format(str(self.flag_2))
        string += '        Flag 3: {}\n'.format(str(self.flag_3))
        string += '        Flag 4: {}\n'.format(str(self.flag_4))
        string += '        Flag 5: {}\n'.format(str(self.flag_5))
        string += '        Flag 6: {}\n'.format(str(self.flag_6))
        string += '        Flag 7: {}\n'.format(str(self.flag_7))
        string += '        Flag 8: {}\n'.format(str(self.flag_8))
        string += '        Flag 9: {}\n'.format(str(self.flag_9))
        return string

    def im_pos(self):
        return self.im_coor_x, self.im_coor_y, self.im_coor_z

    def spatial_pos(self):
        return self.spatial_coor_x, self.spatial_coor_y, self.spatial_coor_z

    def set_probability_from_advanced(self):
        self.probability_vector = {}
        for advanced_species, advanced_prob in self.advanced_probability_vector.items():
            atomic_species = self.parent_graph.species_dict[advanced_species]['atomic_species']
            if atomic_species in self.probability_vector:
                self.probability_vector[atomic_species] += advanced_prob
            else:
                self.probability_vector[atomic_species] = advanced_prob
        self.probability_vector = utils.normalize_dict(self.probability_vector)

    def set_advanced_from_probability(self):
        pass

    def reset_probability_vector(self, bias='Un_1'):
        self.advanced_probability_vector = {}
        self.probability_vector = {}
        if self.parent_graph is not None:
            for category, value in self.parent_graph.species_dict.items():
                if category == bias:
                    self.advanced_probability_vector[category] = 1.1
                    if value['atomic_species'] in self.probability_vector:
                        self.probability_vector[value['atomic_species']] += 1.1
                    else:
                        self.probability_vector[value['atomic_species']] = 1.1
                else:
                    self.advanced_probability_vector[category] = 1.0
                    if value['atomic_species'] in self.probability_vector:
                        self.probability_vector[value['atomic_species']] += 1.0
                    else:
                        self.probability_vector[value['atomic_species']] = 1.0
            self.advanced_probability_vector = utils.normalize_dict(self.advanced_probability_vector)
            self.probability_vector = utils.normalize_dict(self.probability_vector)
            self.atomic_species = self.parent_graph.species_dict[bias]['atomic_species']
            self.advanced_species = bias

    def determine_species_from_probability_vector(self):
        self.advanced_species = max(self.advanced_probability_vector, key=self.advanced_probability_vector.get)
        self.atomic_species = self.parent_graph.species_dict[self.advanced_species]['atomic_species']
        self.set_probability_from_advanced()
        self.n = self.parent_graph.species_dict[self.advanced_species]['symmetry']

    def reset_probability_vector_from_atomic_species(self):
        self.reset_probability_vector(bias=self.advanced_species)

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

    def partner_query(self, j):
        if j in self.district[:self.n]:
            return True
        else:
            return False

    def anti_zeta(self):
        if self.zeta == 0:
            return 1
        else:
            return 0

    def decrement_n(self):
        for species, item in self.parent_graph.species_dict.items():
            if item['symmetry'] == self.n - 1:
                self.reset_probability_vector(bias=species)
                self.determine_species_from_probability_vector()
                break
        else:
            return False
        return True

    def increment_n(self):
        for species, item in self.parent_graph.species_dict.items():
            if item['symmetry'] == self.n + 1:
                self.reset_probability_vector(bias=species)
                self.determine_species_from_probability_vector()
                break
        else:
            return False
        return True


class Arc:

    # Standard AutomAl 6000 class header:
    al_lattice_const = 404.95

    def __init__(self, j, vertex_a, vertex_b, parent_graph=None):

        self.j = j
        self.vertex_a = vertex_a
        self.vertex_b = vertex_b

        self.parent = parent_graph

        if vertex_a.i in vertex_b.out_neighbourhood:
            self.dual_arc = True
        else:
            self.dual_arc = False

        if vertex_a.zeta == vertex_b.zeta:
            self.co_planar = True
        else:
            self.co_planar = False

        self.im_separation = 0
        self.im_projected_separation = 0
        self.spatial_separation = 0
        self.spatial_projected_separation = 0
        self.hard_sphere_separation = 0
        self.redshift = 0

        self.calc_properties()

    def calc_properties(self):
        pos_i = self.vertex_a.im_pos()
        pos_j = self.vertex_b.im_pos()
        self.im_separation = np.sqrt((pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2 + (pos_i[2] - pos_j[2]) ** 2)
        self.im_projected_separation = np.sqrt((pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2)
        pos_i = self.vertex_a.spatial_pos()
        pos_j = self.vertex_b.spatial_pos()
        self.spatial_separation = np.sqrt((pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2 + (pos_i[2] - pos_j[2]) ** 2)
        self.spatial_projected_separation = np.sqrt((pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2)
        if self.parent:
            radii_1 = self.parent.species_dict[self.vertex_a.advanced_species]['atomic_radii']
            radii_2 = self.parent.species_dict[self.vertex_b.advanced_species]['atomic_radii']
        else:
            radii_1 = 100
            radii_2 = 100
        self.hard_sphere_separation = radii_1 + radii_2
        self.redshift = self.hard_sphere_separation - self.spatial_separation


class AtomicGraph:

    al_lattice_const = 404.95

    def __init__(self, scale, active_model=None, district_size=8, species_dict=None):

        # Categorization:
        if species_dict is None:
            self.species_dict = {
                'Si_1': {'symmetry': 3, 'atomic_species': 'Si', 'atomic_radii': 117.50, 'color': (255, 0, 0), 'species_color': (255, 0, 0)},
                'Si_2': {'symmetry': 3, 'atomic_species': 'Si', 'atomic_radii': 117.50, 'color': (235, 20, 20), 'species_color': (255, 0, 0)},
                'Cu_1': {'symmetry': 3, 'atomic_species': 'Cu', 'atomic_radii': 127.81, 'color': (255, 255, 0), 'species_color': (255, 255, 0)},
                'Al_1': {'symmetry': 4, 'atomic_species': 'Al', 'atomic_radii': 143.00, 'color': (0, 255, 0), 'species_color': (0, 255, 0)},
                'Al_2': {'symmetry': 4, 'atomic_species': 'Al', 'atomic_radii': 143.00, 'color': (20, 235, 20), 'species_color': (0, 255, 0)},
                'Mg_1': {'symmetry': 5, 'atomic_species': 'Mg', 'atomic_radii': 160.00, 'color': (138, 43, 226), 'species_color': (138, 43, 226)},
                'Mg_2': {'symmetry': 5, 'atomic_species': 'Mg', 'atomic_radii': 160.00, 'color': (118, 63, 206), 'species_color': (138, 43, 226)},
                'Un_1': {'symmetry': 3, 'atomic_species': 'Un', 'atomic_radii': 200.00, 'color': (0, 0, 255), 'species_color': (0, 0, 255)}
            }
        else:
            self.species_dict = species_dict

        # Contents:
        self.vertices = []
        self.arcs = []
        self.arc_indices = []
        self.anti_arcs = []
        self.anti_arc_indices = []
        self.meshes = []
        self.mesh_indices = []

        self.particle_boarder_indices = []
        self.structures = []

        self.adjacency_matrix = None
        self.separation_matrix = None

        self.scale = scale
        self.district_size = district_size

        if active_model is None:
            self.active_model = 'default_model'
        else:
            self.active_model = active_model

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
        string = 'Atomic Graph summary:\n'
        string += '    Scale: {:.6f}\n'.format(self.scale)
        string += '    Order: {}\n'.format(self.order)
        string += '    Size: {}\n'.format(self.size)
        string += '    Chi: {:.3f}\n'.format(self.chi)
        string += '    Average degree: {:.3f}\n'.format(self.avg_degree)
        string += '    Matrix redshift: {:.3f}\n'.format(self.matrix_redshift)
        string += '    Particle redshift: {:.3f}\n'.format(self.particle_redshift)
        string += '    Total redshift: {:.3f}\n'.format(self.total_redshift)
        return string

    def vertex_report(self, i):
        string = self.vertices[i].report()
        return string

    def add_vertex(self, new_vertex):
        for i, vertex in enumerate(self.vertices):
            if vertex.void:
                vertex = new_vertex
                vertex.i = i
                break
        else:
            self.vertices.append(new_vertex)
            if not new_vertex.i == len(self.vertices) - 1:
                new_vertex.i = len(self.vertices) - 1
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
        self.vertices[i].void = True
        self.vertices[i].district = []
        self.num_void_vertices += 1
        self.reduced_order -= 1

        # Remap district sets all over the graph
        self.build_maps()
        self.summarize_stats()

    def get_non_void_vertices(self):
        vertices = []
        for vertex in self.vertices:
            if not vertex.void:
                vertices.append(vertex)
        return vertices

    def get_species(self):
        species = set()
        for item in self.species_dict.values():
            species.add(item['atomic_species'])
        return species

    def get_classes(self):
        classes = set()
        for item in self.species_dict:
            classes.add(item)
        return classes

    def get_arc(self, i, j):
        """Return the arc object adjacent from i to j, if any.

        :param i: Index of the tail vertex
        :param j: Index of the head vertex
        :type i: int
        :type j: int

        :returns Arc instance:
        :rtype graph_2.Arc:

        """
        result = None
        for arc in self.arcs:
            if arc.vertex_a.i == i and arc.vertex_b.i == j:
                result = arc
                break
        else:
            if j in self.vertices[i].out_neighbourhood:
                result = Arc(-1, self.vertices[i], self.vertices[j])
        return result

    def get_vertex_objects_from_indices(self, vertex_indices):
        """Get a list of Vertex objects corresponding to the indices given

        :param vertex_indices: List of indices
        :type vertex_indices: list(int)

        :returns List of vertex objects:
        :rtype list(graph_2.Vertex):

        """
        vertices = []
        for index in vertex_indices:
            vertices.append(self.vertices[index])
        return vertices

    def get_alpha_angles(self, i, selection_type='district'):
        pivot = (self.vertices[i].im_coor_x, self.vertices[i].im_coor_y)
        district = self.vertices[i].district
        if len(district) == 0:
            return []
        out_neighbourhood = self.vertices[i].out_neighbourhood
        partners = self.vertices[i].partners

        if selection_type == 'partners':
            j = []
            for citizen in district:
                if citizen in partners:
                    j.append(citizen)
                if len(j) == 3:
                    break
            else:
                for citizen in district:
                    if citizen not in j:
                        j.append(citizen)
                    if len(j) == 3:
                        break
            j.append(j[0])
            for k, index in enumerate(j):
                j[k] = (self.vertices[index].im_coor_x, self.vertices[index].im_coor_y)

        elif selection_type == 'zeta':
            j = []
            for citizen in district:
                if not self.vertices[citizen].zeta == self.vertices[i].zeta:
                    j.append(citizen)
                if len(j) == 3:
                    break
            else:
                for citizen in district:
                    if citizen not in j:
                        j.append(citizen)
                    if len(j) == 3:
                        break
            j.append(j[0])
            for k, index in enumerate(j):
                j[k] = (self.vertices[index].im_coor_x, self.vertices[index].im_coor_y)

        elif selection_type == 'out':
            j = []
            for citizen in district:
                if citizen in out_neighbourhood:
                    j.append(citizen)
                if len(j) == 3:
                    break
            else:
                for citizen in district:
                    if citizen not in j:
                        j.append(citizen)
                    if len(j) == 3:
                        break
            j.append(j[0])
            for k, index in enumerate(j):
                j[k] = (self.vertices[index].im_coor_x, self.vertices[index].im_coor_y)

        else:
            j_1 = (self.vertices[district[0]].im_coor_x, self.vertices[district[0]].im_coor_y)
            j_2 = (self.vertices[district[1]].im_coor_x, self.vertices[district[1]].im_coor_y)
            j_3 = (self.vertices[district[2]].im_coor_x, self.vertices[district[2]].im_coor_y)
            j = [j_1, j_2, j_3, j_1]

        alpha = []
        for k in range(0, 3):
            alpha.append(utils.find_angle_from_points(j[k], j[k + 1], pivot))

        if sum(alpha) > 6.5:
            for x in range(0, 3):
                alpha[x] = 2 * np.pi - alpha[x]

        return alpha

    def get_theta_angles(self, i):
        sub_graph = self.get_column_centered_subgraph(i)
        theta = []
        for mesh in sub_graph.meshes:
            theta.append(mesh.angles[0])
        return theta

    def get_redshift(self, i, j):
        hard_sphere_separation = self.get_hard_sphere_separation(i, j)
        actual_separation = self.get_separation(i, j)
        return hard_sphere_separation - actual_separation

    def get_image_separation(self, i, j):
        pos_i = self.vertices[i].im_pos()
        pos_j = self.vertices[j].im_pos()
        separation = np.sqrt((pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2 + (pos_i[2] - pos_j[2]) ** 2)
        return separation

    def get_projected_image_separation(self, i, j):
        pos_i = self.vertices[i].im_pos()
        pos_j = self.vertices[j].im_pos()
        projected_separation = np.sqrt((pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2)
        return projected_separation

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

    def get_hard_sphere_separation(self, i, j):
        radii_1 = self.species_dict[self.vertices[i].advanced_species]['atomic_radii']
        radii_2 = self.species_dict[self.vertices[j].advanced_species]['atomic_radii']
        return radii_1 + radii_2

    def get_im_separation_from_spatial_separation(self, spatial_searation):
        return spatial_searation / self.scale

    def get_spatial_separation_from_im_separation(self, im_separation):
        return im_separation * self.scale

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

    def get_spatial_district(self, i, n=8, exclude=None, return_separations=False):
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
        separations = []

        for k in range(0, n):
            min_index = projected_separations.index(min(projected_separations))
            district.append(indices[min_index])
            separations.append(projected_separations[min_index])
            projected_separations[min_index] = max(projected_separations) + 5

        if return_separations:
            return district, separations, n
        else:
            return district

    def get_anti_graph(self):
        anti_graph = AntiGraph(self)
        return anti_graph

    def get_mesh(self, i, j, mesh_index=0):
        indices, angles, vectors = self.get_mesh_numerics(i, j)
        mesh = Mesh(mesh_index, self.get_vertex_objects_from_indices(indices))
        mesh.angles = angles
        mesh.angle_vectors = vectors
        return mesh

    def get_induced_subgraph(self, vertex_indices):
        pass

    def get_mesh_centered_subgraph(self, i, j, order=1):
        sub_graph = SubGraph()
        mesh = self.get_mesh(i, j)
        sub_graph.add_mesh(mesh)
        sub_graph.finalize_init()
        return sub_graph

    def get_arc_centered_subgraph(self, i, j, order=1):
        mesh_0 = self.get_mesh(i, j, 0)
        mesh_1 = self.get_mesh(j, i, 1)
        sub_graph = SubGraph()
        sub_graph.add_mesh(mesh_0)
        sub_graph.add_mesh(mesh_1)
        sub_graph.finalize_init()
        return sub_graph

    def get_column_centered_subgraph(self, i, order=1):
        sub_graph = SubGraph()
        for mesh_index, neighbour in enumerate(self.vertices[i].neighbourhood):
            mesh = self.get_mesh(i, neighbour, mesh_index)
            sub_graph.add_mesh(mesh)
        sub_graph.finalize_init()
        return sub_graph

    def get_mesh_numerics(self, i, j, order=0):
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
                # logger.warning('Emergency stop!')
                stop = True
        angles = []
        vectors = []
        for m, corner in enumerate(corners):
            pivot = self.vertices[corner].im_pos()
            if m == 0:
                p1 = self.vertices[corners[len(corners) - 1]].im_pos()
                p2 = self.vertices[corners[m + 1]].im_pos()
            elif m == len(corners) - 1:
                p1 = self.vertices[corners[m - 1]].im_pos()
                p2 = self.vertices[corners[0]].im_pos()
            else:
                p1 = self.vertices[corners[m - 1]].im_pos()
                p2 = self.vertices[corners[m + 1]].im_pos()
            angle = utils.find_angle_from_points(p1[:2], p2[:2], pivot[:2])
            theta = angle / 2
            vector = (p1[0] - pivot[0], p1[1] - pivot[1])
            length = utils.vector_magnitude(vector)
            vector = (vector[0] / length, vector[1] / length)
            vector = (vector[0] * np.cos(theta) + vector[1] * np.sin(theta),
                      -vector[0] * np.sin(theta) + vector[1] * np.cos(theta))
            angles.append(angle)
            vectors.append(vector)
        return corners, angles, vectors

    def set_zeta(self, i, zeta):
        """Set the zeta value of the vertex with index i.

        :param i: vertex index
        :param zeta: Zeta
        :type i: int
        :type zeta: bool

        """
        if zeta is True:
            zeta = 1
        elif zeta is False:
            zeta = 0
        self.vertices[i].zeta = zeta

    def set_n(self, i, n):
        for key, item in self.species_dict.items():
            if item['symmetry'] == n:
                self.set_species(i, key)
                break
        else:
            logger.info('Could not set the symmetry of vertex {}, n = {} is not present in the species dictionary!'.format(i, n))

    def set_species(self, i, advanced_species):
        self.vertices[i].reset_probability_vector(bias=advanced_species)
        self.vertices[i].determine_species_from_probability_vector()
        self.build_local_map([i] + self.vertices[i].district)

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
                alpha = utils.find_angle_from_points(p1[:2], p2[:2], pivot[:2])
                if alpha < min_angle:
                    min_angle = alpha
                    next_index = k
        logger.debug('Found next: {}'.format(next_index))
        return min_angle, next_index

    def invert_levels(self):
        for vertex in self.vertices:
            if vertex.level == 0:
                vertex.level = 1
            else:
                vertex.level = 0

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
            vertex.flag_9 = False
        logger.info('All flags reset!')

    def build_local_map(self, indices, out_selection_type='district', build_out=True):
        # Determine out_neighbourhoods:
        if build_out:
            for vertex in self.get_vertex_objects_from_indices(indices):
                vertex.out_neighbourhood = set()
                if not vertex.void:
                    counter = 0

                    if out_selection_type == 'zeta':
                        for citizen in vertex.district:
                            if not vertex.zeta == self.vertices[citizen].zeta:
                                vertex.out_neighbourhood.add(citizen)
                                counter += 1
                            if counter == vertex.n:
                                break
                        else:
                            for alternative_citizen in vertex.district[0:vertex.n]:
                                if alternative_citizen not in vertex.out_neighbourhood:
                                    vertex.out_neighbourhood.add(alternative_citizen)
                                    counter += 1
                                if counter == vertex.n:
                                    break

                    elif out_selection_type == 'district':
                        for citizen in vertex.district:
                            vertex.out_neighbourhood.add(citizen)
                            counter += 1
                            if counter == vertex.n:
                                break

                    elif out_selection_type == 'separation':
                        for citizen in np.argsort(self.separation_matrix[vertex.i, :])[1:vertex.n + 1].tolist():
                            vertex.out_neighbourhood.add(citizen)

                    else:
                        logger.info('Unknown out-neighbourhood selection type')

        # Determine in_neighbourhoods:
        for vertex in self.get_vertex_objects_from_indices(indices):
            vertex.in_neighbourhood = set()
            if not vertex.void:
                for candidate in self.vertices:
                    if vertex.i in candidate.out_neighbourhood:
                        vertex.in_neighbourhood.add(candidate.i)
        # Determine neighbourhood:
        for vertex in self.get_vertex_objects_from_indices(indices):
            vertex.neighbourhood = set()
            if not vertex.void:
                vertex.neighbourhood = vertex.out_neighbourhood.union(vertex.in_neighbourhood)
        # Determine anti_neighbourhood:
        for vertex in self.get_vertex_objects_from_indices(indices):
            vertex.anti_neighbourhood = set()
            if not vertex.void:
                for citizen in vertex.district:
                    if citizen not in vertex.neighbourhood:
                        vertex.anti_neighbourhood.add(citizen)
        # Determine partners and semi-partners:
        for vertex in self.get_vertex_objects_from_indices(indices):
            vertex.partners = set()
            vertex.semi_partners = set()
            vertex.in_semi_partners = set()
            vertex.out_semi_partners = set()
            if not vertex.void:
                vertex.partners = vertex.out_neighbourhood.intersection(vertex.in_neighbourhood)
                vertex.semi_partners = vertex.neighbourhood - vertex.partners
                vertex.out_semi_partners = vertex.semi_partners.intersection(vertex.out_neighbourhood)
                vertex.in_semi_partners = vertex.semi_partners.intersection(vertex.in_neighbourhood)

                vertex.in_degree = len(vertex.in_neighbourhood)
                vertex.out_degree = len(vertex.out_neighbourhood)
                vertex.degree = len(vertex.neighbourhood)

    def build_maps(self, out_selection_type='district'):
        # Determine out_neighbourhoods:
        for vertex in self.vertices:
            vertex.out_neighbourhood = set()
            if not vertex.void:
                counter = 0

                if out_selection_type == 'zeta':
                    for citizen in vertex.district:
                        if not vertex.zeta == self.vertices[citizen].zeta:
                            vertex.out_neighbourhood.add(citizen)
                            counter += 1
                        if counter == vertex.n:
                            break
                    else:
                        for alternative_citizen in vertex.district[0:vertex.n]:
                            if alternative_citizen not in vertex.out_neighbourhood:
                                vertex.out_neighbourhood.add(alternative_citizen)
                                counter += 1
                            if counter == vertex.n:
                                break

                elif out_selection_type == 'district':
                    for citizen in vertex.district:
                        vertex.out_neighbourhood.add(citizen)
                        counter += 1
                        if counter == vertex.n:
                            break

                elif out_selection_type == 'separation':
                    for citizen in np.argsort(self.separation_matrix[vertex.i, :])[1:vertex.n + 1].tolist():
                        vertex.out_neighbourhood.add(citizen)

                else:
                    logger.info('Unknown out-neighbourhood selection type')

        # Determine in_neighbourhoods:
        for vertex in self.vertices:
            vertex.in_neighbourhood = set()
            if not vertex.void:
                for candidate in self.vertices:
                    if vertex.i in candidate.out_neighbourhood:
                        vertex.in_neighbourhood.add(candidate.i)
        # Determine neighbourhood:
        for vertex in self.vertices:
            vertex.neighbourhood = set()
            if not vertex.void:
                vertex.neighbourhood = vertex.out_neighbourhood.union(vertex.in_neighbourhood)
        # Determine anti_neighbourhood:
        for vertex in self.vertices:
            vertex.anti_neighbourhood = set()
            if not vertex.void:
                for citizen in vertex.district:
                    if citizen not in vertex.neighbourhood:
                        vertex.anti_neighbourhood.add(citizen)
        # Determine partners and semi-partners:
        for vertex in self.vertices:
            vertex.partners = set()
            vertex.semi_partners = set()
            vertex.in_semi_partners = set()
            vertex.out_semi_partners = set()
            if not vertex.void:
                vertex.partners = vertex.out_neighbourhood.intersection(vertex.in_neighbourhood)
                vertex.semi_partners = vertex.neighbourhood - vertex.partners
                vertex.out_semi_partners = vertex.semi_partners.intersection(vertex.out_neighbourhood)
                vertex.in_semi_partners = vertex.semi_partners.intersection(vertex.in_neighbourhood)

                vertex.in_degree = len(vertex.in_neighbourhood)
                vertex.out_degree = len(vertex.out_neighbourhood)
                vertex.degree = len(vertex.neighbourhood)

    def permute_j_k(self, i, j, k, map_type='district'):
        if self.vertices[i].permute_j_k(j, k):
            closed_district = copy.deepcopy(self.vertices[i].district)
            closed_district.append(i)
            self.vertices[i].out_neighbourhood.discard(j)
            self.vertices[i].out_neighbourhood.add(k)
            self.build_local_map(closed_district, out_selection_type=map_type, build_out=False)

    def weak_remove_edge(self, i, j, aggressive=False):

        config = self.get_column_centered_subgraph(i)
        options = []

        for mesh in config.meshes:
            for m, corner in enumerate(mesh.vertex_indices):
                if m not in [0, 1, len(mesh.vertex_indices) - 1]:
                    options.append(corner)
                if m == 1 and corner not in self.vertices[i].out_neighbourhood:
                    options.append(corner)

        for option in options:

            mesh_1 = self.get_mesh(i, option)
            mesh_2 = self.get_mesh(option, i)
            if mesh_1.order == 4 and mesh_2.order == 4:
                k = option
                break

        else:

            if aggressive:
                for option in options:
                    mesh_1 = self.get_mesh(i, option)
                    mesh_2 = self.get_mesh(option, i)
                    if mesh_1.order == 4 or mesh_2.order == 4:
                        k = option
                        break

                else:
                    return -1

            else:
                return -1

        return k

    def weak_preserve_edge(self, i, j):

        config = self.get_column_centered_subgraph(j)
        options = []

        for mesh in config.meshes:
            for m, corner in enumerate(mesh.vertex_indices):
                if (m == 1 or m == len(mesh.vertex_indices) - 1) and corner in self.vertices[j].out_neighbourhood and not corner == i:
                    options.append(corner)

        k = -1
        for option in options:
            mesh_1 = self.get_mesh(j, option)
            mesh_2 = self.get_mesh(option, j)
            if mesh_1.order == 3 and mesh_2.order == 3:
                k = option
                break

        return k

    def find_intersections(self):

        intersecting_segments = []

        for a in self.vertices:
            a_coor = a.im_pos()
            a_coor = (a_coor[0], a_coor[1])
            for b in [self.vertices[index] for index in a.out_neighbourhood]:
                if not a.is_edge_column and not b.is_edge_column:
                    b_coor = b.im_pos()
                    b_coor = (b_coor[0], b_coor[1])
                    for c in [self.vertices[index] for index in a.out_neighbourhood]:
                        if not c.i == b.i:
                            c_coor = c.im_pos()
                            c_coor = (c_coor[0], c_coor[1])
                            for d in [self.vertices[index] for index in c.out_neighbourhood]:
                                d_coor = d.im_pos()
                                d_coor = (d_coor[0], d_coor[1])
                                intersects = utils.closed_segment_intersect(a_coor, b_coor, c_coor, d_coor)
                                if intersects and (a.i, b.i, c.i, d.i) not in intersecting_segments and \
                                        (c.i, d.i, a.i, b.i) not in intersecting_segments:
                                    intersecting_segments.append((a.i, b.i, c.i, d.i))
                    for c in [self.vertices[index] for index in a.out_neighbourhood]:
                        c_coor = c.im_pos()
                        c_coor = (c_coor[0], c_coor[1])
                        for d in [self.vertices[index] for index in c.out_neighbourhood]:
                            d_coor = d.im_pos()
                            d_coor = (d_coor[0], d_coor[1])
                            for e in [self.vertices[index] for index in d.out_neighbourhood]:
                                e_coor = e.im_pos()
                                e_coor = (e_coor[0], e_coor[1])
                                intersects = utils.closed_segment_intersect(a_coor, b_coor, d_coor, e_coor)
                                if intersects and (a.i, b.i, d.i, e.i) not in intersecting_segments and \
                                        (d.i, e.i, a.i, b.i) not in intersecting_segments:
                                    intersecting_segments.append((a.i, b.i, d.i, e.i))

        return intersecting_segments

    def terminate_arc(self, i, j):
        if self.vertices[i].permute_j_k(j, self.vertices[i].district[self.vertices[i].n - 1]):
            if self.vertices[i].decrement_n():
                self.build_local_map([i] + self.vertices[i].district)
                return True
            else:
                return False
        else:
            return True

    def calc_vertex_parameters(self, i):
        vertex = self.vertices[i]
        vertex.alpha_angles = self.get_alpha_angles(i)
        if vertex.alpha_angles is not None and not len(vertex.alpha_angles) == 0:
            vertex.alpha_max = max(vertex.alpha_angles)
            vertex.alpha_min = min(vertex.alpha_angles)
        else:
            vertex.alpha_max = 0
            vertex.alpha_min = 0
        vertex.theta_angles = self.get_theta_angles(i)
        if vertex.theta_angles is not None and not len(vertex.theta_angles) == 0:
            vertex.theta_max = max(vertex.theta_angles)
            vertex.theta_min = min(vertex.theta_angles)
            vertex.theta_angle_variance = utils.variance(vertex.theta_angles)
            vertex.theta_angle_mean = utils.mean_val(vertex.theta_angles)
        else:
            vertex.theta_max = 0
            vertex.theta_min = 0
            vertex.theta_angle_variance = 0
            vertex.theta_angle_mean = 0

    def calc_normalized_gamma(self):
        peak_gammas = []
        avg_gammas = []
        for vertex in self.vertices:
            if not vertex.void:
                if not vertex.is_in_precipitate and vertex.atomic_species == 'Al':
                    peak_gammas.append(vertex.peak_gamma)
                    avg_gammas.append(vertex.avg_gamma)
        peak_mean = utils.mean_val(peak_gammas)
        avg_mean = utils.mean_val(avg_gammas)
        peak_mean_diff = peak_mean - 0.3
        avg_mean_diff = avg_mean - 0.3
        for vertex in self.vertices:
            vertex.normalized_peak_gamma = vertex.peak_gamma - peak_mean_diff
            vertex.normalized_avg_gamma = vertex.avg_gamma - avg_mean_diff

    def calc_redshifts(self):
        self.total_redshift = 0
        self.matrix_redshift = 0
        self.particle_redshift = 0
        anti_graph = AntiGraph(self)
        for vertex in self.vertices:
            if not vertex.is_edge_column and not vertex.void:
                vertex.redshift = 0
                vertex.avg_central_separation = 0
                counter = 0
                for partner in vertex.partners:
                    vertex.redshift += self.get_redshift(vertex.i, partner)
                    vertex.avg_central_separation += self.get_separation(vertex.i, partner)
                    counter += 1
                for partner in anti_graph.vertices[vertex.i].partners:
                    vertex.redshift += self.get_redshift(vertex.i, partner)
                    vertex.avg_central_separation += self.get_separation(vertex.i, partner)
                    counter += 1
                if not counter == 0:
                    vertex.avg_central_separation /= counter
                    vertex.avg_redshift = vertex.redshift / counter
                self.total_redshift += vertex.redshift
                if vertex.is_in_precipitate:
                    self.particle_redshift += vertex.redshift
                else:
                    self.matrix_redshift += vertex.redshift

    def calc_all_parameters(self):
        for vertex in self.vertices:
            if not vertex.void:
                self.calc_vertex_parameters(vertex.i)
        self.calc_normalized_gamma()
        self.calc_redshifts()

    def calc_model_predictions(self):
        for vertex in self.vertices:
            if not vertex.void:
                alpha_dict = {
                    'alpha_max': vertex.alpha_max,
                    'alpha_min': vertex.alpha_min
                }
                model_dict = {
                    'alpha_max': vertex.alpha_max,
                    'alpha_min': vertex.alpha_min,
                    'theta_max': vertex.theta_max,
                    'theta_min': vertex.theta_min,
                    'theta_angle_mean': vertex.theta_angle_mean,
                    'normalized_peak_gamma': vertex.normalized_peak_gamma,
                    'normalized_avg_gamma': vertex.normalized_avg_gamma
                }
                alpha_result = self.active_model.calc_prediction(alpha_dict)
                model_result = self.active_model.calc_prediction(model_dict)
                if self.active_model.categorization == 'advanced':
                    vertex.advanced_alpha_model = [
                        alpha_result['Si_1'],
                        alpha_result['Si_2'],
                        alpha_result['Cu'],
                        alpha_result['Al_1'],
                        alpha_result['Al_2'],
                        alpha_result['Mg_1'],
                        alpha_result['Mg_2'],
                    ]
                    vertex.alpha_model = Vertex.determine_simple_from_advanced_model(vertex.advanced_alpha_model)
                    vertex.advanced_model = [
                        model_result['Si_1'],
                        model_result['Si_2'],
                        model_result['Cu'],
                        model_result['Al_1'],
                        model_result['Al_2'],
                        model_result['Mg_1'],
                        model_result['Mg_2'],
                    ]
                    vertex.model = Vertex.determine_simple_from_advanced_model(vertex.advanced_model)
                elif self.active_model.categorization == 'simple':
                    vertex.alpha_model = [
                        alpha_result['Si'],
                        alpha_result['Cu'],
                        alpha_result['Al'],
                        alpha_result['Mg']
                    ]
                    vertex.advanced_alpha_model = Vertex.determine_advanced_from_simple_model(vertex.alpha_model)
                    vertex.model = [
                        model_result['Si'],
                        model_result['Cu'],
                        model_result['Al'],
                        model_result['Mg']
                    ]
                    vertex.advanced_model = Vertex.determine_advanced_from_simple_model(vertex.model)
                else:
                    logger.error('Unknown categorization')

    def refresh_graph(self):
        logger.info('Refreshing graph...')
        logger.info('    Mapping districts')
        self.build_maps(search_extended_district=False)
        logger.info('    Calculating vertex parameters')
        self.calc_all_parameters()
        logger.info('    Evaluating species variants')
        self.evaluate_sub_categories()
        # logger.info('    Calculating model predictions')
        # self.calc_model_predictions()
        # logger.info('    Mapping meshes')
        # self.map_meshes(0)
        logger.info('    Mapping arcs')
        self.map_arcs()
        logger.info('    Summarizing graph stats')
        self.summarize_stats()
        logger.info('Graph refreshed!')

    def evaluate_sub_categories(self):
        for vertex in self.vertices:
            if not vertex.void:
                if vertex.atomic_species == 'Si':
                    sub_graph = self.get_column_centered_subgraph(vertex.i)
                    for mesh in sub_graph.meshes:
                        if mesh.vertices[1].atomic_species == 'Mg' and mesh.vertices[2].atomic_species == 'Cu' and mesh.vertices[3].atomic_species == 'Mg':
                            if not vertex.advanced_species == 'Si_2':
                                self.set_species(vertex.i, 'Si_2')
                            break
                    else:
                        if not vertex.advanced_species == 'Si_1':
                            self.set_species(vertex.i, 'Si_1')
                elif vertex.atomic_species == 'Al':
                    if vertex.is_in_precipitate:
                        if not vertex.advanced_species == 'Al_2':
                            self.set_species(vertex.i, 'Al_2')
                    else:
                        if not vertex.advanced_species == 'Al_1':
                            self.set_species(vertex.i, 'Al_1')
                elif vertex.atomic_species == 'Mg':
                    if vertex.alpha_max < 3.175:
                        if not vertex.advanced_species == 'Mg_1':
                            self.set_species(vertex.i, 'Mg_1')
                    else:
                        if not vertex.advanced_species == 'Mg_2':
                            self. set_species(vertex.i, 'Mg_2')

    def calc_condensed_property_data(self, filter_=None, recalc=True, keys=None):

        if filter_ is None:
            filter_ = {
                'exclude_edge_columns': True,
                'exclude_matrix_columns': False,
                'exclude_particle_columns': False,
                'exclude_hidden_columns': False,
                'exclude_flag_1_columns': False,
                'exclude_flag_2_columns': False,
                'exclude_flag_3_columns': False,
                'exclude_flag_4_columns': False
            }

        if keys is None:
            keys = [
                'species_index', 'species_variant', 'advanced_category_index', 'alpha_max', 'alpha_min', 'theta_max',
                'theta_min', 'theta_angle_mean', 'normalized_peak_gamma', 'normalized_avg_gamma',
                'avg_central_separation'
            ]

        if recalc:
            self.refresh_graph()

        data = []
        for vertex in self.vertices:
            if not vertex.void:
                if not (filter_['exclude_edge_columns'] and vertex.is_edge_column):
                    if not (filter_['exclude_matrix_columns'] and not vertex.is_in_precipitate):
                        if not (filter_['exclude_particle_columns'] and vertex.is_in_precipitate):
                            if not (filter_['exclude_hidden_columns'] and not vertex.show_in_overlay):
                                if not (filter_['exclude_flag_1_columns'] and vertex.flag_1):
                                    if not (filter_['exclude_flag_2_columns'] and vertex.flag_2):
                                        if not (filter_['exclude_flag_3_columns'] and vertex.flag_3):
                                            if not (filter_['exclude_flag_4_columns'] and vertex.flag_4):

                                                data_item = {}
                                                for key in keys:
                                                    data_item[key] = eval('vertex.{}'.format(key))
                                                data.append(data_item)

        return data

    def map_arcs(self):
        self.arcs = []
        self.size = 0
        for vertex in self.vertices:
            if not vertex.void:
                for out_neighbour in self.get_vertex_objects_from_indices(vertex.out_neighbourhood):
                    if not out_neighbour.void:
                        arc = Arc(len(self.arcs), vertex, out_neighbour)
                        self.arcs.append(arc)
                        self.size += 1

    def map_meshes(self, i):
        """Automatically generate a connected relational map of all meshes in graph.

        The index of a mesh is temporarily indexed during the mapping by the following algorithm: Take the indices of
        its corners, and circularly permute them such that the lowest index comes first. After the mapping is complete,
        these indices are replaced by the integers 0 to the number of meshes.

        :param i: Index of starting vertex.
        :type i: int

        """
        print('debug 1')
        if not len(self.vertices[i].district) == 0:
            self.meshes = []
            self.mesh_indices = []
            sub_graph_0 = self.get_column_centered_subgraph(i)
            mesh_0 = sub_graph_0.meshes[0]
            mesh_0.mesh_index = self.determine_temp_index(mesh_0)
            print('debug 2')
            self.meshes.append(mesh_0)
            self.mesh_indices.append(mesh_0.mesh_index)
            print('debug 3')
            sys.setrecursionlimit(100000)
            self.walk_mesh_edges(mesh_0)
            print('debug 4')
            new_indices = [i for i in range(0, len(self.mesh_indices))]

            for k, mesh in enumerate(self.meshes):
                for j, neighbour in enumerate(mesh.surrounding_meshes):
                    mesh.surrounding_meshes[j] = self.mesh_indices.index(neighbour)
                mesh.mesh_index = self.mesh_indices.index(mesh.mesh_index)
            print('debug 5')
            self.mesh_indices = new_indices

    def walk_mesh_edges(self, mesh, counter=0):
        for k, corner in enumerate(vertex.i for vertex in mesh.vertices):
            new_mesh = self.get_mesh(corner, mesh.vertices[k - 1].i, 0)
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
                    print(counter)
                    if counter == 1395:
                        print(mesh)
                    self.walk_mesh_edges(new_mesh, counter=(counter + 1))

    @staticmethod
    def determine_temp_index(mesh):
        return utils.make_int_from_list(utils.cyclic_sort(mesh.vertex_indices))

    def summarize_stats(self):
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
        if not self.size == 0:
            self.chi = num_weak_arcs / self.size
        else:
            self.chi = 0

        # Calc average degree
        counted_columns = 0
        degrees = 0
        for vertex in self.vertices:
            if not vertex.is_edge_column and not vertex.void:
                degrees += vertex.degree
                counted_columns += 1
        if not counted_columns == 0:
            self.avg_degree = degrees / counted_columns
        else:
            self.avg_degree = 0


class Mesh:

    def __init__(self, mesh_index, vertices):

        self.mesh_index = mesh_index
        self.vertices = vertices
        self.vertex_indices = []
        self.arcs = []
        self.angles = []
        self.angle_vectors = []
        self.surrounding_meshes = []

        self.is_enclosed = True
        self.is_consistent = True
        self.order = 0
        self.size = 0
        self.cm = (0, 0)

        for vertex in self.vertices:
            self.vertex_indices.append(vertex.i)
            self.order += 1
        for vertex in self.vertices:
            for out_neighbour in vertex.out_neighbourhood:
                if out_neighbour in self.vertex_indices:
                    index = self.vertex_indices.index(out_neighbour)
                    self.arcs.append(Arc(self.size, vertex, self.vertices[index]))
                    self.size += 1
        self.calc_cm()

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
            x_fit += corner.im_coor_x
            y_fit += corner.im_coor_y
        x_fit = x_fit / mass
        y_fit = y_fit / mass
        self.cm = (x_fit, y_fit)

    def test_sidedness(self):
        if not self.order == 4:
            return False
        else:
            return True

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

                if mesh_1.vertex_indices[utils.circularize_next_index(k - 1, k_max)] == edge[1] or \
                        mesh_1.vertex_indices[utils.circularize_next_index(k + 1, k_max) == edge[1]]:
                    pass
                else:
                    assure_edge = False

            elif corner == edge[1]:

                if mesh_1.vertex_indices[utils.circularize_next_index(k - 1, k_max)] == edge[0] or \
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


class SubGraph:

    def __init__(self):
        self.vertices = []
        self.vertex_indices = []
        self.arcs = []
        self.meshes = []

        self.num_vertices = 0
        self.num_arcs = 0
        self.num_meshes = 0

        self.class_ = None
        self.configuration = None

    def finalize_init(self):
        self.redraw_edges()
        self.sort_meshes()
        self.summarize_stats()

    def summarize_stats(self):
        self.num_vertices = len(self.vertices)
        self.num_arcs = len(self.arcs)
        self.num_meshes = len(self.meshes)

    def sort_meshes(self):
        new_list = []
        for mesh in self.meshes:
            if mesh.vertex_indices[1] == self.vertices[0].district[0]:
                new_list.append(mesh)
                break
        closed = False
        if len(self.meshes) == 0:
            closed = True
        backup_counter = 0
        while not closed:
            for mesh in self.meshes:
                if mesh.vertex_indices[1] ==\
                        new_list[-1].vertex_indices[-1]:
                    new_list.append(mesh)

                    if new_list[-1].vertex_indices[-1] ==\
                            new_list[0].vertex_indices[1]:
                        closed = True
                    break
            backup_counter += 1
            if backup_counter > 26:
                break
        if len(new_list) == len(self.meshes) and closed:
            self.meshes = new_list

    def add_vertex(self, vertex):
        self.vertices.append(vertex)
        self.vertex_indices.append(vertex.i)
        self.num_vertices += 1

    def add_mesh(self, mesh):
        self.meshes.append(mesh)
        self.num_meshes += 1
        for vertex in self.meshes[-1].vertices:
            if vertex.i not in self.vertex_indices:
                self.add_vertex(vertex)

    def get_ind_from_mother(self, i):
        for index, mother_index in enumerate(self.vertex_indices):
            if mother_index == i:
                sub_index = index
                break
        else:
            sub_index = -1
        return sub_index

    def remove_vertex(self, vertex_index):
        raise NotImplemented

    def increase_h(self, i):
        i = self.get_ind_from_mother(i)
        if not i == -1:
            changed = self.vertices[i].increment_species_index()
        else:
            changed = False
        return changed

    def decrease_h(self, i):
        i = self.get_ind_from_mother(i)
        if not i == -1:
            changed = self.vertices[i].decrement_species_index()
        else:
            changed = False
        return changed

    def add_arc(self, j, vertex_a, vertex_b):
        arc = Arc(j, vertex_a, vertex_b)
        self.arcs.append(arc)
        self.num_arcs += 1

    def remove_arcs(self, arc_index):
        raise NotImplemented

    def redraw_edges(self):
        self.arcs = []
        self.num_arcs = 0
        for vertex in self.vertices:
            for out_neighbour in vertex.out_neighbourhood:
                if out_neighbour in self.vertex_indices:
                    self.add_arc(self.num_arcs, vertex, self.vertices[self.get_ind_from_mother(out_neighbour)])


class AntiGraph:

    def __init__(self, graph):

        self.graph = graph
        self.vertices = copy.deepcopy(graph.vertices)
        self.arcs = []
        self.size = 0

        self.build()

    def build(self):
        for i, vertex in enumerate(self.graph.vertices):
            if not vertex.is_edge_column and not vertex.void:
                sub_graph = self.graph.get_column_centered_subgraph(vertex.i)
                for mesh in sub_graph.meshes:
                    self.vertices[i].permute_j_k(mesh.vertex_indices[1], mesh.vertex_indices[2])
        self.graph = AtomicGraph(self.graph.scale)
        for vertex in self.vertices:
            self.graph.add_vertex(vertex)
        self.graph.build_maps()
        self.graph.summarize_stats()

    def map_arcs(self):
        self.arcs = []
        self.size = 0
        for vertex in self.vertices:
            if not vertex.void:
                for out_neighbour in self.graph.get_vertex_objects_from_indices(vertex.out_neighbourhood):
                    if not out_neighbour.void:
                        arc = Arc(len(self.arcs), vertex, out_neighbour)
                        self.arcs.append(arc)
                        self.size += 1



