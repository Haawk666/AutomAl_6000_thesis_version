
import numpy as np
import utils
import params
from copy import deepcopy
import logging

# Instantiate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def sort_neighbourhood(graph_obj):

    for vertex in graph_obj.vertices:
        i_level = vertex.level
        tmp_1 = []
        tmp_2 = []
        for j, neighbour in enumerate(vertex.neighbour_indices):
            j_level = graph_obj.vertices[neighbour].level
            if i_level == j_level:
                tmp_2.append(neighbour)
            else:
                tmp_1.append(neighbour)
        vertex.neighbour_indices = tmp_1 + tmp_2


def statistical_level_bleed(graph_obj, starting_index, level):

    graph_obj.reset_all_flags()

    if level == 0:
        graph_obj.vertices[starting_index].level_vector = [0.9, 0.1]
    elif level == 1:
        graph_obj.vertices[starting_index].level_vector = [0.1, 0.9]

    runs = []
    measures = []

    for m in range(0, 20):
        level_tree_traverse(graph_obj, starting_index)
        runs.append(m)
        measures.append(graph_obj.calc_avg_level_confidence())
        graph_obj.reset_all_flags()

    return runs, measures


def level_tree_traverse(graph_obj, i):
    conf = graph_obj.vertices[i].analyse_level_vector_confidence()
    level = graph_obj.vertices[i].set_level_from_vector()
    edgy = graph_obj.vertices[i].is_edge_column
    graph_obj.vertices[i].flag_1 = True

    for partner in graph_obj.vertices[i].partners():
        if not edgy:
            if level == 0:
                graph_obj.vertices[partner].level_vector[0] = graph_obj.vertices[partner].level_vector[0]
                graph_obj.vertices[partner].level_vector[1] = graph_obj.vertices[partner].level_vector[1] + conf
            elif level == 1:
                graph_obj.vertices[partner].level_vector[1] = graph_obj.vertices[partner].level_vector[1]
                graph_obj.vertices[partner].level_vector[0] = graph_obj.vertices[partner].level_vector[0] + conf
            else:
                print('Error in graph_op.level_tree_traverse')
            graph_obj.vertices[partner].renorm_level_vector()
        if not graph_obj.vertices[partner].flag_1:
            level_tree_traverse(graph_obj, partner)


def experimental_remove_intersections(graph_obj):
    intersections = graph_obj.find_intersects()

    for intersection in intersections:

        edge_1 = (intersection[0], intersection[1])
        edge_2 = (intersection[2], intersection[3])

        if graph_obj.vertices[edge_1[0]].partner_query(edge_1[1]):
            if edge_2[0] in graph_obj.vertices[edge_1[0]].anti_partners():
                graph_obj.perturb_j_k(edge_1[0], edge_1[1], edge_2[0])
            else:
                if not graph_obj.strong_remove_edge(edge_1[0], edge_1[1]):
                    print('Could not remove {} {}'.format(edge_1[0], edge_1[1]))

        if graph_obj.vertices[edge_2[0]].partner_query(edge_2[1]):
            if edge_1[0] in graph_obj.vertices[edge_2[0]].anti_partners():
                graph_obj.perturb_j_k(edge_2[0], edge_2[1], edge_1[0])
            else:
                if not graph_obj.strong_remove_edge(edge_2[0], edge_2[1]):
                    print('Could not remove {} {}'.format(edge_2[0], edge_2[1]))


def remove_intersections(graph_obj):

    intersections = graph_obj.find_intersects()
    remove_edges = []
    strong_intersections = []
    weak_weak_intersections = 0
    strong_stong_intersections = 0
    not_removed = 0

    # First identify inconsistent edges that cross consistent edges

    for intersection in intersections:

        edge_1 = (intersection[0], intersection[1])
        edge_2 = (intersection[2], intersection[3])

        if graph_obj.vertices[edge_1[1]].partner_query(edge_1[0]):
            edge_1_is_strong = True
        else:
            edge_1_is_strong = False

        if graph_obj.vertices[edge_2[1]].partner_query(edge_2[0]):
            edge_2_is_strong = True
        else:
            edge_2_is_strong = False

        if edge_1_is_strong and not edge_2_is_strong:

            if edge_2 not in remove_edges:
                remove_edges.append(edge_2)

        elif not edge_1_is_strong and edge_2_is_strong:

            if edge_1 not in remove_edges:
                remove_edges.append(edge_1)

        elif not edge_1_is_strong and not edge_2_is_strong:

            if edge_1[0] not in graph_obj.vertices[edge_2[0]].partners():
                graph_obj.perturb_j_k(edge_2[0], edge_2[1], edge_1[0])
            else:
                if edge_2 not in remove_edges:
                    remove_edges.append(edge_2)

            if edge_2[0] not in graph_obj.vertices[edge_1[0]].partners():
                graph_obj.perturb_j_k(edge_1[0], edge_2[0], edge_1[1])
            else:
                if edge_1 not in remove_edges:
                    remove_edges.append(edge_1)

        else:

            permutations = []
            permutations.append((edge_1[0], edge_1[1], edge_2[0], edge_2[1]))
            permutations.append((edge_1[0], edge_1[1], edge_2[1], edge_2[0]))
            permutations.append((edge_1[0], edge_1[1], edge_2[0], edge_2[1]))
            permutations.append((edge_1[0], edge_1[1], edge_2[1], edge_2[0]))
            permutations.append((edge_2[0], edge_2[1], edge_1[0], edge_1[1]))
            permutations.append((edge_2[0], edge_2[1], edge_1[1], edge_1[0]))
            permutations.append((edge_2[0], edge_2[1], edge_1[0], edge_1[1]))
            permutations.append((edge_2[0], edge_2[1], edge_1[1], edge_1[0]))

            add = True

            for permutation in permutations:

                if permutation in strong_intersections:

                    add = False

            if add:
                strong_intersections.append(permutations[0])
                print(permutations[0])
                strong_stong_intersections += 1

    for edge in remove_edges:
        # Test inclusion of weak remove here!
        if not graph_obj.strong_remove_edge(edge[0], edge[1]):
            not_removed += 1
            print()

    graph_obj.redraw_edges()
    graph_obj.summarize_stats()

    return not_removed, strong_intersections, weak_weak_intersections, strong_stong_intersections


def base_pca_score(graph_obj, i, apply=True):

    alpha = graph_obj.produce_alpha_angles(i)

    if apply:
        return alpha
    else:
        return max(alpha), min(alpha)


def base_stat_score(graph_obj, i):

    alpha = graph_obj.produce_alpha_angles(i)
    alpha_min = min(alpha)
    alpha_max = max(alpha)

    theta = graph_obj.produce_theta_angles(i, exclude_angles_from_inconsistent_meshes=True)
    theta_min = min(theta)
    theta_max = max(theta)

    theta_avg = graph_obj.produce_theta_mean(i, exclude_angles_from_inconsistent_meshes=True)
    normalized_peak_gamma = graph_obj.vertices[i].normalized_peak_gamma
    normalized_avg_gamma = graph_obj.vertices[i].normalized_avg_gamma

    values = [alpha_min, alpha_max, theta_min, theta_max, theta_avg, normalized_peak_gamma, normalized_avg_gamma]
    parameters = params.produce_params()
    coefficients = []

    for i, v in enumerate(values):
        value_coefficients = []
        for mu, sigma in parameters[i]:
            value_coefficients.append(utils.normal_dist(v, mu, sigma))
        coefficients.append(value_coefficients)
    probs = []

    for value_coefficients in coefficients:
        probability = 1
        for coefficient in value_coefficients:
            probability *= coefficient
        probs.append(probability)

    reduced_probability = [probs[0], probs[1] + probs[2], 0, probs[3] + probs[4], 0, probs[5] + probs[6], 0]

    return reduced_probability


def base_angle_score(graph_obj, i, apply=True):

    alpha = graph_obj.produce_alpha_angles(i)

    # cu_min_mean = 1.92
    # si_min_mean = 1.69
    # al_min_mean = 1.56
    # mg_min_mean = 1.26
    #
    # cu_min_std = 0.19
    # si_min_std = 0.21
    # al_min_std = 0.05
    # mg_min_std = 0.05
    #
    # cu_max_mean = 2.28
    # si_max_mean = 2.40
    # al_max_mean = 3.11
    # mg_max_mean = 3.50
    #
    # cu_max_std = 0.26
    # si_max_std = 0.17
    # al_max_std = 0.07
    # mg_max_std = 0.42

    # cu_min_mean = 1.99
    # si_min_mean = 1.92
    # al_min_mean = 1.53
    # mg_min_mean = 1.24
    #
    # cu_min_std = 0.03
    # si_min_std = 0.10
    # al_min_std = 0.06
    # mg_min_std = 0.08
    #
    # cu_max_mean = 2.25
    # si_max_mean = 2.29
    # al_max_mean = 3.11
    # mg_max_mean = 2.90
    #
    # cu_max_std = 0.06
    # si_max_std = 0.12
    # al_max_std = 0.09
    # mg_max_std = 0.49

    # cu_min_mean = 1.96
    # si_min_mean = 1.81
    # al_min_mean = 1.55
    # mg_min_mean = 1.25
    #
    # cu_min_std = 0.11
    # si_min_std = 0.16
    # al_min_std = 0.06
    # mg_min_std = 0.07
    #
    # cu_max_mean = 2.27
    # si_max_mean = 2.35
    # al_max_mean = 3.11
    # mg_max_mean = 3.20
    #
    # cu_max_std = 0.16
    # si_max_std = 0.15
    # al_max_std = 0.08
    # mg_max_std = 0.46

    if apply:

        cu_min_mean = 1.92
        si_1_min_mean = 1.94
        si_2_min_mean = 1.56
        al_min_mean = 1.56
        mg_1_min_mean = 1.30
        mg_2_min_mean = 1.26

        cu_min_std = 0.19
        si_1_min_std = 0.14
        si_2_min_std = 0.08
        al_min_std = 0.05
        mg_1_min_std = 0.09
        mg_2_min_std = 0.05

        cu_max_mean = 2.28
        si_1_max_mean = 2.25
        si_2_max_mean = 2.48
        al_max_mean = 3.11
        mg_1_max_mean = 2.57
        mg_2_max_mean = 3.69

        cu_max_std = 0.26
        si_1_max_std = 0.12
        si_2_max_std = 0.15
        al_max_std = 0.07
        mg_1_max_std = 0.10
        mg_2_max_std = 0.09

        cf_cu_min = utils.normal_dist(min(alpha), cu_min_mean, cu_min_std)
        cf_si_1_min = utils.normal_dist(min(alpha), si_1_min_mean, si_1_min_std)
        cf_si_2_min = utils.normal_dist(min(alpha), si_2_min_mean, si_2_min_std)
        cf_al_min = utils.normal_dist(min(alpha), al_min_mean, al_min_std)
        cf_mg_1_min = utils.normal_dist(min(alpha), mg_1_min_mean, mg_1_min_std)
        cf_mg_2_min = utils.normal_dist(min(alpha), mg_2_min_mean, mg_2_min_std)

        cf_cu_max = utils.normal_dist(max(alpha), cu_max_mean, cu_max_std)
        cf_si_1_max = utils.normal_dist(max(alpha), si_1_max_mean, si_1_max_std)
        cf_si_2_max = utils.normal_dist(max(alpha), si_2_max_mean, si_2_max_std)
        cf_al_max = utils.normal_dist(max(alpha), al_max_mean, al_max_std)
        cf_mg_1_max = utils.normal_dist(max(alpha), mg_1_max_mean, mg_1_max_std)
        cf_mg_2_max = utils.normal_dist(max(alpha), mg_2_max_mean, mg_2_max_std)

        cf_min = [cf_cu_min, cf_si_1_min, cf_si_2_min, cf_al_min, cf_mg_1_min, cf_mg_2_min]
        cf_max = [cf_cu_max, cf_si_1_max, cf_si_2_max, cf_al_max, cf_mg_1_max, cf_mg_2_max]

        cf = [a * b for a, b in zip(cf_min, cf_max)]
        probs = utils.normalize_list(cf)
        sum_probs = [probs[1] + probs[2], probs[0], 0, probs[3], 0, probs[4] + probs[5], 0]

        print('alpha: {}\nmax: {}, min: {}\ncf_min: {}\ncf_max: {}\ncf: {}\nprobs: {}\nsum_probs: {}\n\n'.format(alpha, max(alpha), min(alpha), cf_min, cf_max, cf, probs, sum_probs))

        return sum_probs

    else:

        return max(alpha), min(alpha)


def mesh_angle_score(graph_obj, i, dist_3_std, dist_4_std, dist_5_std):

    _, meshes = graph_obj.get_atomic_configuration(i, True)

    mu_3 = 2 * np.pi / 3
    mu_4 = np.pi / 2
    mu_5 = 2 * np.pi / 5

    probs = [1/3, 1/3, 1/3]

    for mesh in meshes:
        if mesh.test_sidedness():
            cf_3 = utils.normal_dist(mesh.angles[0], mu_3, dist_3_std)
            cf_4 = utils.normal_dist(mesh.angles[0], mu_4, dist_4_std)
            cf_5 = utils.normal_dist(mesh.angles[0], mu_5, dist_5_std)
            cf = [cf_3, cf_4, cf_5]
            probs = [a * b for a, b in zip(cf, probs)]
            probs = utils.normalize_list(probs)

    probs = [a * b for a, b in zip(probs, graph_obj.vertices[i].symmetry_vector)]
    probs = utils.normalize_list(probs)

    return probs


def apply_angle_score(graph_obj, i, dist_3_std, dist_4_std, dist_5_std, num_selections):

    n = 3

    a = np.ndarray([n], dtype=np.int)
    b = np.ndarray([n], dtype=np.int)
    alpha = np.ndarray([n], dtype=np.float64)

    for x in range(0, n):
        a[x] = graph_obj.vertices[graph_obj.vertices[i].neighbour_indices[x]].real_coor_x - graph_obj.vertices[i].real_coor_x
        b[x] = graph_obj.vertices[graph_obj.vertices[i].neighbour_indices[x]].real_coor_y - graph_obj.vertices[i].real_coor_y

    for x in range(0, n):

        x_pluss = 0

        if x == n - 1:

            pass

        else:

            x_pluss = x + 1

        alpha[x] = utils.find_angle(a[x], a[x_pluss], b[x], b[x_pluss])

        # Deal with cases where the angle is over pi radians:
        max_index = alpha.argmax()
        angle_sum = 0.0
        for x in range(0, n):
            if x == max_index:
                pass
            else:
                angle_sum = angle_sum + alpha[x]
        if alpha.max() == angle_sum:
            alpha[max_index] = 2 * np.pi - alpha.max()

    symmetry_3 = 2 * np.pi / 3
    symmetry_4 = np.pi / 2
    symmetry_5 = 2 * np.pi / 5

    correction_factor_3 = utils.normal_dist(alpha.max(), symmetry_3, dist_3_std)
    correction_factor_4 = utils.normal_dist(alpha.max(), 2 * symmetry_4, dist_4_std)
    correction_factor_5 = utils.normal_dist(alpha.min(), symmetry_5, dist_5_std)

    if alpha.min() < symmetry_5:
        correction_factor_5 = utils.normal_dist(symmetry_5, symmetry_5, dist_5_std)

    for k in range(0, num_selections):
        if k == 0 or k == 1:
            graph_obj.vertices[i].prob_vector[k] *= correction_factor_3
        elif k == 3:
            graph_obj.vertices[i].prob_vector[k] *= correction_factor_4
        elif k == 5:
            graph_obj.vertices[i].prob_vector[k] *= correction_factor_5
        elif k == 6:
            graph_obj.vertices[i].prob_vector[k] *= 0

    graph_obj.vertices[i].renorm_prob_vector()

    correction_factor_3 = utils.normal_dist(alpha.min(), symmetry_3, dist_3_std)
    correction_factor_4 = utils.normal_dist(alpha.min(), symmetry_4, dist_4_std)

    for k in range(0, num_selections):
        if k == 0 or k == 1:
            graph_obj.vertices[i].prob_vector[k] *= correction_factor_3
        elif k == 3:
            graph_obj.vertices[i].prob_vector[k] *= correction_factor_4
        elif k == 5:
            graph_obj.vertices[i].prob_vector[k] *= correction_factor_5
        elif k == 6:
            graph_obj.vertices[i].prob_vector[k] *= 0

    graph_obj.vertices[i].renorm_prob_vector()
    graph_obj.vertices[i].define_species()


def apply_intensity_score(graph_obj, i, num_selections, intensities, dist_8_std):

    for x in range(0, num_selections):
        graph_obj.vertices[i].prob_vector[x] *= utils.normal_dist(
            graph_obj.vertices[i].peak_gamma, intensities[x], dist_8_std)

    graph_obj.vertices[i].renorm_prob_vector()
    graph_obj.vertices[i].define_species()


def apply_dist_score(graph_obj, i, other_radii, num_selections, radii, dist_1_std, dist_2_std):

    for y in range(0, num_selections):

        if radii[y] <= other_radii:
            graph_obj.vertices[i].prob_vector[y] *= utils.normal_dist(
                radii[y], other_radii, dist_1_std)
        else:
            graph_obj.vertices[i].prob_vector[y] *= utils.normal_dist(
                radii[y], other_radii, dist_2_std)

    graph_obj.vertices[i].renorm_prob_vector()
    graph_obj.vertices[i].define_species()


def set_levels_basic(graph_obj, i, level, report, indent_string='        '):

    if level is not None:

        if not graph_obj.vertices[i].flag_1:
            graph_obj.vertices[i].level = level
            graph_obj.vertices[i].flag_1 = True
            report('{}Set vertex {} to level {}'.format(indent_string, i, level), force=False)
        else:
            report('{}Vertex {} already set to level {}'.format(indent_string, i, level), force=False)

        anti_level = graph_obj.vertices[i].anti_level()

        j = graph_obj.vertices[i].neighbour_indices[0]
        k = graph_obj.vertices[i].neighbour_indices[1]
        l = graph_obj.vertices[i].neighbour_indices[2]

        if anti_level is not None:
            if not graph_obj.vertices[j].flag_1:
                report('{}    Neighbour 1 was not set. Sending {}'.format(indent_string, j), force=False)
                set_levels_basic(graph_obj, j, anti_level, report, indent_string=indent_string + '        ')
                if not graph_obj.vertices[k].flag_1:
                    report('{}    Neighbour 2 was not set. Sending {}'.format(indent_string, k), force=False)
                    set_levels_basic(graph_obj, k, anti_level, report, indent_string=indent_string + '        ')
                    if not graph_obj.vertices[l].flag_1:
                        report('{}    Neighbour 3 was not set. Sending {}'.format(indent_string, l), force=False)
                        set_levels_basic(graph_obj, l, anti_level, report, indent_string=indent_string + '        ')


def set_levels(graph_obj, i, report, level, num_selections):

    report('        Setting levels of matrix columns...', force=True)
    graph_obj.reset_all_flags()
    set_matrix_levels(graph_obj, i, level, report)
    report('        Matrix levels set.', force=True)
    report('        Setting levels of particle columns...', force=True)
    complete = False
    emer_abort = False
    overcounter = 0
    neighbour_level = 0

    while not complete and not emer_abort:

        found = False
        counter = 0

        while counter < graph_obj.num_vertices:

            if graph_obj.vertices[counter].is_in_precipitate and not graph_obj.vertices[counter].flag_1:

                x = 0

                while x <= neighbour_level:

                    if graph_obj.vertices[graph_obj.vertices[counter].neighbour_indices[x]].is_in_precipitate and \
                            graph_obj.vertices[graph_obj.vertices[counter].neighbour_indices[x]].flag_1:

                        neighbour = graph_obj.vertices[counter].neighbour_indices[x]
                        if graph_obj.vertices[neighbour].level == 0:
                            graph_obj.vertices[counter].level = 1
                        else:
                            graph_obj.vertices[counter].level = 0
                            graph_obj.vertices[counter].flag_1 = True
                        found = True

                    x = x + 1

            counter = counter + 1

        complete = True

        for y in range(0, graph_obj.num_vertices):

            if graph_obj.vertices[y].is_in_precipitate and not graph_obj.vertices[y].flag_1:
                complete = False

        if found and neighbour_level > 0:
            neighbour_level = neighbour_level - 1

        if not found and neighbour_level < 2:
            neighbour_level = neighbour_level + 1

        overcounter += 1
        if overcounter > 100:
            emer_abort = True
            report('            Emergency abort', force=True)

    graph_obj.reset_all_flags()
    report('        Particle levels set.', force=True)


def set_matrix_levels(graph_obj, i, level, report):

    if graph_obj.vertices[i].is_in_precipitate:

        graph_obj.vertices[i].flag_1 = True
        graph_obj.set_level(i, level)

    else:

        graph_obj.vertices[i].flag_1 = True

        next_level = 0
        if level == 0:
            next_level = 1
        elif level == 1:
            next_level = 0
        else:
            report('            Disaster!', force=True)

        graph_obj.set_level(i, level)

        indices = graph_obj.vertices[i].neighbour_indices

        for x in range(0, graph_obj.vertices[i].n()):

            reciprocal = graph_obj.test_reciprocality(i, indices[x])

            if not graph_obj.vertices[indices[x]].flag_1 and not graph_obj.vertices[i].is_edge_column and reciprocal:
                set_matrix_levels(graph_obj, indices[x], next_level, report)


def set_particle_levels(graph_obj, i, level, report):
    if not graph_obj.vertices[i].is_in_precipitate:

        graph_obj.vertices[i].flag_1 = True

    else:

        graph_obj.vertices[i].flag_1 = True

        next_level = 0
        if level == 0:
            next_level = 1
        elif level == 1:
            next_level = 0
        else:
            report('            Disaster!', force=True)

        graph_obj.set_level(i, level)

        indices = graph_obj.vertices[i].neighbour_indices

        complete = False
        counter_1 = 0
        counter_2 = 0

        while not complete:

            if not graph_obj.vertices[indices[counter_1]].flag_1:

                if graph_obj.test_reciprocality(i, indices[counter_1]):

                    graph_obj.precipitate_levels(indices[counter_1], next_level)
                    counter_1 = counter_1 + 1
                    counter_2 = counter_2 + 1

                else:

                    counter_1 = counter_1 + 1

            else:

                counter_1 = counter_1 + 1

            if counter_2 == graph_obj.vertices[i].n() - 2 or counter_1 == graph_obj.vertices[i].n() - 2:
                complete = True


def precipitate_controller(graph_obj, i):

    graph_obj.reset_all_flags()

    precipitate_finder(graph_obj, i)

    counter = 0

    for x in range(0, graph_obj.num_vertices):

        if graph_obj.vertices[x].flag_1 or graph_obj.vertices[x].h_index == 6:
            graph_obj.vertices[x].is_in_precipitate = False
        else:
            graph_obj.vertices[x].is_in_precipitate = True

        if graph_obj.vertices[x].flag_2:

            graph_obj.particle_boarder.append(x)

            counter = counter + 1

    graph_obj.reset_all_flags()
    sort_boarder(graph_obj)


def sort_boarder(graph_obj):

    temp_boarder = deepcopy(graph_obj.particle_boarder)
    selected = np.ndarray([len(graph_obj.particle_boarder)], dtype=bool)
    for y in range(0, len(graph_obj.particle_boarder)):
        selected[y] = False
    next_index = 0
    index = 0
    cont_var = True
    selected[0] = True

    while cont_var:

        distance = 10000000

        for x in range(0, len(graph_obj.particle_boarder)):

            current_distance = np.sqrt((graph_obj.vertices[graph_obj.particle_boarder[x]].real_coor_x -
                                        graph_obj.vertices[temp_boarder[index]].real_coor_x)**2 +
                                       (graph_obj.vertices[graph_obj.particle_boarder[x]].real_coor_y -
                                        graph_obj.vertices[temp_boarder[index]].real_coor_y)**2)

            if current_distance < distance and not temp_boarder[index] == graph_obj.particle_boarder[x] and not selected[x]:
                distance = current_distance
                next_index = x

        selected[next_index] = True
        index = index + 1

        temp_boarder[index] = graph_obj.particle_boarder[next_index]

        if index == len(graph_obj.particle_boarder) - 1:
            cont_var = False

    graph_obj.particle_boarder = deepcopy(temp_boarder)


def precipitate_finder(graph_obj, i):

    indices, distances, n = graph_obj.find_nearest(i, graph_obj.vertices[i].n())

    graph_obj.vertices[i].flag_1 = True

    for x in range(0, n):

        if not graph_obj.vertices[indices[x]].h_index == 3:

            if not graph_obj.vertices[indices[x]].h_index == 6:
                graph_obj.vertices[i].flag_2 = True

        else:

            if not graph_obj.vertices[indices[x]].flag_1:
                precipitate_finder(graph_obj, indices[x])


