
import numpy as np
import utils
import params
import graph
from copy import deepcopy
import logging

# Instantiate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def base_stat_score(graph_obj, i, get_individual_predictions=False, scaled=True):
    alpha = graph_obj.get_alpha_angles(i)
    alpha_min = min(alpha)
    alpha_max = max(alpha)
    theta = graph_obj.get_theta_angles(i)
    if theta:
        theta_min = min(theta)
        theta_max = max(theta)
        theta_avg = sum(theta) / len(theta)
    else:
        theta_min = 0
        theta_max = 0
        theta_avg = 0

    normalized_peak_gamma = graph_obj.vertices[i].normalized_peak_gamma
    normalized_avg_gamma = graph_obj.vertices[i].normalized_avg_gamma

    values = [alpha_min, alpha_max, theta_min, theta_max, theta_avg, normalized_peak_gamma, normalized_avg_gamma]
    parameters, covar_matrices, reduced_model_covar_matrices, covar_determinants, reduced_model_covar_determinants,\
        inverse_covar_matrices, inverse_reduced_model_covar_matrices = params.produce_params(calc=False, scaled_model=scaled)

    probs = []

    alpha_min_probs = []
    alpha_max_probs = []
    theta_min_probs = []
    theta_max_probs = []
    theta_avg_probs = []
    norm_peak_gamma_probs = []
    norm_avg_gamma_probs = []

    for j, element in enumerate(['Cu', 'Si_1', 'Si_2', 'Al_1', 'Al_2', 'Mg_1', 'Mg_2']):

        if theta:
            means = [a[0] for a in parameters[j]]
            probs.append(utils.multivariate_normal_dist(values, means, covar_determinants[j], inverse_covar_matrices[j]))

        else:
            means = []
            reduced_values = []
            for k in [0, 1, 5, 6]:
                means.append(parameters[j][k][0])
                reduced_values.append(values[k])
            probs.append(utils.multivariate_normal_dist(reduced_values, means, reduced_model_covar_determinants[j], inverse_reduced_model_covar_matrices[j]))

        if get_individual_predictions:

            alpha_min_probs.append(utils.normal_dist(values[0], parameters[j][0][0], parameters[j][0][1]))
            alpha_max_probs.append(utils.normal_dist(values[1], parameters[j][1][0], parameters[j][1][1]))
            theta_min_probs.append(utils.normal_dist(values[2], parameters[j][2][0], parameters[j][2][1]))
            theta_max_probs.append(utils.normal_dist(values[3], parameters[j][3][0], parameters[j][3][1]))
            theta_avg_probs.append(utils.normal_dist(values[4], parameters[j][4][0], parameters[j][4][1]))
            norm_peak_gamma_probs.append(utils.normal_dist(values[5], parameters[j][5][0], parameters[j][5][1]))
            norm_avg_gamma_probs.append(utils.normal_dist(values[6], parameters[j][6][0], parameters[j][6][1]))

    probs = utils.normalize_list(probs, 1)
    sum_probs = [probs[1] + probs[2], probs[0], 0, probs[3] + probs[4], 0, probs[5] + probs[6], 0]

    if get_individual_predictions:

        alpha_min_probs = utils.normalize_list(alpha_min_probs, 1)
        alpha_max_probs = utils.normalize_list(alpha_max_probs, 1)
        theta_min_probs = utils.normalize_list(theta_min_probs, 1)
        theta_max_probs = utils.normalize_list(theta_max_probs, 1)
        theta_avg_probs = utils.normalize_list(theta_avg_probs, 1)
        norm_peak_gamma_probs = utils.normalize_list(norm_peak_gamma_probs, 1)
        norm_avg_gamma_probs = utils.normalize_list(norm_avg_gamma_probs, 1)

        alpha_min_probs = [alpha_min_probs[1] + alpha_min_probs[2], alpha_min_probs[0], 0, alpha_min_probs[3] + alpha_min_probs[4], 0, alpha_min_probs[5] + alpha_min_probs[6], 0]
        alpha_max_probs = [alpha_max_probs[1] + alpha_max_probs[2], alpha_max_probs[0], 0, alpha_max_probs[3] + alpha_max_probs[4], 0, alpha_max_probs[5] + alpha_max_probs[6], 0]
        theta_min_probs = [theta_min_probs[1] + theta_min_probs[2], theta_min_probs[0], 0, theta_min_probs[3] + theta_min_probs[4], 0, theta_min_probs[5] + theta_min_probs[6], 0]
        theta_max_probs = [theta_max_probs[1] + theta_max_probs[2], theta_max_probs[0], 0, theta_max_probs[3] + theta_max_probs[4], 0, theta_max_probs[5] + theta_max_probs[6], 0]
        theta_avg_probs = [theta_avg_probs[1] + theta_avg_probs[2], theta_avg_probs[0], 0, theta_avg_probs[3] + theta_avg_probs[4], 0, theta_avg_probs[5] + theta_avg_probs[6], 0]
        norm_peak_gamma_probs = [norm_peak_gamma_probs[1] + norm_peak_gamma_probs[2], norm_peak_gamma_probs[0], 0, norm_peak_gamma_probs[3] + norm_peak_gamma_probs[4], 0, norm_peak_gamma_probs[5] + norm_peak_gamma_probs[6], 0]
        norm_avg_gamma_probs = [norm_avg_gamma_probs[1] + norm_avg_gamma_probs[2], norm_avg_gamma_probs[0], 0, norm_avg_gamma_probs[3] + norm_avg_gamma_probs[4], 0, norm_avg_gamma_probs[5] + norm_avg_gamma_probs[6], 0]

        product_probs = [a * b * c * d * e * f * g for a, b, c, d, e, f, g in zip(alpha_min_probs, alpha_max_probs, theta_min_probs, theta_max_probs, theta_avg_probs, norm_peak_gamma_probs, norm_avg_gamma_probs)]
        product_probs = utils.normalize_list(product_probs, 1)
        weighted_product_probs = [a * b * c * d * e * f * g for a, b, c, d, e, f, g in zip(alpha_min_probs, alpha_max_probs, theta_min_probs, theta_max_probs, theta_avg_probs, norm_peak_gamma_probs, norm_avg_gamma_probs)]
        weighted_product_probs = utils.normalize_list(weighted_product_probs, 1)

        # print('\n{} :'.format(i))
        # print('values: {}'.format(values))
        # print('alpha min probs: {}'.format(alpha_min_probs))
        # print('alpha max probs: {}'.format(alpha_max_probs))
        # print('theta min probs: {}'.format(theta_min_probs))
        # print('theta max probs: {}'.format(theta_max_probs))
        # print('theta avg probs: {}'.format(theta_avg_probs))
        # print('peak gamma probs: {}'.format(norm_peak_gamma_probs))
        # print('avg gamma probs: {}'.format(norm_avg_gamma_probs))
        # print('\n')
        # print('Product probs: {}'.format(product_probs))
        # print('Model: {}'.format(sum_probs))
        # print('\n')

        return [sum_probs, alpha_min_probs, alpha_max_probs, theta_min_probs, theta_max_probs, theta_avg_probs, norm_peak_gamma_probs, norm_avg_gamma_probs, product_probs, weighted_product_probs]

    else:

        return sum_probs


def base_angle_score(graph_obj, i, apply=True):

    alpha = graph_obj.get_alpha_angles(i)

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

        # print('alpha: {}\nmax: {}, min: {}\ncf_min: {}\ncf_max: {}\ncf: {}\nprobs: {}\nsum_probs: {}\n\n'.format(alpha, max(alpha), min(alpha), cf_min, cf_max, cf, probs, sum_probs))

        return sum_probs

    else:

        return max(alpha), min(alpha)


def naive_determine_z(graph_obj, i, level):
    graph_obj.vertices[i].flag_1 = True
    graph_obj.set_level(i, level)

    next_level = 0
    if level == 0:
        next_level = 1
    elif level == 1:
        next_level = 0

    for j in graph_obj.vertices[i].true_partner_indices:
        if not graph_obj.vertices[j].flag_1:
            if graph_obj.vertices[j].is_edge_column:
                graph_obj.set_level(j, next_level)
            else:
                naive_determine_z(graph_obj, j, next_level)


def revise_z(graph_obj):
    for vertex in graph_obj.vertices:
        if not vertex.is_edge_column and not vertex.set_by_user:
            agree = 0
            disagree = 0
            for true_partner in vertex.true_partner_indices:
                if graph_obj.vertices[true_partner].level == vertex.level:
                    disagree += 1
                else:
                    agree += 1
            if disagree > agree:
                vertex.level = vertex.anti_level()


def determine_z_heights(graph_obj, i, level):
    graph_obj.reset_all_flags()
    graph_obj.sort_all_subsets_by_distance()
    determine_matrix_z_heights(graph_obj, i, level)

    for vertex in graph_obj.vertices:
        if vertex.flag_2:
            j = vertex.i
            break
    else:
        j = -1

    if not j == -1:
        determine_particle_z_heights(graph_obj, j, graph_obj.vertices[j].level)

    graph_obj.reset_all_flags()


def determine_matrix_z_heights(graph_obj, i, level):
    if graph_obj.vertices[i].is_in_precipitate:
        graph_obj.vertices[i].flag_1 = True
        graph_obj.vertices[i].flag_2 = True
        graph_obj.set_level(i, level)

    else:
        graph_obj.vertices[i].flag_1 = True
        graph_obj.set_level(i, level)

        next_level = 0
        if level == 0:
            next_level = 1
        elif level == 1:
            next_level = 0

        for j in graph_obj.vertices[i].true_partner_indices:
            if not graph_obj.vertices[j].flag_1:
                if graph_obj.vertices[j].is_edge_column:
                    graph_obj.set_level(j, next_level)
                else:
                    determine_matrix_z_heights(graph_obj, j, next_level)


def determine_particle_z_heights(graph_obj, i, level):
    graph_obj.vertices[i].flag_1 = True
    graph_obj.set_level(i, level)

    next_level = 0
    if level == 0:
        next_level = 1
    elif level == 1:
        next_level = 0

    for j in graph_obj.vertices[i].true_partner_indices:
        if not graph_obj.vertices[j].flag_1:
            if graph_obj.vertices[j].is_edge_column:
                graph_obj.set_level(j, next_level)
            else:
                determine_particle_z_heights(graph_obj, j, next_level)


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


def experimental_remove_intersections(graph_obj):
    intersections = graph_obj.find_intersections()

    for intersection in intersections:

        edge_1 = (intersection[0], intersection[1])
        edge_2 = (intersection[2], intersection[3])

        if graph_obj.vertices[edge_1[0]].partner_query(edge_1[1]):
            if edge_2[0] in graph_obj.vertices[edge_1[0]].anti_neighbourhood:
                graph_obj.permute_j_k(edge_1[0], edge_1[1], edge_2[0])
            else:
                if not graph_obj.terminate_arc(edge_1[0], edge_1[1]):
                    logger.warning('Could not remove {} {}'.format(edge_1[0], edge_1[1]))

        if graph_obj.vertices[edge_2[0]].partner_query(edge_2[1]):
            if edge_1[0] in graph_obj.vertices[edge_2[0]].anti_neighbourhood:
                graph_obj.permute_j_k(edge_2[0], edge_2[1], edge_1[0])
            else:
                if not graph_obj.terminate_arc(edge_2[0], edge_2[1]):
                    logger.warning('Could not remove {} {}'.format(edge_2[0], edge_2[1]))


