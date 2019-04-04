
import numpy as np
import utils


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

