
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


def set_levels(graph_obj, i, report):

    report('        Setting levels of matrix columns...', force=True)
    set_matrix_levels(graph_obj, i)
    report('        Matrix levels set.', force=True)
    report('        Setting levels of particle columns...', force=True)
    set_particle_levels(graph_obj)
    report('        Particle levels set.', force=True)


def set_matrix_levels(graph_obj, i):

    pass


def set_particle_levels(graph_obj):

    pass

