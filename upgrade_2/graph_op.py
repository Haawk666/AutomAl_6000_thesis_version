
import numpy as np
import utils
from copy import deepcopy


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

