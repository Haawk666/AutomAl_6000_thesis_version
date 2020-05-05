
import numpy as np
from copy import deepcopy
import logging
# Instantiate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


