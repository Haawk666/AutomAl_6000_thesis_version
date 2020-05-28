# Internal imports:
import statistics
# External imports:
import numpy as np
import time
import copy
import logging
# Instantiate logger:
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def find_edge_columns(graph_obj, im_width, im_height):
    logger.info('Detecting edge columns')
    time_1 = time.time()
    for vertex in graph_obj.vertices:
        x_coor = vertex.im_coor_x
        y_coor = vertex.im_coor_y
        margin = 3 * vertex.r
        if x_coor < margin or x_coor > im_width - margin - 1 or y_coor < margin or y_coor > im_height - margin - 1:
            graph_obj.vertices[vertex.i].is_edge_column = True
        else:
            graph_obj.vertices[vertex.i].is_edge_column = False
    time_2 = time.time()
    logger.info('Found edge columns in {} seconds'.format(time_2 - time_1))


def map_districts(graph_obj, district_size=8, method='matrix'):
    if method == 'matrix':
        time_1 = time.time()
        matrix = np.zeros([graph_obj.order, graph_obj.order], dtype=float)
        for i in range(0, len(graph_obj.vertices)):
            for j in range(i, len(graph_obj.vertices)):
                if not i == j:
                    dist = graph_obj.get_projected_separation(i, j)
                    matrix[j, i] = dist
                    matrix[i, j] = dist
        time_2 = time.time()
        for vertex in graph_obj.vertices:
            vertex.district = np.argsort(matrix[vertex.i, :])[1:district_size + 1].tolist()
        time_3 = time.time()
        summary_string = 'Districts mapped in {} seconds with matrix method.\n'.format(time_3 - time_1)
        summary_string += '    Distance calculations took {} seconds.\n'.format(time_2 - time_1)
        summary_string += '    Sorting took {} seconds.'.format(time_3 - time_2)
        logger.info(summary_string)
        graph_obj.separation_matrix = matrix


def particle_detection(graph_obj):
    for vertex in graph_obj.vertices:
        if vertex.void:
            vertex.is_in_precipitate = False
        else:
            if vertex.atomic_species == 'Al':
                num_foreign_species = 0
                for neighbour in vertex.neighbourhood:
                    if not graph_obj.vertices[neighbour].atomic_species == 'Al':
                        num_foreign_species += 1
                    if num_foreign_species == 2:
                        vertex.is_in_precipitate = True
                        break
                else:
                    vertex.is_in_precipitate = False
            else:
                vertex.is_in_precipitate = True


def zeta_analysis(graph_obj, starting_index, starting_zeta=0):
    logger.info('Starting zeta analysis')
    time_1 = time.time()
    votes = [0] * graph_obj.order
    if starting_zeta == 0:
        votes[starting_index] = 2
    else:
        votes[starting_index] = -2

    counter = 0
    cont = True
    while cont:
        for vertex in graph_obj.vertices:
            if not vertex.void:
                for out_neighbour in vertex.out_neighbourhood:
                    votes[out_neighbour] -= votes[vertex.i]
        counter += 1
        if counter > 1000:
            cont = False

    for vertex in graph_obj.vertices:
        if votes[vertex.i] > 0:
            vertex.zeta = 0
        else:
            vertex.zeta = 1

    for vertex in graph_obj.vertices:
        sep_1 = []
        sep_2 = []
        for citizen in vertex.district:
            if graph_obj.vertices[citizen].zeta == vertex.zeta:
                if len(sep_2) < 3:
                    sep_2.append(graph_obj.separation_matrix[vertex.i, citizen])
            else:
                if len(sep_1) < 3:
                    sep_1.append(graph_obj.separation_matrix[vertex.i, citizen])
        sep_1 = sum(sep_1) / len(sep_1)
        sep_2 = sum(sep_2) / len(sep_2)

        if sep_2 < sep_1:
            vertex.zeta = vertex.anti_zeta()
            logger.info('Altering zeta of vertex {}'.format(vertex.i))
    graph_obj.build_maps()

    time_2 = time.time()

    logger.info('Zeta analysis completed in {} seconds'.format(time_2 - time_1))


def symmetry_characterization(graph_obj, im_width, im_height, starting_index, separation_threshold=320):
    logger.info('Running symmetry characterization')
    time_1 = time.time()
    graph_obj.build_maps()
    for vertex in graph_obj.vertices:
        if vertex.is_edge_column:
            graph_obj.set_species(vertex.i, 'Al_1')
    zeta_analysis(graph_obj, starting_index)
    for vertex in graph_obj.vertices:
        sep_1 = []
        sep_2 = []
        for citizen in vertex.district:
            if graph_obj.vertices[citizen].zeta == vertex.zeta:
                if len(sep_2) < 3:
                    sep_2.append(graph_obj.separation_matrix[vertex.i, citizen])
            else:
                if len(sep_1) < 3:
                    sep_1.append(graph_obj.separation_matrix[vertex.i, citizen])
        sep_1 = sum(sep_1) / len(sep_1)
        sep_2 = sum(sep_2) / len(sep_2)

        if sep_2 < sep_1:
            vertex.zeta = vertex.anti_zeta()
            logger.info('Altering zeta of vertex {}'.format(vertex.i))
    graph_obj.build_maps()
    time_2 = time.time()
    logger.info('Completed symmetry characterization in {} seconds'.format(time_2 - time_1))


def arc_intersection_denial(graph_obj):
    time_1 = time.time()
    intersections = graph_obj.find_intersections()
    for intersection in intersections:
        if not graph_obj.vertices[intersection[0]].partner_query(intersection[1]):
            if not graph_obj.terminate_arc(intersection[0], intersection[1]):
                if not graph_obj.terminate_arc(intersection[2], intersection[3]):
                    logger.info('Could not remove intersection {}'.format(intersection))
        elif not graph_obj.vertices[intersection[2]].partner_query(intersection[3]):
            if not graph_obj.terminate_arc(intersection[2], intersection[3]):
                if not graph_obj.terminate_arc(intersection[0], intersection[1]):
                    logger.info('Could not remove intersection {}'.format(intersection))
        else:
            if not graph_obj.terminate_arc(intersection[2], intersection[3]):
                if not graph_obj.terminate_arc(intersection[0], intersection[1]):
                    logger.info('Could not remove intersection {}'.format(intersection))
    graph_obj.build_maps()
    intersections = graph_obj.find_intersections()
    for intersection in intersections:
        if not graph_obj.vertices[intersection[0]].partner_query(intersection[1]):
            if not graph_obj.terminate_arc(intersection[0], intersection[1]):
                if not graph_obj.terminate_arc(intersection[2], intersection[3]):
                    logger.info('Could not remove intersection {}'.format(intersection))
        elif not graph_obj.vertices[intersection[2]].partner_query(intersection[3]):
            if not graph_obj.terminate_arc(intersection[2], intersection[3]):
                if not graph_obj.terminate_arc(intersection[0], intersection[1]):
                    logger.info('Could not remove intersection {}'.format(intersection))
        else:
            if not graph_obj.terminate_arc(intersection[2], intersection[3]):
                if not graph_obj.terminate_arc(intersection[0], intersection[1]):
                    logger.info('Could not remove intersection {}'.format(intersection))
    graph_obj.build_maps()
    time_3 = time.time()
    logger.info('Performed arc intersection denial in {} seconds'.format(time_3 - time_1))


def apply_alpha_model(graph_obj, model=None, alpha_selection_type='zeta'):
    if model is None:
        this_model = statistics.VertexDataManager.load(graph_obj.active_model)
    else:
        this_model = model
    for vertex in graph_obj.vertices:
        vertex.alpha_angles = graph_obj.get_alpha_angles(vertex.i, selection_type='zeta')
        if vertex.alpha_angles is not None and not len(vertex.alpha_angles) == 0:
            vertex.alpha_max = max(vertex.alpha_angles)
            vertex.alpha_min = min(vertex.alpha_angles)
        else:
            vertex.alpha_max = 0
            vertex.alpha_min = 0
    for vertex in graph_obj.vertices:
        if not vertex.is_edge_column and not vertex.void and not vertex.is_set_by_user:
            vertex.advanced_probability_vector = this_model.calc_prediction({
                'alpha_max': vertex.alpha_max,
                'alpha_min': vertex.alpha_min
            })
            vertex.advanced_probability_vector['Un_1'] = 0.0
            vertex.determine_species_from_probability_vector()
    graph_obj.build_maps()




