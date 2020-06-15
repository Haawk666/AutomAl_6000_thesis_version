# Internal imports:
import statistics
import utils
import untangling_2
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


def map_districts(graph_obj, method='matrix'):
    if method == 'matrix':
        time_1 = time.time()
        matrix = np.zeros([graph_obj.order, graph_obj.order], dtype=float)
        for i in range(0, len(graph_obj.vertices)):
            for j in range(i, len(graph_obj.vertices)):
                if not i == j:
                    dist = graph_obj.get_projected_separation(i, j)
                    if graph_obj.vertices[i].void or graph_obj.vertices[j].void:
                        dist = 10000000000
                    matrix[j, i] = dist
                    matrix[i, j] = dist
        time_2 = time.time()
        for vertex in graph_obj.vertices:
            if not vertex.void:
                vertex.projected_separation_district = np.argsort(matrix[vertex.i, :])[1:graph_obj.district_size + 1].tolist()
                vertex.district = copy.deepcopy(vertex.projected_separation_district)
            else:
                vertex.district = []
                vertex.projected_separation_district = []
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


def zeta_analysis(graph_obj, starting_index, starting_zeta=0, method='district', use_n=False):
    logger.info('Starting zeta analysis')
    time_1 = time.time()
    votes = [0] * graph_obj.order
    if starting_zeta == 0:
        votes[starting_index] = 1
    else:
        votes[starting_index] = -1
    counter = 0
    cont = True
    while cont:
        for vertex in graph_obj.vertices:
            if not vertex.void:
                if use_n:
                    n = vertex.n
                else:
                    n = 3
                if method == 'district':
                    candidates = vertex.district[0:n]
                elif method == 'separation':
                    candidates = vertex.projected_separation_district[0:n]
                elif method == 'partners':
                    candidates = list(vertex.partners)
                elif method == 'out_neighbours':
                    candidates = list(vertex.out_neighbourhood)
                else:
                    candidates = vertex.district[0:n]
                if vertex.is_edge_column:
                    for citizen in candidates:
                        votes[citizen] -= 0.01 * votes[vertex.i]
                else:
                    for citizen in candidates:
                        votes[citizen] -= 0.05 * votes[vertex.i]
        counter += 1
        if counter > 1000:
            cont = False

    for vertex in graph_obj.vertices:
        if votes[vertex.i] > 0:
            vertex.zeta = 0
        else:
            vertex.zeta = 1

    graph_obj.build_local_zeta_maps()
    graph_obj.build_local_maps()

    time_2 = time.time()
    logger.info('Zeta analysis completed in {} seconds'.format(time_2 - time_1))


def symmetry_characterization(graph_obj, im_width, im_height, starting_index, separation_threshold=320):
    logger.info('Running symmetry characterization')
    time_1 = time.time()
    graph_obj.build_local_maps()
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
    graph_obj.build_local_maps()
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
    graph_obj.build_local_maps()
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
    graph_obj.build_local_maps()
    time_3 = time.time()
    logger.info('Performed arc intersection denial in {} seconds'.format(time_3 - time_1))


def apply_alpha_model(graph_obj, model=None, alpha_selection_type='zeta'):
    if model is None:
        this_model = statistics.VertexDataManager.load(graph_obj.active_model)
    else:
        this_model = model
    for vertex in graph_obj.vertices:
        vertex.alpha_angles = graph_obj.get_alpha_angles(vertex.i, selection_type=alpha_selection_type)
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
    graph_obj.build_local_maps()


def apply_composite_model(graph_obj, model=None, alpha_selection_type='zeta'):
    if model is None:
        this_model = statistics.VertexDataManager.load(graph_obj.active_model)
    else:
        this_model = model
    for vertex in graph_obj.vertices:
        vertex.alpha_angles = graph_obj.get_alpha_angles(vertex.i, selection_type=alpha_selection_type)
        if vertex.alpha_angles is not None and not len(vertex.alpha_angles) == 0:
            vertex.alpha_max = max(vertex.alpha_angles)
            vertex.alpha_min = min(vertex.alpha_angles)
        else:
            vertex.alpha_max = 0
            vertex.alpha_min = 0
        vertex.theta_angles = graph_obj.get_theta_angles(vertex.i, selection_type='selective')
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
    for vertex in graph_obj.vertices:
        if not vertex.is_edge_column and not vertex.void and not vertex.is_set_by_user:
            dict_ = {
                'alpha_max': vertex.alpha_max,
                'alpha_min': vertex.alpha_min,
                'theta_max': vertex.theta_max,
                'theta_min': vertex.theta_min,
                'theta_angle_mean': vertex.theta_angle_mean,
                'normalized_peak_gamma': vertex.normalized_peak_gamma,
                'normalized_avg_gamma': vertex.normalized_peak_gamma
            }
            keys_to_pop = []
            for key, value in dict_.items():
                if value == 0:
                    keys_to_pop.append(key)
            for key in keys_to_pop:
                dict_.pop(key)
            vertex.advanced_probability_vector = this_model.calc_prediction(dict_)
            vertex.advanced_probability_vector['Un_1'] = 0.0
            vertex.determine_species_from_probability_vector()
    graph_obj.build_local_maps()
    graph_obj.build_local_zeta_maps()


def untangle(graph_obj, ui_obj=None):

    # Untangling:
    for vertex in graph_obj.vertices:
        if not vertex.is_edge_column and not vertex.void:
            if len(vertex.out_semi_partners) > 0:
                district_copy = copy.deepcopy(vertex.district)
                for citizen in district_copy:
                    if citizen in vertex.out_semi_partners:
                        sub_graph = graph_obj.get_arc_centered_subgraph(vertex.i, citizen)
                        untangling_2.determine_sub_graph_class([sub_graph])
                        untangling_2.determine_sub_graph_configuration(graph_obj, [sub_graph])
                        untangling_2.weak_resolve(graph_obj, [sub_graph], ui_obj=ui_obj)

    for vertex in graph_obj.vertices:
        if not vertex.is_edge_column and not vertex.void:
            if len(vertex.out_semi_partners) > 0:
                district_copy = copy.deepcopy(vertex.district)
                for citizen in district_copy:
                    if citizen in vertex.out_semi_partners:
                        sub_graph = graph_obj.get_arc_centered_subgraph(vertex.i, citizen)
                        untangling_2.determine_sub_graph_class([sub_graph])
                        untangling_2.determine_sub_graph_configuration(graph_obj, [sub_graph])
                        untangling_2.strong_resolve(graph_obj, [sub_graph], ui_obj=ui_obj)





