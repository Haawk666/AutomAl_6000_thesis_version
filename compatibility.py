"""Module to handle forwards compatibility between versions.

.. note::

    Only forwards compatibility is maintained. Opening a project file that was saved in a more recent version is
    generally not guarantied to be possible, but opening old project files in newer version should always be possible.

"""
# Internal imports
import graph_2
import statistics
# External imports
import logging
# Instantiate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def convert(obj, old_version, version):
    """Return project instance in a compatible state.

    :param obj: The core.SuchSoftware instance to be upgraded.
    :param old_version: The version that 'obj' was saved with.
    :param version: The version that 'obj' should be converted to.

    :type obj: core.SuchSoftware() instance
    :type old_version: list(<int>)
    :type version: list(<int>)

    :returns project instance in a compatible state.
    :rtype core.SuchSoftware() instance

    """

    # Return obj in a compatible state
    # Return None if not possible!

    if old_version == [0, 0, 0]:
        # Set new attributes
        obj.starting_index = None
        obj.graph.avg_species_confidence = 0.0
        obj.graph.avg_symmetry_confidence = 0.0
        obj.graph.avg_level_confidence = 0.0
        for vertex in obj.graph.vertices:
            vertex.level_confidence = 0.0
            vertex.symmetry_confidence = 0.0
            vertex.level_vector = [1 / 2, 1 / 2]
            vertex.symmetry_vector = [1/3, 1/3, 1/3]
            vertex.flag_5 = False
            vertex.flag_6 = False
            vertex.flag_7 = False
            vertex.flag_8 = False
            vertex.ad_hoc_list_1 = []
            vertex.ad_hoc_list_1 = []
            vertex.ad_hoc_value_1 = 0
            vertex.ad_hoc_value_2 = 0
            vertex.central_angle_variance = 0.0
            vertex.spatial_coor_x = obj.scale * vertex.real_coor_x
            vertex.spatial_coor_y = obj.scale * vertex.real_coor_y
            vertex.normalized_peak_gamma = vertex.peak_gamma
            vertex.normalized_avg_gamma = vertex.avg_gamma
            vertex.friendly_indices = []
            vertex.anti_partner_indices = []
            vertex.true_partner_indices = []
            vertex.unfriendly_partner_indices = []
            vertex.true_anti_partner_indices = []
            vertex.anti_friend_indices = []
            vertex.friend_indices = []
            vertex.outsider_indices = []
        obj.graph.avg_central_variance = 0.0
        obj.graph.meshes = []
        obj.graph.mesh_indices = []
        obj.graph.scale = obj.scale
        # Updated graph module:
        old_graph = obj.graph
        new_graph = graph_2.AtomicGraph(obj.scale)
        for vertex in old_graph.vertices:
            new_vertex = graph_2.Vertex(vertex.i, vertex.real_coor_x, vertex.real_coor_y, vertex.r, vertex.peak_gamma,
                                        vertex.avg_gamma, obj.scale, level=vertex.level, species_index=vertex.h_index)
            for i, citizen in enumerate(vertex.neighbour_indices):
                new_vertex.district.append(citizen)
            new_vertex.is_in_precipitate = vertex.is_in_precipitate
            new_vertex.is_edge_column = vertex.is_edge_column
            new_vertex.is_set_by_user = vertex.set_by_user
            new_vertex.show_in_overlay = vertex.show_in_overlay
            new_vertex.flag_1 = False
            new_vertex.flag_2 = False
            new_vertex.flag_3 = False
            new_vertex.flag_4 = False
            new_vertex.flag_5 = False
            new_vertex.flag_6 = False
            new_vertex.flag_7 = False
            new_vertex.flag_8 = False
            new_vertex.flag_9 = False

            new_vertex.probability_vector = vertex.prob_vector.tolist()

            new_graph.add_vertex(new_vertex)
        new_graph.refresh_graph()
        obj.graph = new_graph
        obj.active_model = statistics.DataManager.load('default_model')
        obj.graph.active_model = obj.active_model
        fresh_obj = obj

    elif old_version == [0, 0, 1]:
        # Set new attributes
        obj.graph.avg_species_confidence = 0.0
        obj.graph.avg_symmetry_confidence = 0.0
        obj.graph.avg_level_confidence = 0.0
        for vertex in obj.graph.vertices:
            vertex.level_confidence = 0.0
            vertex.symmetry_confidence = 0.0
            vertex.level_vector = [1 / 2, 1 / 2]
            vertex.symmetry_vector = [1/3, 1/3, 1/3]
            vertex.flag_5 = False
            vertex.flag_6 = False
            vertex.flag_7 = False
            vertex.flag_8 = False
            vertex.ad_hoc_list_1 = []
            vertex.ad_hoc_list_1 = []
            vertex.ad_hoc_value_1 = 0
            vertex.ad_hoc_value_2 = 0
            vertex.central_angle_variance = 0.0
            vertex.spatial_coor_x = obj.scale * vertex.real_coor_x
            vertex.spatial_coor_y = obj.scale * vertex.real_coor_y
            vertex.normalized_peak_gamma = vertex.peak_gamma
            vertex.normalized_avg_gamma = vertex.avg_gamma
            vertex.friendly_indices = []
            vertex.anti_partner_indices = []
            vertex.true_partner_indices = []
            vertex.unfriendly_partner_indices = []
            vertex.true_anti_partner_indices = []
            vertex.anti_friend_indices = []
            vertex.friend_indices = []
            vertex.outsider_indices = []
        obj.graph.avg_central_variance = 0.0
        obj.graph.meshes = []
        obj.graph.mesh_indices = []
        obj.graph.scale = obj.scale
        # Updated graph module:
        old_graph = obj.graph
        new_graph = graph_2.AtomicGraph(obj.scale)
        for vertex in old_graph.vertices:
            new_vertex = graph_2.Vertex(vertex.i, vertex.real_coor_x, vertex.real_coor_y, vertex.r, vertex.peak_gamma,
                                        vertex.avg_gamma, obj.scale, level=vertex.level, species_index=vertex.h_index)
            for i, citizen in enumerate(vertex.neighbour_indices):
                new_vertex.district.append(citizen)
            new_vertex.is_in_precipitate = vertex.is_in_precipitate
            new_vertex.is_edge_column = vertex.is_edge_column
            new_vertex.is_set_by_user = vertex.set_by_user
            new_vertex.show_in_overlay = vertex.show_in_overlay
            new_vertex.flag_1 = False
            new_vertex.flag_2 = False
            new_vertex.flag_3 = False
            new_vertex.flag_4 = False
            new_vertex.flag_5 = False
            new_vertex.flag_6 = False
            new_vertex.flag_7 = False
            new_vertex.flag_8 = False
            new_vertex.flag_9 = False

            new_vertex.probability_vector = vertex.prob_vector.tolist()

            new_graph.add_vertex(new_vertex)
        new_graph.refresh_graph()
        obj.graph = new_graph
        obj.active_model = statistics.DataManager.load('default_model')
        obj.graph.active_model = obj.active_model
        fresh_obj = obj

    elif old_version == [0, 0, 2]:
        obj.graph.avg_species_confidence = 0.0
        obj.graph.avg_symmetry_confidence = 0.0
        obj.graph.avg_level_confidence = 0.0
        for vertex in obj.graph.vertices:
            vertex.level_confidence = 0.0
            vertex.symmetry_confidence = 0.0
            vertex.level_vector = [1 / 2, 1 / 2]
            vertex.central_angle_variance = 0.0
            vertex.spatial_coor_x = obj.scale * vertex.real_coor_x
            vertex.spatial_coor_y = obj.scale * vertex.real_coor_y
            vertex.normalized_peak_gamma = vertex.peak_gamma
            vertex.normalized_avg_gamma = vertex.avg_gamma
            vertex.friendly_indices = []
            vertex.anti_partner_indices = []
            vertex.true_partner_indices = []
            vertex.unfriendly_partner_indices = []
            vertex.true_anti_partner_indices = []
            vertex.anti_friend_indices = []
            vertex.friend_indices = []
            vertex.outsider_indices = []
        obj.graph.avg_central_variance = 0.0
        obj.graph.meshes = []
        obj.graph.mesh_indices = []
        obj.graph.scale = obj.scale
        # Updated graph module:
        old_graph = obj.graph
        new_graph = graph_2.AtomicGraph(obj.scale)
        for vertex in old_graph.vertices:
            new_vertex = graph_2.Vertex(vertex.i, vertex.real_coor_x, vertex.real_coor_y, vertex.r, vertex.peak_gamma,
                                        vertex.avg_gamma, obj.scale, level=vertex.level, species_index=vertex.h_index)
            for i, citizen in enumerate(vertex.neighbour_indices):
                new_vertex.district.append(citizen)
            new_vertex.is_in_precipitate = vertex.is_in_precipitate
            new_vertex.is_edge_column = vertex.is_edge_column
            new_vertex.is_set_by_user = vertex.set_by_user
            new_vertex.show_in_overlay = vertex.show_in_overlay
            new_vertex.flag_1 = False
            new_vertex.flag_2 = False
            new_vertex.flag_3 = False
            new_vertex.flag_4 = False
            new_vertex.flag_5 = False
            new_vertex.flag_6 = False
            new_vertex.flag_7 = False
            new_vertex.flag_8 = False
            new_vertex.flag_9 = False

            new_vertex.probability_vector = vertex.prob_vector.tolist()

            new_graph.add_vertex(new_vertex)
        new_graph.refresh_graph()
        obj.graph = new_graph
        obj.active_model = statistics.DataManager.load('default_model')
        obj.graph.active_model = obj.active_model
        fresh_obj = obj

    elif old_version == [0, 0, 3]:
        for vertex in obj.graph.vertices:
            vertex.central_angle_variance = 0.0
            vertex.spatial_coor_x = obj.scale * vertex.real_coor_x
            vertex.spatial_coor_y = obj.scale * vertex.real_coor_y
            vertex.normalized_peak_gamma = vertex.peak_gamma
            vertex.normalized_avg_gamma = vertex.avg_gamma
            vertex.friendly_indices = []
            vertex.anti_partner_indices = []
            vertex.true_partner_indices = []
            vertex.unfriendly_partner_indices = []
            vertex.true_anti_partner_indices = []
            vertex.anti_friend_indices = []
            vertex.friend_indices = []
            vertex.outsider_indices = []
        obj.graph.avg_central_variance = 0.0
        obj.graph.meshes = []
        obj.graph.mesh_indices = []
        obj.graph.scale = obj.scale
        # Updated graph module:
        old_graph = obj.graph
        new_graph = graph_2.AtomicGraph(obj.scale)
        for vertex in old_graph.vertices:
            new_vertex = graph_2.Vertex(vertex.i, vertex.real_coor_x, vertex.real_coor_y, vertex.r, vertex.peak_gamma,
                                        vertex.avg_gamma, obj.scale, level=vertex.level, species_index=vertex.h_index)
            for i, citizen in enumerate(vertex.neighbour_indices):
                new_vertex.district.append(citizen)
            new_vertex.is_in_precipitate = vertex.is_in_precipitate
            new_vertex.is_edge_column = vertex.is_edge_column
            new_vertex.is_set_by_user = vertex.set_by_user
            new_vertex.show_in_overlay = vertex.show_in_overlay
            new_vertex.flag_1 = False
            new_vertex.flag_2 = False
            new_vertex.flag_3 = False
            new_vertex.flag_4 = False
            new_vertex.flag_5 = False
            new_vertex.flag_6 = False
            new_vertex.flag_7 = False
            new_vertex.flag_8 = False
            new_vertex.flag_9 = False

            new_vertex.probability_vector = vertex.prob_vector.tolist()

            new_graph.add_vertex(new_vertex)
        new_graph.refresh_graph()
        obj.graph = new_graph
        obj.active_model = statistics.DataManager.load('default_model')
        obj.graph.active_model = obj.active_model
        fresh_obj = obj

    elif old_version == [0, 0, 4]:
        for vertex in obj.graph.vertices:
            vertex.spatial_coor_x = obj.scale * vertex.real_coor_x
            vertex.spatial_coor_y = obj.scale * vertex.real_coor_y
            vertex.normalized_peak_gamma = vertex.peak_gamma
            vertex.normalized_avg_gamma = vertex.avg_gamma
            vertex.friendly_indices = []
            vertex.anti_partner_indices = []
            vertex.true_partner_indices = []
            vertex.unfriendly_partner_indices = []
            vertex.true_anti_partner_indices = []
            vertex.anti_friend_indices = []
            vertex.friend_indices = []
            vertex.outsider_indices = []
        obj.graph.meshes = []
        obj.graph.mesh_indices = []
        obj.graph.scale = obj.scale
        # Updated graph module:
        old_graph = obj.graph
        new_graph = graph_2.AtomicGraph(obj.scale)
        for vertex in old_graph.vertices:
            new_vertex = graph_2.Vertex(vertex.i, vertex.real_coor_x, vertex.real_coor_y, vertex.r, vertex.peak_gamma,
                                        vertex.avg_gamma, obj.scale, level=vertex.level, species_index=vertex.h_index)
            for i, citizen in enumerate(vertex.neighbour_indices):
                new_vertex.district.append(citizen)
            new_vertex.is_in_precipitate = vertex.is_in_precipitate
            new_vertex.is_edge_column = vertex.is_edge_column
            new_vertex.is_set_by_user = vertex.set_by_user
            new_vertex.show_in_overlay = vertex.show_in_overlay
            new_vertex.flag_1 = False
            new_vertex.flag_2 = False
            new_vertex.flag_3 = False
            new_vertex.flag_4 = False
            new_vertex.flag_5 = False
            new_vertex.flag_6 = False
            new_vertex.flag_7 = False
            new_vertex.flag_8 = False
            new_vertex.flag_9 = False

            new_vertex.probability_vector = vertex.prob_vector.tolist()

            new_graph.add_vertex(new_vertex)
        new_graph.refresh_graph()
        obj.graph = new_graph
        obj.active_model = statistics.DataManager.load('default_model')
        obj.graph.active_model = obj.active_model
        fresh_obj = obj

    elif old_version == [0, 0, 5]:
        for vertex in obj.graph.vertices:
            vertex.normalized_peak_gamma = vertex.peak_gamma
            vertex.normalized_avg_gamma = vertex.avg_gamma
            vertex.friendly_indices = []
            vertex.anti_partner_indices = []
            vertex.true_partner_indices = []
            vertex.unfriendly_partner_indices = []
            vertex.true_anti_partner_indices = []
            vertex.anti_friend_indices = []
            vertex.friend_indices = []
            vertex.outsider_indices = []
        obj.graph.meshes = []
        obj.graph.mesh_indices = []
        obj.graph.scale = obj.scale
        # Updated graph module:
        old_graph = obj.graph
        new_graph = graph_2.AtomicGraph(obj.scale)
        for vertex in old_graph.vertices:
            new_vertex = graph_2.Vertex(vertex.i, vertex.real_coor_x, vertex.real_coor_y, vertex.r, vertex.peak_gamma,
                                        vertex.avg_gamma, obj.scale, level=vertex.level, species_index=vertex.h_index)
            for i, citizen in enumerate(vertex.neighbour_indices):
                new_vertex.district.append(citizen)
            new_vertex.is_in_precipitate = vertex.is_in_precipitate
            new_vertex.is_edge_column = vertex.is_edge_column
            new_vertex.is_set_by_user = vertex.set_by_user
            new_vertex.show_in_overlay = vertex.show_in_overlay
            new_vertex.flag_1 = False
            new_vertex.flag_2 = False
            new_vertex.flag_3 = False
            new_vertex.flag_4 = False
            new_vertex.flag_5 = False
            new_vertex.flag_6 = False
            new_vertex.flag_7 = False
            new_vertex.flag_8 = False
            new_vertex.flag_9 = False

            new_vertex.probability_vector = vertex.prob_vector.tolist()

            new_graph.add_vertex(new_vertex)
        new_graph.refresh_graph()
        obj.graph = new_graph
        obj.active_model = statistics.DataManager.load('default_model')
        obj.graph.active_model = obj.active_model
        fresh_obj = obj

    elif old_version == [0, 0, 6]:
        for vertex in obj.graph.vertices:
            vertex.friendly_indices = []
            vertex.anti_partner_indices = []
            vertex.true_partner_indices = []
            vertex.unfriendly_partner_indices = []
            vertex.true_anti_partner_indices = []
            vertex.anti_friend_indices = []
            vertex.friend_indices = []
            vertex.outsider_indices = []
        obj.graph.meshes = []
        obj.graph.mesh_indices = []
        obj.graph.scale = obj.scale
        # Updated graph module:
        old_graph = obj.graph
        new_graph = graph_2.AtomicGraph(obj.scale)
        for vertex in old_graph.vertices:
            new_vertex = graph_2.Vertex(vertex.i, vertex.real_coor_x, vertex.real_coor_y, vertex.r, vertex.peak_gamma,
                                        vertex.avg_gamma, obj.scale, level=vertex.level, species_index=vertex.h_index)
            for i, citizen in enumerate(vertex.neighbour_indices):
                new_vertex.district.append(citizen)
            new_vertex.is_in_precipitate = vertex.is_in_precipitate
            new_vertex.is_edge_column = vertex.is_edge_column
            new_vertex.is_set_by_user = vertex.set_by_user
            new_vertex.show_in_overlay = vertex.show_in_overlay
            new_vertex.flag_1 = False
            new_vertex.flag_2 = False
            new_vertex.flag_3 = False
            new_vertex.flag_4 = False
            new_vertex.flag_5 = False
            new_vertex.flag_6 = False
            new_vertex.flag_7 = False
            new_vertex.flag_8 = False
            new_vertex.flag_9 = False

            new_vertex.probability_vector = vertex.prob_vector.tolist()

            new_graph.add_vertex(new_vertex)
        new_graph.refresh_graph()
        obj.graph = new_graph
        obj.active_model = statistics.DataManager.load('default_model')
        obj.graph.active_model = obj.active_model
        fresh_obj = obj

    elif old_version == [0, 0, 7]:
        obj.graph.scale = obj.scale
        for vertex in obj.graph.vertices:
            vertex.friendly_indices = []
            vertex.anti_partner_indices = []
            vertex.true_partner_indices = []
            vertex.unfriendly_partner_indices = []
            vertex.true_anti_partner_indices = []
            vertex.anti_friend_indices = []
            vertex.friend_indices = []
            vertex.outsider_indices = []
        # Updated graph module:
        old_graph = obj.graph
        new_graph = graph_2.AtomicGraph(obj.scale)
        for vertex in old_graph.vertices:
            new_vertex = graph_2.Vertex(vertex.i, vertex.real_coor_x, vertex.real_coor_y, vertex.r, vertex.peak_gamma,
                                        vertex.avg_gamma, obj.scale, level=vertex.level, species_index=vertex.h_index)
            for i, citizen in enumerate(vertex.neighbour_indices):
                new_vertex.district.append(citizen)
            new_vertex.is_in_precipitate = vertex.is_in_precipitate
            new_vertex.is_edge_column = vertex.is_edge_column
            new_vertex.is_set_by_user = vertex.set_by_user
            new_vertex.show_in_overlay = vertex.show_in_overlay
            new_vertex.flag_1 = False
            new_vertex.flag_2 = False
            new_vertex.flag_3 = False
            new_vertex.flag_4 = False
            new_vertex.flag_5 = False
            new_vertex.flag_6 = False
            new_vertex.flag_7 = False
            new_vertex.flag_8 = False
            new_vertex.flag_9 = False

            new_vertex.probability_vector = vertex.prob_vector.tolist()

            new_graph.add_vertex(new_vertex)
        new_graph.refresh_graph()
        obj.graph = new_graph
        obj.active_model = statistics.DataManager.load('default_model')
        obj.graph.active_model = obj.active_model
        fresh_obj = obj

    elif old_version == [0, 0, 8]:
        obj.graph.scale = obj.scale
        for vertex in obj.graph.vertices:
            vertex.anti_partner_indices = []
            vertex.true_partner_indices = []
            vertex.unfriendly_partner_indices = []
            vertex.true_anti_partner_indices = []
            vertex.anti_friend_indices = []
            vertex.friend_indices = []
            vertex.outsider_indices = []
        # Updated graph module:
        old_graph = obj.graph
        new_graph = graph_2.AtomicGraph(obj.scale)
        for vertex in old_graph.vertices:
            new_vertex = graph_2.Vertex(vertex.i, vertex.real_coor_x, vertex.real_coor_y, vertex.r, vertex.peak_gamma,
                                        vertex.avg_gamma, obj.scale, level=vertex.level, species_index=vertex.h_index)
            for i, citizen in enumerate(vertex.neighbour_indices):
                new_vertex.district.append(citizen)
            new_vertex.is_in_precipitate = vertex.is_in_precipitate
            new_vertex.is_edge_column = vertex.is_edge_column
            new_vertex.is_set_by_user = vertex.set_by_user
            new_vertex.show_in_overlay = vertex.show_in_overlay
            new_vertex.flag_1 = False
            new_vertex.flag_2 = False
            new_vertex.flag_3 = False
            new_vertex.flag_4 = False
            new_vertex.flag_5 = False
            new_vertex.flag_6 = False
            new_vertex.flag_7 = False
            new_vertex.flag_8 = False
            new_vertex.flag_9 = False

            new_vertex.probability_vector = vertex.prob_vector.tolist()

            new_graph.add_vertex(new_vertex)
        new_graph.refresh_graph()
        obj.graph = new_graph
        obj.active_model = statistics.DataManager.load('default_model')
        obj.graph.active_model = obj.active_model
        fresh_obj = obj

    elif old_version == [0, 0, 9]:
        obj.graph.scale = obj.scale
        # Updated graph module:
        old_graph = obj.graph
        new_graph = graph_2.AtomicGraph(obj.scale)
        for vertex in old_graph.vertices:
            new_vertex = graph_2.Vertex(vertex.i, vertex.real_coor_x, vertex.real_coor_y, vertex.r, vertex.peak_gamma,
                                        vertex.avg_gamma, obj.scale, level=vertex.level, species_index=vertex.h_index)
            for i, citizen in enumerate(vertex.neighbour_indices):
                new_vertex.district.append(citizen)
            new_vertex.is_in_precipitate = vertex.is_in_precipitate
            new_vertex.is_edge_column = vertex.is_edge_column
            new_vertex.is_set_by_user = vertex.set_by_user
            new_vertex.show_in_overlay = vertex.show_in_overlay
            new_vertex.flag_1 = False
            new_vertex.flag_2 = False
            new_vertex.flag_3 = False
            new_vertex.flag_4 = False
            new_vertex.flag_5 = False
            new_vertex.flag_6 = False
            new_vertex.flag_7 = False
            new_vertex.flag_8 = False
            new_vertex.flag_9 = False

            new_vertex.probability_vector = vertex.prob_vector.tolist()

            new_graph.add_vertex(new_vertex)
        new_graph.refresh_graph()
        obj.graph = new_graph
        obj.active_model = statistics.DataManager.load('default_model')
        obj.graph.active_model = obj.active_model
        fresh_obj = obj

    elif old_version == [0, 0, 10]:
        # Updated graph module:
        old_graph = obj.graph
        new_graph = graph_2.AtomicGraph(obj.scale)
        for vertex in old_graph.vertices:
            new_vertex = graph_2.Vertex(vertex.i, vertex.real_coor_x, vertex.real_coor_y, vertex.r, vertex.peak_gamma,
                                        vertex.avg_gamma, obj.scale, zeta=vertex.level, species_index=vertex.h_index)
            for i, citizen in enumerate(vertex.neighbour_indices):
                new_vertex.district.append(citizen)
            new_vertex.is_in_precipitate = vertex.is_in_precipitate
            new_vertex.is_edge_column = vertex.is_edge_column
            new_vertex.is_set_by_user = vertex.set_by_user
            new_vertex.show_in_overlay = vertex.show_in_overlay
            new_vertex.flag_1 = False
            new_vertex.flag_2 = False
            new_vertex.flag_3 = False
            new_vertex.flag_4 = False
            new_vertex.flag_5 = False
            new_vertex.flag_6 = False
            new_vertex.flag_7 = False
            new_vertex.flag_8 = False
            new_vertex.flag_9 = False

            new_vertex.probability_vector = vertex.prob_vector.tolist()

            new_graph.add_vertex(new_vertex)
        new_graph.refresh_graph()
        obj.graph = new_graph
        obj.active_model = statistics.DataManager.load('default_model')
        obj.graph.active_model = obj.active_model
        fresh_obj = obj

    elif old_version == [0, 0, 11]:
        # Updated graph module:
        old_graph = obj.graph
        new_graph = graph_2.AtomicGraph(obj.scale)
        for vertex in old_graph.vertices:
            new_vertex = graph_2.Vertex(vertex.i, vertex.im_coor_x, vertex.im_coor_y, vertex.r, vertex.peak_gamma,
                                        vertex.avg_gamma, obj.scale, zeta=vertex.level, species_index=vertex.species_index)
            new_vertex.district = vertex.district
            new_vertex.is_in_precipitate = vertex.is_in_precipitate
            new_vertex.is_edge_column = vertex.is_edge_column
            new_vertex.is_set_by_user = vertex.is_set_by_user
            new_vertex.show_in_overlay = vertex.show_in_overlay
            new_vertex.flag_1 = False
            new_vertex.flag_2 = False
            new_vertex.flag_3 = False
            new_vertex.flag_4 = False
            new_vertex.flag_5 = False
            new_vertex.flag_6 = False
            new_vertex.flag_7 = False
            new_vertex.flag_8 = False
            new_vertex.flag_9 = False

            new_vertex.probability_vector = vertex.probability_vector
            new_vertex.normalized_peak_gamma = vertex.normalized_peak_gamma
            new_vertex.normalized_avg_gamma = vertex.normalized_avg_gamma

            new_graph.add_vertex(new_vertex)
        new_graph.refresh_graph()
        obj.graph = new_graph
        obj.active_model = statistics.DataManager.load('default_model')
        obj.graph.active_model = obj.active_model
        fresh_obj = obj

    elif old_version == [0, 0, 12]:
        obj.active_model = statistics.DataManager.load('default_model')
        obj.graph.active_model = obj.active_model
        fresh_obj = obj

    elif old_version == [0, 0, 13]:
        obj.graph.active_model = obj.active_model
        fresh_obj = obj

    else:
        fresh_obj = None

    return fresh_obj

