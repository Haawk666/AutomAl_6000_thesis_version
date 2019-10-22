"""Module to handle forwards compatibility between versions.

.. note::

    Only forwards compatibility is maintained. Opening a project file that was saved in a more recent version is
    generally not guarantied to be possible, but opening old project files in newer version should always be possible.

"""


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
        obj.graph.avg_central_variance = 0.0
        obj.graph.meshes = []
        obj.graph.mesh_indices = []
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
        obj.graph.avg_central_variance = 0.0
        obj.graph.meshes = []
        obj.graph.mesh_indices = []
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
        obj.graph.avg_central_variance = 0.0
        obj.graph.meshes = []
        obj.graph.mesh_indices = []
        fresh_obj = obj

    elif old_version == [0, 0, 3]:
        for vertex in obj.graph.vertices:
            vertex.central_angle_variance = 0.0
            vertex.spatial_coor_x = obj.scale * vertex.real_coor_x
            vertex.spatial_coor_y = obj.scale * vertex.real_coor_y
            vertex.normalized_peak_gamma = vertex.peak_gamma
            vertex.normalized_avg_gamma = vertex.avg_gamma
            vertex.friendly_indices = []
        obj.graph.avg_central_variance = 0.0
        obj.graph.meshes = []
        obj.graph.mesh_indices = []
        fresh_obj = obj

    elif old_version == [0, 0, 4]:
        for vertex in obj.graph.vertices:
            vertex.spatial_coor_x = obj.scale * vertex.real_coor_x
            vertex.spatial_coor_y = obj.scale * vertex.real_coor_y
            vertex.normalized_peak_gamma = vertex.peak_gamma
            vertex.normalized_avg_gamma = vertex.avg_gamma
            vertex.friendly_indices = []
        obj.graph.meshes = []
        obj.graph.mesh_indices = []
        fresh_obj = obj

    elif old_version == [0, 0, 5]:
        for vertex in obj.graph.vertices:
            vertex.normalized_peak_gamma = vertex.peak_gamma
            vertex.normalized_avg_gamma = vertex.avg_gamma
            vertex.friendly_indices = []
        obj.graph.meshes = []
        obj.graph.mesh_indices = []
        fresh_obj = obj

    elif old_version == [0, 0, 6]:
        for vertex in obj.graph.vertices:
            vertex.friendly_indices = []
        obj.graph.meshes = []
        obj.graph.mesh_indices = []
        fresh_obj = obj

    elif old_version == [0, 0, 7]:
        for vertex in obj.graph.vertices:
            vertex.friendly_indices = []

    else:
        fresh_obj = None

    return fresh_obj

