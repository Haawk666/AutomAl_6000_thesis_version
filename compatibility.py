import core


def convert(obj, old_version, version):

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
        obj.graph.avg_central_variance = 0.0
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
        obj.graph.avg_central_variance = 0.0
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
        obj.graph.avg_central_variance = 0.0
        fresh_obj = obj
    elif old_version == [0, 0, 3]:
        for vertex in obj.graph.vertices:
            vertex.central_angle_variance = 0.0
        obj.graph.avg_central_variance = 0.0
        fresh_obj = obj
    else:
        fresh_obj = None

    return fresh_obj

