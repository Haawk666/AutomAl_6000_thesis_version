import core


def convert(obj, old_version, version):

    # Return obj in a compatible state
    # Return None if not possible!

    if old_version == [0, 0, 0]:
        # Set new attributes
        obj.starting_index = None
        for vertex in obj.graph.vertices:
            vertex.symmetry_vector = [1/3, 1/3, 1/3]
            vertex.flag_5 = False
            vertex.flag_6 = False
            vertex.flag_7 = False
            vertex.flag_8 = False
            vertex.ad_hoc_list_1 = []
            vertex.ad_hoc_list_1 = []
            vertex.ad_hoc_value_1 = 0
            vertex.ad_hoc_value_2 = 0
        fresh_obj = obj
    elif old_version == [0, 0, 1]:
        # Set new attributes
        for vertex in obj.graph.vertices:
            vertex.symmetry_vector = [1/3, 1/3, 1/3]
            vertex.flag_5 = False
            vertex.flag_6 = False
            vertex.flag_7 = False
            vertex.flag_8 = False
            vertex.ad_hoc_list_1 = []
            vertex.ad_hoc_list_1 = []
            vertex.ad_hoc_value_1 = 0
            vertex.ad_hoc_value_2 = 0
        fresh_obj = obj
    else:
        fresh_obj = None

    return fresh_obj

