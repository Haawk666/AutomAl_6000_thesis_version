"""Module containing the graph untangling algorithm by Haakon Tvedt."""
import graph


def strong_resolve(graph_obj, configs, classes, search_type):
    """Resolve discovered configurations while allowing changes to vertex species.

    :param graph_obj: The atomic graph
    :param configs: The edge configurations found in :code:find_types()
    :param classes: Corresponding class of the configurations
    :param search_type: Specific configuration type in question

    :type graph_obj: :code:graph.AtomicGraph
    :type configs: list<tuple<:code:graph.Mesh>>
    :type classes: list<string>
    :type search_type: int

    :return: number of found configurations corresponding to the search_type and number of changes perfored to the graph.
    :rtype: tuple<int>

    """

    changes = 0

    if search_type == 1:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]
            a = config[0].vertex_indices[2]
            b = config[1].vertex_indices[2]

            if class_ == 'A_1':
                if graph_obj.strong_remove_edge(i, j):
                    changes += 1
            elif class_ == 'B_1':
                if graph_obj.vertices[j].partner_query(b):
                    graph_obj.perturb_j_k(j, b, i)
                    changes += 1
            elif class_ == 'B_2':
                if graph_obj.vertices[j].partner_query(a):
                    graph_obj.perturb_j_k(j, a, i)
                    changes += 1
            elif class_ == 'C_1':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(i, j, a)
                    changes += 1
            elif class_ == 'C_2':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(i, j, b)
                    changes += 1
            elif class_ == 'D_1':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(i, j, a)
                    changes += 1
            elif class_ == 'D_2':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(i, j, b)
                    changes += 1
            elif class_ == 'E_1' or class_ == 'E_2':
                if graph_obj.vertices[i].partner_query(j):
                    for partner in graph_obj.vertices[j].partners():
                        if partner not in [i, j, a, b] and not graph_obj.vertices[partner].partner_query(j):
                            graph_obj.perturb_j_k(j, partner, i)
                            changes += 1
                            break
            elif class_ == 'F_1' or class_ == 'F_2':
                pass
            elif class_ == 'G_1':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(j, b, i)
                    changes += 1
            elif class_ == 'G_2':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(j, a, i)
                    changes += 1
            elif class_ == 'H_1':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(i, j, a)
                    changes += 1
            elif class_ == 'H_2':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(i, j, b)
                    changes += 1

        return len(classes), changes

    elif search_type == 2:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]
            a = config[0].vertex_indices[2]
            b = config[0].vertex_indices[3]

            if class_ == 'A_1':
                pass
            elif class_ == 'B_1':
                if graph_obj.vertices[i].partner_query(j) and graph_obj.vertices[a].partner_query(b):
                    graph_obj.perturb_j_k(i, j, a)
                    graph_obj.perturb_j_k(a, b, i)
                    changes += 1

        return len(classes), changes

    elif search_type == 3:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]
            b = config[1].vertex_indices[2]
            c = config[1].vertex_indices[3]

            if class_ == 'A_1':
                pass
            elif class_ == 'B_1':
                if graph_obj.vertices[i].partner_query(j) and graph_obj.vertices[c].partner_query(b):
                    graph_obj.perturb_j_k(i, j, c)
                    graph_obj.perturb_j_k(c, b, i)
                    changes += 1

        return len(classes), changes

    elif search_type == 4:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]
            d = config[1].vertex_indices[4]

            if class_ == 'A_1':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(i, j, d)
                    changes += 1

        return len(classes), changes

    elif search_type == 5:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]
            a = config[0].vertex_indices[2]

            if class_ == 'A_1':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(i, j, a)
                    changes += 1

        return len(classes), changes

    else:

        return len(classes), 0


def weak_resolve(graph_obj, configs, classes, search_type):

    changes = 0

    if search_type == 1:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]
            a = config[0].vertex_indices[2]
            b = config[1].vertex_indices[2]

            if class_ == 'A_1':
                if graph_obj.weak_remove_edge(i, j):
                    changes += 1
            elif class_ == 'B_1':
                if graph_obj.vertices[j].partner_query(b):
                    graph_obj.perturb_j_k(j, b, i)
                    changes += 1
            elif class_ == 'B_2':
                if graph_obj.vertices[j].partner_query(a):
                    graph_obj.perturb_j_k(j, a, i)
                    changes += 1
            elif class_ == 'C_1':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(i, j, a)
                    changes += 1
            elif class_ == 'C_2':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(i, j, b)
                    changes += 1
            elif class_ == 'D_1':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(i, j, a)
                    changes += 1
            elif class_ == 'D_2':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(i, j, b)
                    changes += 1
            elif class_ == 'E_1' or class_ == 'E_2':
                if graph_obj.vertices[i].partner_query(j):
                    for partner in graph_obj.vertices[j].partners():
                        if partner not in [i, j, a, b] and not graph_obj.vertices[partner].partner_query(j):
                            graph_obj.perturb_j_k(j, partner, i)
                            changes += 1
                            break
            elif class_ == 'F_1' or class_ == 'F_2':
                pass
            elif class_ == 'G_1':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(j, b, i)
                    changes += 1
            elif class_ == 'G_2':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(j, a, i)
                    changes += 1
            elif class_ == 'H_1':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(i, j, a)
                    changes += 1
            elif class_ == 'H_2':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(i, j, b)
                    changes += 1

        return len(classes), changes

    elif search_type == 2:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]
            a = config[0].vertex_indices[2]
            b = config[0].vertex_indices[3]

            if class_ == 'A_1':
                pass
            elif class_ == 'B_1':
                if graph_obj.vertices[i].partner_query(j) and graph_obj.vertices[a].partner_query(b):
                    graph_obj.perturb_j_k(i, j, a)
                    graph_obj.perturb_j_k(a, b, i)
                    changes += 1

        return len(classes), changes

    elif search_type == 3:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]
            b = config[1].vertex_indices[2]
            c = config[1].vertex_indices[3]

            if class_ == 'A_1':
                pass
            elif class_ == 'B_1':
                if graph_obj.vertices[i].partner_query(j) and graph_obj.vertices[c].partner_query(b):
                    graph_obj.perturb_j_k(i, j, c)
                    graph_obj.perturb_j_k(c, b, i)
                    changes += 1

        return len(classes), changes

    elif search_type == 4:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]
            d = config[1].vertex_indices[4]

            if class_ == 'A_1':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(i, j, d)
                    changes += 1

        return len(classes), changes

    elif search_type == 5:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]
            a = config[0].vertex_indices[2]

            if class_ == 'A_1':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.perturb_j_k(i, j, a)
                    changes += 1

        return len(classes), changes

    else:

        return len(classes), 0


def find_class(graph_obj, type_, config):

    if type_ == 1:
        i = config[0].vertex_indices[0]
        j = config[0].vertex_indices[1]
        a = config[0].vertex_indices[2]
        b = config[1].vertex_indices[2]
        s = [(j, a), (a, i), (i, b), (b, j)]
        s_types = []

        for f in range(0, 4):
            if graph_obj.vertices[s[f][0]].partner_query(s[f][1]):
                if graph_obj.vertices[s[f][1]].partner_query(s[f][0]):
                    s_types.append(0)
                else:
                    s_types.append(1)
            else:
                s_types.append(2)

        if s_types == [0, 0, 0, 0]:
            return 'A_1'
        elif s_types == [0, 2, 0, 2]:
            return 'B_1'
        elif s_types == [1, 0, 1, 0]:
            return 'B_2'
        elif s_types == [0, 1, 0, 0]:
            return 'C_1'
        elif s_types == [0, 0, 2, 0]:
            return 'C_2'
        elif s_types == [0, 1, 1, 0]:
            return 'D_1'
        elif s_types == [0, 2, 2, 0]:
            return 'D_2'
        elif s_types == [0, 2, 0, 1]:
            return 'E_1'
        elif s_types == [2, 0, 1, 0]:
            return 'E_2'
        elif s_types == [0, 2, 0, 0]:
            return 'F_1'
        elif s_types == [0, 0, 1, 0]:
            return 'F_2'
        elif s_types == [0, 0, 0, 2]:
            return 'G_1'
        elif s_types == [1, 0, 0, 0]:
            return 'G_2'
        elif s_types == [2, 1, 0, 0]:
            return 'H_1'
        elif s_types == [0, 0, 2, 1]:
            return 'H_2'
        else:
            return 'I'

    elif type_ == 2:
        i = config[0].vertex_indices[0]
        j = config[0].vertex_indices[1]
        a = config[0].vertex_indices[2]
        b = config[0].vertex_indices[3]
        c = config[1].vertex_indices[2]
        s = [(j, a), (a, b), (b, i), (i, c), (c, j)]
        s_types = []

        for f in range(0, 5):
            if graph_obj.vertices[s[f][0]].partner_query(s[f][1]):
                if graph_obj.vertices[s[f][1]].partner_query(s[f][0]):
                    s_types.append(0)
                else:
                    s_types.append(1)
            else:
                s_types.append(2)

        if s_types == [0, 0, 0, 0, 0]:
            return 'A_1'
        elif s_types == [0, 1, 0, 0, 0]:
            return 'B_1'
        else:
            return 'I'

    elif type_ == 3:
        i = config[0].vertex_indices[0]
        j = config[0].vertex_indices[1]
        a = config[0].vertex_indices[2]
        b = config[1].vertex_indices[2]
        c = config[1].vertex_indices[3]
        s = [(j, a), (a, i), (i, b), (b, c), (c, j)]
        s_types = []

        for f in range(0, 5):
            if graph_obj.vertices[s[f][0]].partner_query(s[f][1]):
                if graph_obj.vertices[s[f][1]].partner_query(s[f][0]):
                    s_types.append(0)
                else:
                    s_types.append(1)
            else:
                s_types.append(2)

        if s_types == [0, 0, 0, 0, 0]:
            return 'A_1'
        elif s_types == [0, 0, 0, 2, 0]:
            return 'B_1'
        else:
            return 'I'

    elif type_ == 4:
        i = config[0].vertex_indices[0]
        j = config[0].vertex_indices[1]
        a = config[0].vertex_indices[2]
        b = config[1].vertex_indices[2]
        c = config[1].vertex_indices[3]
        d = config[1].vertex_indices[4]
        s = [(j, a), (a, i), (i, b), (b, c), (c, d), (d, j)]
        s_types = []

        for f in range(0, 6):
            if graph_obj.vertices[s[f][0]].partner_query(s[f][1]):
                if graph_obj.vertices[s[f][1]].partner_query(s[f][0]):
                    s_types.append(0)
                else:
                    s_types.append(1)
            else:
                s_types.append(2)

        if s_types == [0, 0, 0, 0, 0, 0]:
            return 'A_1'
        else:
            return 'I'

    elif type_ == 5:
        i = config[0].vertex_indices[0]
        j = config[0].vertex_indices[1]
        a = config[0].vertex_indices[2]
        b = config[0].vertex_indices[3]
        c = config[0].vertex_indices[4]
        d = config[1].vertex_indices[2]
        s = [(j, a), (a, b), (b, c), (c, i), (i, d), (d, j)]
        s_types = []

        for f in range(0, 6):
            if graph_obj.vertices[s[f][0]].partner_query(s[f][1]):
                if graph_obj.vertices[s[f][1]].partner_query(s[f][0]):
                    s_types.append(0)
                else:
                    s_types.append(1)
            else:
                s_types.append(2)

        if s_types == [0, 0, 0, 0, 0, 0]:
            return 'A_1'
        else:
            return 'I'

    else:
        return 'I'


def find_type(graph_obj, search_type):

    configs = []
    classes = []

    for vertex in graph_obj.vertices:
        for partner in vertex.partners():

            if not graph_obj.vertices[partner].partner_query(vertex.i) and \
                    not vertex.is_edge_column and not graph_obj.vertices[partner].is_edge_column:

                corners_1, angles_1, vectors_1 = graph_obj.find_mesh(vertex.i, partner)
                corners_2, angles_2, vectors_2 = graph_obj.find_mesh(partner, vertex.i)

                mesh_1 = graph.Mesh()
                for k, corner in enumerate(corners_1):
                    mesh_1.add_vertex(graph_obj.vertices[corner])
                    mesh_1.angles.append(angles_1[k])
                    mesh_1.angle_vectors.append(vectors_1[k])
                mesh_1.redraw_edges()

                mesh_2 = graph.Mesh()
                for k, corner in enumerate(corners_2):
                    mesh_2.add_vertex(graph_obj.vertices[corner])
                    mesh_2.angles.append(angles_2[k])
                    mesh_2.angle_vectors.append(vectors_2[k])
                mesh_2.redraw_edges()

                if search_type == 1 and mesh_1.num_corners == 3 and mesh_2.num_corners == 3:
                    config = mesh_1, mesh_2
                    class_ = find_class(graph_obj, 1, config)
                    if not class_ == 'I':
                        configs.append(config)
                        classes.append(class_)
                    else:
                        print('Class I encountered: {} {}'.format(vertex.i, partner))

                elif search_type == 2 and mesh_1.num_corners == 4 and mesh_2.num_corners == 3:
                    config = mesh_1, mesh_2
                    class_ = find_class(graph_obj, 2, config)
                    if not class_ == 'I':
                        configs.append(config)
                        classes.append(class_)
                    else:
                        print('Class I encountered: {} {}'.format(vertex.i, partner))

                elif search_type == 3 and mesh_1.num_corners == 3 and mesh_2.num_corners == 4:
                    config = mesh_1, mesh_2
                    class_ = find_class(graph_obj, 3, config)
                    if not class_ == 'I':
                        configs.append(config)
                        classes.append(class_)
                    else:
                        print('Class I encountered: {} {}'.format(vertex.i, partner))

                elif search_type == 4 and mesh_1.num_corners == 3 and mesh_2.num_corners == 5:
                    config = mesh_1, mesh_2
                    class_ = find_class(graph_obj, 4, config)
                    if not class_ == 'I':
                        configs.append(config)
                        classes.append(class_)
                    else:
                        print('Class I encountered: {} {}'.format(vertex.i, partner))

                elif search_type == 5 and mesh_1.num_corners == 5 and mesh_2.num_corners == 3:
                    config = mesh_1, mesh_2
                    class_ = find_class(graph_obj, 5, config)
                    if not class_ == 'I':
                        configs.append(config)
                        classes.append(class_)
                    else:
                        print('Class I encountered: {} {}'.format(vertex.i, partner))

                else:
                    print('Hitler!')

    return configs, classes


def untangle(graph_obj, search_type, strong=False):
    configs, classes = find_type(graph_obj, search_type)
    if strong:
        num_found, changes = strong_resolve(graph_obj, configs, classes, search_type)
    else:
        num_found, changes = weak_resolve(graph_obj, configs, classes, search_type)
    return changes

