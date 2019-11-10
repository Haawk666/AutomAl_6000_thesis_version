"""Module containing the graph untangling algorithm by Haakon Tvedt."""
import graph
import logging
# Instantiate logger:
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def strong_resolve(graph_obj, configs, classes, search_type, ui_obj=None):
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

    if ui_obj:
        print('This is not good!!..')

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
                pass
            elif class_ == 'B_2':
                pass
            elif class_ == 'C_1':
                pass
            elif class_ == 'C_2':
                pass
            elif class_ == 'D_1':
                pass
            elif class_ == 'D_2':
                pass
            elif class_ == 'E_1' or class_ == 'E_2':
                pass
            elif class_ == 'F_1' or class_ == 'F_2':
                if graph_obj.strong_remove_edge(i, j):
                    changes += 1
            elif class_ == 'G_1':
                pass
            elif class_ == 'G_2':
                pass
            elif class_ == 'H_1':
                pass
            elif class_ == 'H_2':
                pass

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
                    graph_obj.permute_j_k(i, j, a)
                    graph_obj.permute_j_k(a, b, i)
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
                    graph_obj.permute_j_k(i, j, c)
                    graph_obj.permute_j_k(c, b, i)
                    changes += 1

        return len(classes), changes

    elif search_type == 4:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]
            d = config[1].vertex_indices[4]

            if class_ == 'A_1':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.permute_j_k(i, j, d)
                    changes += 1

        return len(classes), changes

    elif search_type == 5:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]
            a = config[0].vertex_indices[2]

            if class_ == 'A_1':
                if graph_obj.vertices[i].partner_query(j):
                    graph_obj.permute_j_k(i, j, a)
                    changes += 1

        return len(classes), changes

    elif search_type == 6:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]

            if class_ == 'A_1' or class_ == 'B_1' or class_ == 'B_2' or class_ == 'C_1' or class_ == 'D_1' or \
                    class_ == 'D_2' or class_ == 'E_1':
                graph_obj.strong_enforce_edge(i, j)
                changes += 1

        return len(classes), changes

    else:

        return len(classes), 0


def weak_resolve(graph_obj, configs, classes, search_type, ui_obj=None):

    changes = 0

    if search_type == 1:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]
            a = config[0].vertex_indices[2]
            b = config[1].vertex_indices[2]

            if class_ == 'A_1':
                k = graph_obj.weak_remove_edge(i, j, aggressive=True)
                if not k == -1:
                    if graph_obj.permute_j_k(i, j, k):
                        changes += 1
                        if ui_obj is not None:
                            ui_obj.gs_atomic_graph.perturb_edge(i, j, k, permute_data=False)
            elif class_ == 'B_1':
                pass

            elif class_ == 'B_2':
                pass

            elif class_ == 'C_1':
                if graph_obj.vertices[i].partner_query(j):
                    if graph_obj.permute_j_k(i, j, a):
                        changes += 1
                        if ui_obj is not None:
                            ui_obj.gs_atomic_graph.perturb_edge(i, j, a, permute_data=False)
            elif class_ == 'C_2':
                if graph_obj.vertices[i].partner_query(j):
                    if graph_obj.permute_j_k(i, j, b):
                        changes += 1
                        if ui_obj is not None:
                            ui_obj.gs_atomic_graph.perturb_edge(i, j, b, permute_data=False)
            elif class_ == 'D_1':
                if graph_obj.vertices[i].partner_query(j):
                    if graph_obj.permute_j_k(i, j, a):
                        changes += 1
                        if ui_obj is not None:
                            ui_obj.gs_atomic_graph.perturb_edge(i, j, a, permute_data=False)
            elif class_ == 'D_2':
                if graph_obj.vertices[i].partner_query(j):
                    if graph_obj.permute_j_k(i, j, b):
                        changes += 1
                        if ui_obj is not None:
                            ui_obj.gs_atomic_graph.perturb_edge(i, j, b, permute_data=False)
            elif class_ == 'E_1' or class_ == 'E_2':
                pass
            elif class_ == 'F_1' or class_ == 'F_2':
                k = graph_obj.weak_remove_edge(i, j, aggressive=False)
                if not k == -1:
                    if graph_obj.permute_j_k(i, j, k):
                        changes += 1
                        if ui_obj is not None:
                            ui_obj.gs_atomic_graph.perturb_edge(i, j, k, permute_data=False)

            elif class_ == 'G_1':
                pass

            elif class_ == 'G_2':
                pass

            elif class_ == 'H_1':
                if graph_obj.vertices[i].partner_query(j):
                    if graph_obj.permute_j_k(i, j, a):
                        changes += 1
                        if ui_obj is not None:
                            ui_obj.gs_atomic_graph.perturb_edge(i, j, a, permute_data=False)
            elif class_ == 'H_2':
                if graph_obj.vertices[i].partner_query(j):
                    if graph_obj.permute_j_k(i, j, b):
                        changes += 1
                        if ui_obj is not None:
                            ui_obj.gs_atomic_graph.perturb_edge(i, j, b, permute_data=False)
            elif class_ == 'I_1':
                if graph_obj.permute_j_k(i, j, b):
                    changes += 1
                    if ui_obj is not None:
                        ui_obj.gs_atomic_graph.perturb_edge(i, j, b, permute_data=False)

        return changes

    elif search_type == 2:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]
            a = config[0].vertex_indices[2]
            b = config[0].vertex_indices[3]
            c = config[1].vertex_indices[2]

            if class_ == 'A_1':
                pass

            elif class_ == 'B_1' or class_ == 'C_1':
                if graph_obj.vertices[i].partner_query(j) and graph_obj.vertices[a].partner_query(b):
                    if graph_obj.permute_j_k(i, j, a):
                        changes += 1
                        if ui_obj is not None:
                            ui_obj.gs_atomic_graph.perturb_edge(i, j, a, permute_data=False)

            elif class_ == 'D_1':
                pass

            elif class_ == 'E_1':
                pass

        return changes

    elif search_type == 3:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]
            a = config[0].vertex_indices[2]
            b = config[1].vertex_indices[2]
            c = config[1].vertex_indices[3]

            if class_ == 'A_1':
                pass

            elif class_ == 'B_1' or class_ == 'C_1':
                if graph_obj.vertices[i].partner_query(j) and graph_obj.vertices[c].partner_query(b):
                    if graph_obj.permute_j_k(i, j, c):
                        changes += 1
                        if ui_obj is not None:
                            ui_obj.gs_atomic_graph.perturb_edge(i, j, c, permute_data=False)

            elif class_ == 'D_1':
                pass

            elif class_ == 'E_1':
                pass

        return changes

    elif search_type == 4:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]
            d = config[1].vertex_indices[4]

            if class_ == 'A_1':
                if graph_obj.permute_j_k(i, j, d):
                    changes += 1
                    if ui_obj is not None:
                        ui_obj.gs_atomic_graph.perturb_edge(i, j, d, permute_data=False)

            elif class_ == 'B_1':
                if graph_obj.permute_j_k(i, j, d):
                    changes += 1
                    if ui_obj is not None:
                        ui_obj.gs_atomic_graph.perturb_edge(i, j, d, permute_data=False)

        return changes

    elif search_type == 5:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]
            a = config[0].vertex_indices[2]

            if class_ == 'A_1':
                if graph_obj.permute_j_k(i, j, a):
                    changes += 1
                    if ui_obj is not None:
                        ui_obj.gs_atomic_graph.perturb_edge(i, j, a, permute_data=False)

            elif class_ == 'B_1':
                if graph_obj.permute_j_k(i, j, a):
                    changes += 1
                    if ui_obj is not None:
                        ui_obj.gs_atomic_graph.perturb_edge(i, j, a, permute_data=False)

        return changes

    elif search_type == 6:
        for class_, config in zip(classes, configs):
            i = config[0].vertex_indices[0]
            j = config[0].vertex_indices[1]
            a = config[0].vertex_indices[2]
            b = config[0].vertex_indices[3]
            c = config[1].vertex_indices[2]
            d = config[1].vertex_indices[3]

            if class_ == 'A_1':
                k = graph_obj.weak_preserve_edge(i, j)
                if not k == -1:
                    if graph_obj.permute_j_k(j, k, i):
                        changes += 1
                        if ui_obj is not None:
                            ui_obj.gs_atomic_graph.perturb_edge(j, k, i, permute_data=False)

            elif class_ == 'B_1' or class_ == 'B_2':
                k = graph_obj.weak_preserve_edge(i, j)
                if not k == -1:
                    if graph_obj.permute_j_k(j, k, i):
                        changes += 1
                        if ui_obj is not None:
                            ui_obj.gs_atomic_graph.perturb_edge(j, k, i, permute_data=False)

            elif class_ == 'C_1':
                k = graph_obj.weak_preserve_edge(i, j)
                if not k == -1:
                    if graph_obj.permute_j_k(j, k, i):
                        changes += 1
                        if ui_obj is not None:
                            ui_obj.gs_atomic_graph.perturb_edge(j, k, i, permute_data=False)

        return changes

    else:

        return 0


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
        elif s_types == [0, 1, 2, 0]:
            return 'I_1'
        else:
            return 'J'

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
        elif s_types == [0, 1, 0, 1, 0]:
            return 'C_1'
        elif s_types == [0, 0, 0, 0, 2]:
            return 'D_1'
        elif s_types == [0, 0, 1, 0, 0]:
            return 'E_1'
        else:
            return 'J'

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
        elif s_types == [0, 2, 0, 2, 0]:
            return 'C_1'
        elif s_types == [1, 0, 0, 0, 0]:
            return 'D_1'
        elif s_types == [0, 0, 2, 0, 0]:
            return 'E_1'
        else:
            return 'J'

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
        elif s_types == [0, 0, 0, 0, 2, 0]:
            return 'B_1'
        else:
            return 'J'

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
        elif s_types == [0, 1, 0, 0, 0, 0]:
            return 'B_1'
        else:
            return 'J'

    elif type_ == 6:
        i = config[0].vertex_indices[0]
        j = config[0].vertex_indices[1]
        a = config[0].vertex_indices[2]
        b = config[0].vertex_indices[3]
        c = config[1].vertex_indices[2]
        d = config[1].vertex_indices[3]
        s = [(j, a), (a, b), (b, i), (i, c), (c, d), (d, j)]
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
        elif s_types == [0, 0, 0, 0, 0, 1]:
            return 'B_1'
        elif s_types == [2, 0, 0, 0, 0, 0]:
            return 'B_2'
        elif s_types == [2, 0, 0, 0, 0, 1]:
            return 'C_1'
        elif s_types == [0, 0, 0, 2, 0, 0]:
            return 'D_1'
        elif s_types == [0, 0, 1, 0, 0, 0]:
            return 'D_2'
        elif s_types == [0, 0, 1, 2, 0, 0]:
            return 'E_1'
        else:
            return 'J'

    else:
        return 'J'


def find_type(graph_obj, search_type, strong=False, ui_obj=None):

    found = 0
    changes = 0

    for vertex in graph_obj.vertices:
        for partner in vertex.partners():

            if not graph_obj.vertices[partner].partner_query(vertex.i) and \
                    not vertex.is_edge_column and not graph_obj.vertices[partner].is_edge_column:

                corners_1, angles_1, vectors_1 = graph_obj.find_mesh(vertex.i, partner, use_friends=True)
                corners_2, angles_2, vectors_2 = graph_obj.find_mesh(partner, vertex.i, use_friends=True)

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
                    if not class_ == 'J':
                        if strong:
                            changes += strong_resolve(graph_obj, [config], [class_], search_type, ui_obj=ui_obj)
                        else:
                            logger.info('{}, {}: {} - {}'.format(config[0].vertex_indices[0], config[0].vertex_indices[1], search_type, class_))
                            changes += weak_resolve(graph_obj, [config], [class_], search_type, ui_obj=ui_obj)
                        found += 1

                elif search_type == 2 and mesh_1.num_corners == 4 and mesh_2.num_corners == 3:
                    config = mesh_1, mesh_2
                    class_ = find_class(graph_obj, 2, config)
                    if not class_ == 'J':
                        if strong:
                            changes += strong_resolve(graph_obj, [config], [class_], search_type, ui_obj=ui_obj)
                        else:
                            logger.info('{}, {}: {} - {}'.format(config[0].vertex_indices[0], config[0].vertex_indices[1],
                                                           search_type, class_))
                            changes += weak_resolve(graph_obj, [config], [class_], search_type, ui_obj=ui_obj)
                        found += 1

                elif search_type == 3 and mesh_1.num_corners == 3 and mesh_2.num_corners == 4:
                    config = mesh_1, mesh_2
                    class_ = find_class(graph_obj, 3, config)
                    if not class_ == 'J':
                        if strong:
                            changes += strong_resolve(graph_obj, [config], [class_], search_type, ui_obj=ui_obj)
                        else:
                            logger.info('{}, {}: {} - {}'.format(config[0].vertex_indices[0], config[0].vertex_indices[1],
                                                           search_type, class_))
                            changes += weak_resolve(graph_obj, [config], [class_], search_type, ui_obj=ui_obj)
                        found += 1

                elif search_type == 4 and mesh_1.num_corners == 3 and mesh_2.num_corners == 5:
                    config = mesh_1, mesh_2
                    class_ = find_class(graph_obj, 4, config)
                    if not class_ == 'J':
                        if strong:
                            changes += strong_resolve(graph_obj, [config], [class_], search_type, ui_obj=ui_obj)
                        else:
                            logger.info('{}, {}: {} - {}'.format(config[0].vertex_indices[0], config[0].vertex_indices[1],
                                                           search_type, class_))
                            changes += weak_resolve(graph_obj, [config], [class_], search_type, ui_obj=ui_obj)
                        found += 1

                elif search_type == 5 and mesh_1.num_corners == 5 and mesh_2.num_corners == 3:
                    config = mesh_1, mesh_2
                    class_ = find_class(graph_obj, 5, config)
                    if not class_ == 'J':
                        if strong:
                            changes += strong_resolve(graph_obj, [config], [class_], search_type, ui_obj=ui_obj)
                        else:
                            logger.info('{}, {}: {} - {}'.format(config[0].vertex_indices[0], config[0].vertex_indices[1],
                                                           search_type, class_))
                            changes += weak_resolve(graph_obj, [config], [class_], search_type, ui_obj=ui_obj)
                        found += 1

                elif search_type == 6 and mesh_1.num_corners == 4 and mesh_2.num_corners == 4:
                    config = mesh_1, mesh_2
                    class_ = find_class(graph_obj, 6, config)
                    if not class_ == 'J':
                        if strong:
                            changes += strong_resolve(graph_obj, [config], [class_], search_type, ui_obj=ui_obj)
                        else:
                            logger.info('{}, {}: {} - {}'.format(config[0].vertex_indices[0], config[0].vertex_indices[1],
                                                           search_type, class_))
                            changes += weak_resolve(graph_obj, [config], [class_], search_type, ui_obj=ui_obj)
                        found += 1

    return found, changes


def untangle(graph_obj, search_type, strong=False, ui_obj=None):
    if search_type in [1, 2, 3, 4, 5, 6]:
        num_found, changes = find_type(graph_obj, search_type, strong=strong, ui_obj=ui_obj)
        return num_found, changes
    else:
        return 0, 0


def find_column_config(sub_graph, search_type):

    s = []

    for mesh in sub_graph.meshes:
        if mesh.num_corners == 3:
            s.append(1)
        elif mesh.num_corners == 4:
            s.append(2)
        elif mesh.num_corners == 5:
            s.append(3)
        elif mesh.num_corners == 6:
            s.append(4)
        else:
            s.append(5)

    for mesh_id, mesh_type in enumerate(s):

        if mesh_type == 1 and s[mesh_id - 1] == 1:
            type_ = 1
        elif mesh_type == 1 and s[mesh_id - 1] == 3:
            type_ = 2
        else:
            type_ = 0

    return type_


def resolve_column_config(graph_obj, search_type):

    pass


def column_centered_untangling(graph_obj, search_type):

    for i in graph_obj.vertex_indices:

        sub_graph = graph_obj.get_atomic_configuration(i)

        type_ = find_column_config(sub_graph, search_type)


def mesh_analysis(graph_obj):

    interesting_meshes = []
    mesh_categories = []

    for mesh in graph_obj.meshes:
        s = []
        s.append(mesh.num_corners)
        for neighbour_index in mesh.surrounding_meshes:
            s.append(graph_obj.meshes[neighbour_index].num_corners)
        if s == [4, 3, 4, 3, 4] or s == [4, 4, 3, 4, 3]:
            interesting_meshes.append(mesh)
            mesh_categories.append(1)
        elif s == [5, 4, 4, 4, 4, 3] or \
                s == [5, 4, 4, 4, 3, 4] or \
                s == [5, 4, 4, 3, 4, 4] or \
                s == [5, 4, 3, 4, 4, 4] or \
                s == [5, 3, 4, 4, 4, 4]:
            interesting_meshes.append(mesh)
            mesh_categories.append(2)

    num_changes = resolve_mesh_configs(graph_obj, interesting_meshes, mesh_categories)
    return num_changes


def resolve_mesh_configs(graph_obj, interesting_meshes, mesh_categories):

    changes_made = 0

    for mesh, mesh_category in zip(interesting_meshes, mesh_categories):
        if mesh_category == 1:
            distance_1 = graph_obj.projected_distance(mesh.vertex_indices[0], mesh.vertex_indices[2])
            distance_2 = graph_obj.projected_distance(mesh.vertex_indices[1], mesh.vertex_indices[3])
            if distance_1 < distance_2:
                if graph_obj.meshes[mesh.surrounding_meshes[0]].num_corners == 4:
                    a = mesh.vertex_indices[0]
                    b = mesh.vertex_indices[1]
                    c = mesh.vertex_indices[2]
                    d = mesh.vertex_indices[3]
                else:
                    a = mesh.vertex_indices[0]
                    b = mesh.vertex_indices[3]
                    c = mesh.vertex_indices[2]
                    d = mesh.vertex_indices[1]
            else:
                if graph_obj.meshes[mesh.surrounding_meshes[0]].num_corners == 4:
                    a = mesh.vertex_indices[3]
                    b = mesh.vertex_indices[2]
                    c = mesh.vertex_indices[1]
                    d = mesh.vertex_indices[0]
                else:
                    a = mesh.vertex_indices[3]
                    b = mesh.vertex_indices[0]
                    c = mesh.vertex_indices[1]
                    d = mesh.vertex_indices[2]
            if a in graph_obj.vertices[b].partners():
                graph_obj.permute_j_to_last_partner(b, a)
                graph_obj.decrease_h(b)
            if b in graph_obj.vertices[a].partners():
                graph_obj.permute_j_k(a, b, c)
            else:
                graph_obj.permute_j_to_first_antipartner(a, c)
                graph_obj.increase_h(a)
            if d in graph_obj.vertices[c].partners():
                graph_obj.permute_j_k(c, d, a)
            else:
                graph_obj.permute_j_to_first_antipartner(c, a)
                graph_obj.increase_h(c)
            if c in graph_obj.vertices[d].partners():
                graph_obj.permute_j_to_last_partner(d, c)
                graph_obj.decrease_h(d)
            changes_made += 1
        elif mesh_category == 2:
            for k, neighbour_geometry in enumerate(graph_obj.meshes[neighbour_mesh_index].num_corners for neighbour_mesh_index in mesh.surrounding_meshes):
                if neighbour_geometry == 3:
                    if k == 0:
                        a = mesh.vertex_indices[0]
                        b = mesh.vertex_indices[1]
                        c = mesh.vertex_indices[2]
                        d = mesh.vertex_indices[3]
                        e = mesh.vertex_indices[4]
                    elif k == 1:
                        a = mesh.vertex_indices[1]
                        b = mesh.vertex_indices[2]
                        c = mesh.vertex_indices[3]
                        d = mesh.vertex_indices[4]
                        e = mesh.vertex_indices[0]
                    elif k == 2:
                        a = mesh.vertex_indices[2]
                        b = mesh.vertex_indices[3]
                        c = mesh.vertex_indices[4]
                        d = mesh.vertex_indices[0]
                        e = mesh.vertex_indices[1]
                    elif k == 3:
                        a = mesh.vertex_indices[3]
                        b = mesh.vertex_indices[4]
                        c = mesh.vertex_indices[0]
                        d = mesh.vertex_indices[1]
                        e = mesh.vertex_indices[2]
                    else:
                        a = mesh.vertex_indices[4]
                        b = mesh.vertex_indices[0]
                        c = mesh.vertex_indices[1]
                        d = mesh.vertex_indices[2]
                        e = mesh.vertex_indices[3]
                    if e in graph_obj.vertices[a].partners():
                        graph_obj.permute_j_k(a, e, d)
                    else:
                        graph_obj.permute_j_to_first_antipartner(a, d)
                        graph_obj.increase_h(a)
                    if a in graph_obj.vertices[e].partners():
                        graph_obj.permute_j_to_last_partner(e, a)
                        graph_obj.decrease_h(e)
                    graph_obj.permute_j_to_first_antipartner(d, a)
                    graph_obj.increase_h(d)
                    changes_made += 1
                    break

    return changes_made


