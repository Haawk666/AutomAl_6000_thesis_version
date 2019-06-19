import graph


def find_type_1(graph_obj):

    type_1 = []
    classes = []

    for vertex in graph_obj.vertices:
        for partner in vertex.partners():

            if not graph_obj.vertices[partner].partner_query(vertex.i):

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

                if mesh_1.num_corners == 3 and mesh_2.num_corners == 3:

                    # Is type 1!

                    config = mesh_1, mesh_2

                    class_ = find_type_1_class(graph_obj, config)

                    if not class_ == 'H':

                        type_1.append(config)
                        classes.append(class_)

    return type_1, classes


def find_type_1_class(graph_obj, config):

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

    # Rewrite as tree??:

    if s_types == [0, 0, 0, 0]:
        return 'A'
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
    else:
        return 'H'


def process_type_1(graph_obj):

    type_1, classes = find_type_1(graph_obj)
    changes = 0

    for class_, config in zip(classes, type_1):

        i = config[0].vertex_indices[0]
        j = config[0].vertex_indices[1]
        a = config[0].vertex_indices[2]
        b = config[1].vertex_indices[2]

        if class_ == 'A':

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

        elif class_ == 'F_1':

            pass

        elif class_ == 'F_2':

            pass

        elif class_ == 'G_1':

            if graph_obj.vertices[i].partner_query(j):
                graph_obj.perturb_j_k(j, b, i)
                changes += 1

        elif class_ == 'G_2':

            if graph_obj.vertices[i].partner_query(j):
                graph_obj.perturb_j_k(j, a, i)
                changes += 1

        else:

            pass

    return len(classes), changes



