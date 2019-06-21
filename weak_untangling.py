import graph


def find_type(graph_obj, type_num):

    types = []
    classes = []

    for vertex in graph_obj.vertices:
        for partner in vertex.partners():

            if not graph_obj.vertices[partner].partner_query(vertex.i) and \
                    not vertex.is_edge_column and not graph_obj.vertices[partner].is_edge_column:

                print('Looking at {} {}'.format(vertex.i, partner))

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

                if type_num == 1 and mesh_1.num_corners == 3 and mesh_2.num_corners == 3:

                    print('Is type 1!')

                    config = mesh_1, mesh_2

                    class_ = find_type_1_class(graph_obj, config)

                    if not class_ == 'I':

                        types.append(config)
                        print('Fetching class')
                        classes.append(class_)
                        print('debug 1: {} {}, class: {}'.format(vertex.i, partner, class_))

                    else:

                        print('debug 2: {} {}'.format(vertex.i, partner))

                elif type_num == 2 and mesh_1.num_corners == 4 and mesh_2.num_corners == 3:

                    print('Is type 2!')

                    config = mesh_1, mesh_2

                    class_ = find_type_2_class(graph_obj, config)

                    if not class_ == 'I':

                        types.append(config)
                        print('Fetching class')
                        classes.append(class_)
                        print('debug 1: {} {}'.format(vertex.i, partner))

                    else:

                        print('debug 2: {} {}'.format(vertex.i, partner))

                elif type_num == 3 and mesh_1.num_corners == 3 and mesh_2.num_corners == 4:

                    print('Is type 3!')

                    config = mesh_1, mesh_2

                    class_ = find_type_3_class(graph_obj, config)

                    if not class_ == 'I':

                        types.append(config)
                        print('Fetching class')
                        classes.append(class_)
                        print('debug 1: {} {}'.format(vertex.i, partner))

                    else:

                        print('debug 2: {} {}'.format(vertex.i, partner))

                elif type_num == 4 and mesh_1.num_corners == 3 and mesh_2.num_corners == 5:

                    print('Is type 4')

                    config = mesh_1, mesh_2

                    class_ = find_type_4_class(graph_obj, config)

                    if not class_ == 'B':

                        types.append(config)
                        print('Fetching class')
                        classes.append(class_)
                        print('debug 1: {} {}'.format(vertex.i, partner))

                    else:

                        print('debug 2: {} {}'.format(vertex.i, partner))

                elif type_num == 5 and mesh_1.num_corners == 5 and mesh_2.num_corners == 3:

                    print('Is type 5')

                    config = mesh_1, mesh_2

                    class_ = find_type_5_class(graph_obj, config)

                    if not class_ == 'B':

                        types.append(config)
                        print('Fetching class')
                        classes.append(class_)
                        print('debug 1: {} {}'.format(vertex.i, partner))

                    else:

                        print('debug 2: {} {}'.format(vertex.i, partner))

                else:

                    print('Hitler')

    return types, classes


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
    elif s_types == [2, 1, 0, 0]:
        return 'H_1'
    elif s_types == [0, 0, 2, 1]:
        return 'H_2'
    else:
        return 'I'


def find_type_2_class(graph_obj, config):

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

    # Rewrite as tree??:

    if s_types == [0, 0, 0, 0, 0]:
        return 'A'
    elif s_types == [0, 1, 0, 0, 0]:
        return 'B_1'
    else:
        return 'C'


def find_type_3_class(graph_obj, config):

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

    # Rewrite as tree??:

    if s_types == [0, 0, 0, 0, 0]:
        return 'A'
    elif s_types == [0, 0, 0, 2, 0]:
        return 'B_1'
    else:
        return 'C'


def find_type_4_class(graph_obj, config):

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
        return 'A'
    else:
        print('Hello dipshit!')
        return 'B'


def find_type_5_class(graph_obj, config):
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
        return 'A'
    else:
        print('Hello dippshit!')
        return 'B'


def process_type_1(graph_obj):

    print('Finding type 1\'s')
    type_1, classes = find_type(graph_obj, 1)
    print('Found types!')
    changes = 0

    for class_, config in zip(classes, type_1):

        i = config[0].vertex_indices[0]
        j = config[0].vertex_indices[1]
        a = config[0].vertex_indices[2]
        b = config[1].vertex_indices[2]

        if class_ == 'A':

            print('try: {} {}'.format(i, j))
            if graph_obj.weak_remove_edge(i, j):
                print('    success!')
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

        elif class_ == 'H_1':

            if graph_obj.vertices[i].partner_query(j):
                graph_obj.perturb_j_k(i, j, a)
                changes += 1

        elif class_ == 'H_2':

            if graph_obj.vertices[i].partner_query(j):
                graph_obj.perturb_j_k(i, j, b)
                changes += 1

        else:

            pass

    return len(classes), changes


def process_type_2(graph_obj):

    print('Finding type 2\'s')
    type_2, classes = find_type(graph_obj, 2)
    print('Found types!')
    changes = 0

    for class_, config in zip(classes, type_2):

        i = config[0].vertex_indices[0]
        j = config[0].vertex_indices[1]
        a = config[0].vertex_indices[2]
        b = config[0].vertex_indices[3]
        c = config[1].vertex_indices[2]

        if class_ == 'A':

            pass

        elif class_ == 'B_1':

            if graph_obj.vertices[i].partner_query(j) and graph_obj.vertices[a].partner_query(b):
                graph_obj.perturb_j_k(i, j, a)
                graph_obj.perturb_j_k(a, b, i)
                changes += 1

        else:

            pass

    return len(classes), changes


def process_type_3(graph_obj):

    print('Finding type 3\'s')
    type_3, classes = find_type(graph_obj, 3)
    print('Found types!')
    changes = 0

    for class_, config in zip(classes, type_3):

        i = config[0].vertex_indices[0]
        j = config[0].vertex_indices[1]
        a = config[0].vertex_indices[2]
        b = config[1].vertex_indices[2]
        c = config[1].vertex_indices[3]

        if class_ == 'A':

            pass

        elif class_ == 'B_1':

            if graph_obj.vertices[i].partner_query(j) and graph_obj.vertices[c].partner_query(b):
                graph_obj.perturb_j_k(i, j, c)
                graph_obj.perturb_j_k(c, b, i)
                changes += 1

        else:

            pass

    return len(classes), changes


def process_type_4(graph_obj):

    print('Finding type 4\'s')
    type_4, classes = find_type(graph_obj, 4)
    print('Found types!')
    changes = 0

    for class_, config in zip(classes, type_4):

        i = config[0].vertex_indices[0]
        j = config[0].vertex_indices[1]
        a = config[0].vertex_indices[2]
        b = config[1].vertex_indices[2]
        c = config[1].vertex_indices[3]
        d = config[1].vertex_indices[4]

        if class_ == 'A':

            if graph_obj.vertices[i].partner_query(j):
                graph_obj.perturb_j_k(i, j, d)
                changes += 1

        else:

            pass

    return len(classes), changes


def process_type_5(graph_obj):

    print('Finding type 5\'s')
    type_5, classes = find_type(graph_obj, 5)
    print('Found types: {}!'.format(len(classes)))
    changes = 0

    for class_, config in zip(classes, type_5):

        i = config[0].vertex_indices[0]
        j = config[0].vertex_indices[1]
        a = config[0].vertex_indices[2]
        b = config[0].vertex_indices[3]
        c = config[0].vertex_indices[4]
        d = config[1].vertex_indices[2]

        if class_ == 'A':

            if graph_obj.vertices[i].partner_query(j):
                graph_obj.perturb_j_k(i, j, a)
                changes += 1

        else:

            pass

    return len(classes), changes

