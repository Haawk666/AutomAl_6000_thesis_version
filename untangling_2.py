
import logging
import graph_2
# Instantiate logger:
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_single_arcs(atomic_graph):
    single_arcs = []
    for vertex in atomic_graph.vertices:
        if not vertex.is_edge_column:
            for out_semi_partner in vertex.out_semi_partners:
                single_arc = graph_2.Arc(len(single_arcs), vertex, atomic_graph.vertices[out_semi_partner])
                single_arcs.append(single_arc)

    return single_arcs


def get_arc_centered_sub_graphs(atomic_graph, single_arcs):
    sub_graphs = []
    for single_arc in single_arcs:
        sub_graph = atomic_graph.get_arc_centered_subgraph(single_arc.vertex_a.i, single_arc.vertex_b.i)
        sub_graphs.append(sub_graph)

    determine_sub_graph_class(sub_graphs)
    determine_sub_graph_configuration(sub_graphs)

    return sub_graphs


def determine_sub_graph_class(sub_graphs):
    for sub_graph in sub_graphs:
        if sub_graph.meshes[0].order == 3 and sub_graph.meshes[1].order == 3:
            sub_graph.class_ = 1
        elif sub_graph.meshes[0].order == 4 and sub_graph.meshes[1].order == 3:
            sub_graph.class_ = 2
        elif sub_graph.meshes[0].order == 3 and sub_graph.meshes[1].order == 4:
            sub_graph.class_ = 3
        elif sub_graph.meshes[0].order == 3 and sub_graph.meshes[1].order == 5:
            sub_graph.class_ = 4
        elif sub_graph.meshes[0].order == 5 and sub_graph.meshes[1].order == 3:
            sub_graph.class_ = 5
        elif sub_graph.meshes[0].order == 4 and sub_graph.meshes[1].order == 4:
            sub_graph.class_ = 6


def determine_sub_graph_configuration(atomic_graph, sub_graphs):
    for sub_graph in sub_graphs:
        if sub_graph.class_ == 1:
            i = sub_graph.meshes[0].vertex_indices[0]
            j = sub_graph.meshes[0].vertex_indices[1]
            a = sub_graph.meshes[0].vertex_indices[2]
            b = sub_graph.meshes[1].vertex_indices[2]
            s = [(j, a), (a, i), (i, b), (b, j)]
            s_types = []

            for f in range(0, 4):
                if atomic_graph.vertices[s[f][0]].partner_query(s[f][1]):
                    if atomic_graph.vertices[s[f][1]].partner_query(s[f][0]):
                        s_types.append(0)
                    else:
                        s_types.append(1)
                else:
                    s_types.append(2)

            if s_types == [0, 0, 0, 0]:
                sub_graph.configuration = 'A_1'
            elif s_types == [0, 2, 0, 2]:
                sub_graph.configuration = 'B_1'
            elif s_types == [1, 0, 1, 0]:
                sub_graph.configuration = 'B_2'
            elif s_types == [0, 1, 0, 0]:
                sub_graph.configuration = 'C_1'
            elif s_types == [0, 0, 2, 0]:
                sub_graph.configuration = 'C_2'
            elif s_types == [0, 1, 1, 0]:
                sub_graph.configuration = 'D_1'
            elif s_types == [0, 2, 2, 0]:
                sub_graph.configuration = 'D_2'
            elif s_types == [0, 2, 0, 1]:
                sub_graph.configuration = 'E_1'
            elif s_types == [2, 0, 1, 0]:
                sub_graph.configuration = 'E_2'
            elif s_types == [0, 2, 0, 0]:
                sub_graph.configuration = 'F_1'
            elif s_types == [0, 0, 1, 0]:
                sub_graph.configuration = 'F_2'
            elif s_types == [0, 0, 0, 2]:
                sub_graph.configuration = 'G_1'
            elif s_types == [1, 0, 0, 0]:
                sub_graph.configuration = 'G_2'
            elif s_types == [2, 1, 0, 0]:
                sub_graph.configuration = 'H_1'
            elif s_types == [0, 0, 2, 1]:
                sub_graph.configuration = 'H_2'
            elif s_types == [0, 1, 2, 0]:
                sub_graph.configuration = 'I_1'

        elif sub_graph.class_ == 2:
            i = sub_graph.meshes[0].vertex_indices[0]
            j = sub_graph.meshes[0].vertex_indices[1]
            a = sub_graph.meshes[0].vertex_indices[2]
            b = sub_graph.meshes[0].vertex_indices[3]
            c = sub_graph.meshes[1].vertex_indices[2]
            s = [(j, a), (a, b), (b, i), (i, c), (c, j)]
            s_types = []

            for f in range(0, 5):
                if atomic_graph.vertices[s[f][0]].partner_query(s[f][1]):
                    if atomic_graph.vertices[s[f][1]].partner_query(s[f][0]):
                        s_types.append(0)
                    else:
                        s_types.append(1)
                else:
                    s_types.append(2)

            if s_types == [0, 0, 0, 0, 0]:
                sub_graph.configuration = 'A_1'
            elif s_types == [0, 1, 0, 0, 0]:
                sub_graph.configuration = 'B_1'
            elif s_types == [0, 1, 0, 1, 0]:
                sub_graph.configuration = 'C_1'
            elif s_types == [0, 0, 0, 0, 2]:
                sub_graph.configuration = 'D_1'
            elif s_types == [0, 0, 1, 0, 0]:
                sub_graph.configuration = 'E_1'

        elif sub_graph.class_ == 3:
            i = sub_graph.meshes[0].vertex_indices[0]
            j = sub_graph.meshes[0].vertex_indices[1]
            a = sub_graph.meshes[0].vertex_indices[2]
            b = sub_graph.meshes[1].vertex_indices[2]
            c = sub_graph.meshes[1].vertex_indices[3]
            s = [(j, a), (a, i), (i, b), (b, c), (c, j)]
            s_types = []

            for f in range(0, 5):
                if atomic_graph.vertices[s[f][0]].partner_query(s[f][1]):
                    if atomic_graph.vertices[s[f][1]].partner_query(s[f][0]):
                        s_types.append(0)
                    else:
                        s_types.append(1)
                else:
                    s_types.append(2)

            if s_types == [0, 0, 0, 0, 0]:
                sub_graph.configuration = 'A_1'
            elif s_types == [0, 0, 0, 2, 0]:
                sub_graph.configuration = 'B_1'
            elif s_types == [0, 2, 0, 2, 0]:
                sub_graph.configuration = 'C_1'
            elif s_types == [1, 0, 0, 0, 0]:
                sub_graph.configuration = 'D_1'
            elif s_types == [0, 0, 2, 0, 0]:
                sub_graph.configuration = 'E_1'

        elif sub_graph.class_ == 4:
            i = sub_graph.meshes[0].vertex_indices[0]
            j = sub_graph.meshes[0].vertex_indices[1]
            a = sub_graph.meshes[0].vertex_indices[2]
            b = sub_graph.meshes[1].vertex_indices[2]
            c = sub_graph.meshes[1].vertex_indices[3]
            d = sub_graph.meshes[1].vertex_indices[4]
            s = [(j, a), (a, i), (i, b), (b, c), (c, d), (d, j)]
            s_types = []

            for f in range(0, 6):
                if atomic_graph.vertices[s[f][0]].partner_query(s[f][1]):
                    if atomic_graph.vertices[s[f][1]].partner_query(s[f][0]):
                        s_types.append(0)
                    else:
                        s_types.append(1)
                else:
                    s_types.append(2)

            if s_types == [0, 0, 0, 0, 0, 0]:
                sub_graph.configuration = 'A_1'
            elif s_types == [0, 0, 0, 0, 2, 0]:
                sub_graph.configuration = 'B_1'

        elif sub_graph.class_ == 5:
            i = sub_graph.meshes[0].vertex_indices[0]
            j = sub_graph.meshes[0].vertex_indices[1]
            a = sub_graph.meshes[0].vertex_indices[2]
            b = sub_graph.meshes[0].vertex_indices[3]
            c = sub_graph.meshes[0].vertex_indices[4]
            d = sub_graph.meshes[1].vertex_indices[2]
            s = [(j, a), (a, b), (b, c), (c, i), (i, d), (d, j)]
            s_types = []

            for f in range(0, 6):
                if atomic_graph.vertices[s[f][0]].partner_query(s[f][1]):
                    if atomic_graph.vertices[s[f][1]].partner_query(s[f][0]):
                        s_types.append(0)
                    else:
                        s_types.append(1)
                else:
                    s_types.append(2)

            if s_types == [0, 0, 0, 0, 0, 0]:
                sub_graph.configuration = 'A_1'
            elif s_types == [0, 1, 0, 0, 0, 0]:
                sub_graph.configuration = 'B_1'

        elif sub_graph.class_ == 6:
            i = sub_graph.meshes[0].vertex_indices[0]
            j = sub_graph.meshes[0].vertex_indices[1]
            a = sub_graph.meshes[0].vertex_indices[2]
            b = sub_graph.meshes[0].vertex_indices[3]
            c = sub_graph.meshes[1].vertex_indices[2]
            d = sub_graph.meshes[1].vertex_indices[3]
            s = [(j, a), (a, b), (b, i), (i, c), (c, d), (d, j)]
            s_types = []

            for f in range(0, 6):
                if atomic_graph.vertices[s[f][0]].partner_query(s[f][1]):
                    if atomic_graph.vertices[s[f][1]].partner_query(s[f][0]):
                        s_types.append(0)
                    else:
                        s_types.append(1)
                else:
                    s_types.append(2)

            if s_types == [0, 0, 0, 0, 0, 0]:
                sub_graph.configuration = 'A_1'
            elif s_types == [0, 0, 0, 0, 0, 1]:
                sub_graph.configuration = 'B_1'
            elif s_types == [2, 0, 0, 0, 0, 0]:
                sub_graph.configuration = 'B_2'
            elif s_types == [2, 0, 0, 0, 0, 1]:
                sub_graph.configuration = 'C_1'
            elif s_types == [0, 0, 0, 2, 0, 0]:
                sub_graph.configuration = 'D_1'
            elif s_types == [0, 0, 1, 0, 0, 0]:
                sub_graph.configuration = 'D_2'
            elif s_types == [0, 0, 1, 2, 0, 0]:
                sub_graph.configuration = 'E_1'



