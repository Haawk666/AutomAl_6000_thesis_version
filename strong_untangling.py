import graph


def level_1_untangling(graph_obj):

    changes = 0

    for vertex in graph_obj.vertices:

        if not vertex.is_edge_column:

            sub_graph = graph_obj.get_atomic_configuration(vertex.i)

            for m, mesh in enumerate(sub_graph.meshes):

                if mesh.num_corners == 3 and sub_graph.meshes[m - 1].num_corners == 3:

                    j = mesh.vertex_indices[1]

                    if vertex.partner_query(j):

                        for k in sub_graph.configuration_partners():

                            if k not in vertex.partners():

                                graph_obj.perturb_j_k(vertex.i, j, k)
                                changes += 1
                                break

                        else:

                            if graph_obj.strong_remove_edge(vertex.i, j):
                                changes += 1

    return changes

