# Internal imports:
# External imports:
import numpy as np
import time
import logging
# Instantiate logger:
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def map_districts(graph_obj, district_size=8, method='matrix'):
    if method == 'matrix':
        time_1 = time.time()
        matrix = np.zeros([graph_obj.order, graph_obj.order], dtype=float)
        for i in range(0, len(graph_obj.vertices)):
            for j in range(i, len(graph_obj.vertices)):
                if i == j:
                    matrix[j, i] = 0
                else:
                    dist = graph_obj.get_projected_separation(i, j)
                    matrix[j, i] = dist
                    matrix[i, j] = dist
        time_2 = time.time()
        for vertex in graph_obj.vertices:
            vertex.district = np.argsort(matrix[vertex.i, :])[1:district_size + 1]
        time_3 = time.time()
        summary_string = 'Districts mapped in {} seconds with matrix method.\n'.format(time_3 - time_1)
        summary_string += '    Distance calculations took {} seconds.\n'.format(time_2 - time_1)
        summary_string += '    Sorting took {} seconds.'.format(time_3 - time_2)
        logger.info(summary_string)




