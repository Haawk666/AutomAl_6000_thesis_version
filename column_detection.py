# Internal imports:
import core
import utils
# External imports:
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import copy
import time
import logging
# Instantiate logger:
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def column_detection(project, search_mode='t', plot=False):

    peak_values = []
    non_edge_peak_values = []
    if len(project.graph.vertices) == 0:
        logger.info('Starting column detection. Search mode is \'{}\''.format(search_mode))
        project.search_mat = copy.deepcopy(project.im_mat)
        cont = True
    else:
        logger.info('Continuing column detection. Search mode is \'{}\''.format(search_mode))
        min_val = 2
        for vertex in project.graph.vertices:
            if vertex.peak_gamma < min_val:
                min_val = vertex.peak_gamma
        if min_val < project.threshold:
            new_graph = core.graph_2.AtomicGraph(
                project.scale,
                active_model=project.graph.active_model,
                species_dict=project.graph.species_dict
            )
            project.num_columns = 0
            project.search_mat = copy.deepcopy(project.im_mat)
            for vertex in project.graph.get_non_void_vertices():
                if vertex.peak_gamma > project.threshold:
                    project.num_columns += 1
                    peak_values.append(vertex.peak_gamma)
                    new_graph.add_vertex(vertex)
                    project.search_mat = utils.delete_pixels(
                        project.search_mat,
                        int(vertex.im_coor_x),
                        int(vertex.im_coor_y),
                        project.r + project.overhead
                    )
            project.graph = new_graph
            cont = False
        else:
            project.redraw_search_mat()
            for vertex in project.graph.vertices:
                peak_values.append(vertex.peak_gamma)
            cont = True

    counter = project.num_columns

    while cont:

        pos = np.unravel_index(project.search_mat.argmax(), (project.im_height, project.im_width))
        max_val = project.search_mat[pos]
        peak_values.append(max_val)

        x_fit, y_fit = utils.cm_fit(project.im_mat, pos[1], pos[0], project.r)
        x_fit_int = int(x_fit)
        y_fit_int = int(y_fit)

        project.search_mat = utils.delete_pixels(project.search_mat, x_fit_int, y_fit_int, project.r + project.overhead)

        vertex = core.graph_2.Vertex(counter, x_fit, y_fit, project.r, project.scale, parent_graph=project.graph)
        vertex.avg_gamma, vertex.peak_gamma = utils.circular_average(project.im_mat, x_fit_int, y_fit_int, project.r)
        if not max_val == vertex.peak_gamma:
            logger.debug(
                'Vertex {}\n    Pos: ({}, {})\n    Fit: ({}, {})\n    max_val: {}\n    peak_gamma: {}\n'.format(
                    counter,
                    pos[1],
                    pos[0],
                    x_fit,
                    y_fit,
                    max_val,
                    vertex.peak_gamma
                )
            )
        project.graph.add_vertex(vertex)

        project.num_columns += 1
        counter += 1

        if search_mode == 's':
            if counter >= project.search_size:
                cont = False
        elif search_mode == 't':
            if np.max(project.search_mat) < project.threshold:
                cont = False
        else:
            logger.error('Invalid search_type')

    project.summarize_stats()

    project.find_edge_columns()
    for vertex in project.graph.vertices:
        if not vertex.is_edge_column:
            non_edge_peak_values.append(vertex.peak_gamma)

    logger.info('Column detection complete! Found {} columns.'.format(project.num_columns))

    if plot:
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(4, 1, figure=fig)
        ax_values = fig.add_subplot(gs[0, 0])
        ax_slope = fig.add_subplot(gs[1, 0])
        ax_cum_var = fig.add_subplot(gs[2, 0])
        ax_cum_var_slope = fig.add_subplot(gs[3, 0])

        ax_values.plot(
            range(0, len(peak_values)),
            peak_values,
            c='k',
            label='Column peak intensity'
        )
        ax_values.set_title('Column detection summary')
        ax_values.set_xlabel('# Column')
        ax_values.set_ylabel('Peak intensity')
        ax_values.legend()

        slope = [0]
        for ind in range(1, len(peak_values)):
            slope.append(peak_values[ind] - peak_values[ind - 1])

        ax_slope.plot(
            range(0, len(peak_values)),
            slope,
            c='b',
            label='2 point slope of peak intensity'
        )
        ax_slope.set_title('Slope')
        ax_slope.set_xlabel('# Column')
        ax_slope.set_ylabel('slope')
        ax_slope.legend()

        cumulative_slope_variance = [0]
        cum_slp_var_slp = [0]  # lol
        for ind in range(1, len(peak_values)):
            cumulative_slope_variance.append(utils.variance(slope[0:ind]))
            cum_slp_var_slp.append(cumulative_slope_variance[ind] - cumulative_slope_variance[ind - 1])

        ax_cum_var.plot(
            range(0, len(non_edge_peak_values)),
            non_edge_peak_values,
            c='r',
            label='None edge peak values'
        )
        ax_cum_var.set_title('Cumulative variance')
        ax_cum_var.set_xlabel('# Column')
        ax_cum_var.set_ylabel('$\\sigma^2$')
        ax_cum_var.legend()

        ax_cum_var_slope.plot(
            range(0, len(peak_values)),
            cum_slp_var_slp,
            c='g',
            label='Slope of cumulative slope variance'
        )
        ax_cum_var_slope.set_title('Variance slope')
        ax_cum_var_slope.set_xlabel('# Column')
        ax_cum_var_slope.set_ylabel('2-point slope approximation')
        ax_cum_var_slope.legend()

        inflection_points = []
        for ind in range(1, len(cum_slp_var_slp)):
            if cum_slp_var_slp[ind] > cum_slp_var_slp[ind - 1]:
                inflection_points.append(ind)

        plt.show()













