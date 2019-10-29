# Program imports:
import core
import utils
# External imports:
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import logging
# Instantiate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def calculate_params(files, exclude_edges=True, exclude_matrix=False, exclude_particle=False, exclude_hidden=False,
                     exclude_1=False, exclude_2=False, exclude_3=False, exclude_4=False, plot=False):

    # Accumulate data:
    logger.info('Accumulating data...')
    values = []
    for i in range(0, 7):
        values.append([])
        for j in range(0, 7):
            values[i].append([])
    number_of_files = 0
    number_of_vertices = 0
    for file_ in iter(files.splitlines()):
        instance = core.SuchSoftware.load(file_)
        number_of_files += 1

        instance.calc_avg_gamma()
        instance.normalize_gamma()
        instance.graph.map_friends()
        instance.graph.sort_subsets_by_distance()

        for vertex in instance.graph.vertices:
            if not (exclude_edges and vertex.is_edge_column):
                if not (exclude_matrix and not vertex.is_in_precipitate):
                    if not (exclude_particle and vertex.is_in_precipitate):
                        if not (exclude_hidden and not vertex.show_in_overlay):
                            if not (exclude_1 and vertex.flag_1):
                                if not (exclude_2 and vertex.flag_2):
                                    if not (exclude_3 and vertex.flag_3):
                                        if not (exclude_4 and vertex.flag_4):

                                            number_of_vertices += 1

                                            peak_gamma = vertex.normalized_peak_gamma
                                            avg_gamma = vertex.normalized_avg_gamma
                                            alpha = instance.graph.produce_alpha_angles(vertex.i)
                                            theta = instance.graph.produce_theta_angles(vertex.i, exclude_angles_from_inconsistent_meshes=True)
                                            alpha_max = max(alpha)
                                            alpha_min = min(alpha)
                                            if theta:
                                                theta_max = max(theta)
                                                theta_min = min(theta)
                                                theta_mean = utils.mean_val(theta)
                                            else:
                                                theta_max = 0
                                                theta_min = 0
                                                theta_mean = 0

                                            index = -1

                                            if vertex.species() == 'Cu':
                                                index = 0

                                            elif vertex.species() == 'Si':
                                                if vertex.flag_2:
                                                    index = 1
                                                else:
                                                    index = 2

                                            elif vertex.species() == 'Al':
                                                if vertex.is_in_precipitate:
                                                    index = 3
                                                else:
                                                    index = 4

                                            elif vertex.species() == 'Mg':
                                                if alpha_max <= 3.15:
                                                    index = 5
                                                else:
                                                    index = 6

                                            values[index][0].append(alpha_min)
                                            values[index][1].append(alpha_max)
                                            values[index][2].append(theta_min)
                                            values[index][3].append(theta_max)
                                            values[index][4].append(theta_mean)
                                            values[index][5].append(peak_gamma)
                                            values[index][6].append(avg_gamma)

    # Calculate model parameters:
    logger.info('Calculating model parameters...')
    params = []
    for i in range(0, 7):
        params.append([])

    for i, elements in enumerate(values):
        for j, data in enumerate(elements):
            if j == 4:
                params[i].append((utils.mean_val(data), 0.1))
            else:
                params[i].append((utils.mean_val(data), np.sqrt(utils.variance(data))))

    # Plot results:
    if plot:
        plot_pane(params)

    return params


def plot_pane(params):
    logger.info('Generating plots...')

    alpha = [
        np.linspace(0.5, 2.5, 1000),
        np.linspace(1.5, 4.5, 1000),
        np.linspace(0.5, 2.5, 1000),
        np.linspace(0.5, 3.5, 1000),
        np.linspace(0.5, 2.5, 1000),
        np.linspace(0, 1, 1000),
        np.linspace(0, 0.8, 1000)
    ]

    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(3, 3, figure=fig)
    ax = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, 1])
    ]

    for i, axis in enumerate(ax):
        axis.plot(alpha[i], utils.normal_dist(alpha[i], params[0][i][0], params[0][i][1]), 'y',
                  label='Cu ($\mu = ${:.2f}, $\sigma = ${:.2f})'.format(params[0][i][0], params[0][i][1]))
        axis.plot(alpha[i], utils.normal_dist(alpha[i], params[1][i][0], params[1][i][1]), 'r',
                  label='Si$_1$ ($\mu = ${:.2f}, $\sigma = ${:.2f})'.format(params[1][i][0], params[1][i][1]))
        axis.plot(alpha[i], utils.normal_dist(alpha[i], params[2][i][0], params[2][i][1]), 'k',
                  label='Si$_2$ ($\mu = ${:.2f}, $\sigma = ${:.2f})'.format(params[2][i][0], params[2][i][1]))
        axis.plot(alpha[i], utils.normal_dist(alpha[i], params[3][i][0], params[3][i][1]), 'g',
                  label='Al$_1$ ($\mu = ${:.2f}, $\sigma = ${:.2f})'.format(params[3][i][0], params[3][i][1]))
        axis.plot(alpha[i], utils.normal_dist(alpha[i], params[4][i][0], params[4][i][1]), 'b',
                  label='Al$_2$ ($\mu = ${:.2f}, $\sigma = ${:.2f})'.format(params[4][i][0], params[4][i][1]))
        axis.plot(alpha[i], utils.normal_dist(alpha[i], params[5][i][0], params[5][i][1]), 'm',
                  label='Mg$_1$ ($\mu = ${:.2f}, $\sigma = ${:.2f})'.format(params[5][i][0], params[5][i][1]))
        axis.plot(alpha[i], utils.normal_dist(alpha[i], params[6][i][0], params[6][i][1]), 'c',
                  label='Mg$_2$ ($\mu = ${:.2f}, $\sigma = ${:.2f})'.format(params[6][i][0], params[6][i][1]))

    ax[0].set_title('Minimum alpha angles fitted density')
    ax[0].set_xlabel('min angle (radians)')
    ax[0].legend()

    ax[1].set_title('Maximum alpha angles fitted density')
    ax[1].set_xlabel('max angle (radians)')
    ax[1].legend()

    ax[2].set_title('Minimum theta angles fitted density')
    ax[2].set_xlabel('min angle (radians)')
    ax[2].legend()

    ax[3].set_title('Maximum theta angles fitted density')
    ax[3].set_xlabel('max angle (radians)')
    ax[3].legend()

    ax[4].set_title('Theta mean fitted density')
    ax[4].set_xlabel('mean angle (radians)')
    ax[4].legend()

    ax[5].set_title('Normalized peak gamma fitted density')
    ax[5].set_xlabel('peak gamma (1)')
    ax[5].legend()

    ax[6].set_title('Normalized average gamma fitted density')
    ax[6].set_xlabel('average gamma (1)')
    ax[6].legend()

    fig.suptitle('98 parameter model fitted densities from {} images'.format(number_of_files))

    logger.info(
        'Plotted parameters over {} files and {} vertices!'.format(number_of_files, number_of_vertices))

    plt.show()


def produce_params(calc=False):
    if calc:
        path = 'C:\\Users\\haakot\\OneDrive\\NTNU\\TFY4900 Master\\Data_a\\'
        params = calculate_params('{}008_control\n{}012a_control\n{}023_control\n{}030_control\n{}Smart_aligned_Qprime_control\n{}Small_Qprime_control'.replace('{}', path),
                                  plot=True)
    else:
        params = [
            [(1.94, 0.15), (2.25, 0.20), (1.94, 0.16), (2.19, 0.13), (2.07, 0.2), (0.52, 0.22), (0.48, 0.14)],
            [(1.60, 0.13), (2.45, 0.15), (1.61, 0.14), (2.43, 0.13), (2.09, 0.2), (0.33, 0.06), (0.32, 0.04)],
            [(1.94, 0.10), (2.26, 0.11), (1.93, 0.12), (2.26, 0.11), (2.09, 0.2), (0.32, 0.07), (0.32, 0.05)],
            [(1.46, 0.13), (3.13, 0.21), (1.37, 0.13), (1.76, 0.14), (1.57, 0.2), (0.29, 0.13), (0.31, 0.08)],
            [(1.54, 0.13), (3.11, 0.08), (1.48, 0.06), (1.66, 0.06), (1.57, 0.2), (0.30, 0.04), (0.31, 0.03)],
            [(1.25, 0.11), (2.63, 0.11), (1.13, 0.06), (1.42, 0.10), (1.26, 0.2), (0.19, 0.09), (0.24, 0.05)],
            [(1.23, 0.07), (3.72, 0.13), (1.10, 0.07), (1.43, 0.11), (1.26, 0.2), (0.14, 0.12), (0.22, 0.06)]
        ]
    return params

