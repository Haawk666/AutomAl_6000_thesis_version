import core
from matplotlib import pyplot as plt
import graph_op
import numpy as np
import utils
from matplotlib.gridspec import GridSpec


def set_mg_type(graph_obj):

    for k, vertex in enumerate(graph_obj.vertices):

        max_, min_, *_ = graph_op.base_angle_score(graph_obj, vertex.i, apply=False)

        if vertex.species() == 'Mg':

            if max_ > 3.15:

                graph_obj.vertices[k].flag_2 = True

            else:

                graph_obj.vertices[k].flag_2 = False


def set_si_type(graph_obj):

    pass


def accumulate_statistics():

    cu_min_angles = []
    si_1_min_angles = []
    si_2_min_angles = []
    al_min_angles = []
    mg_1_min_angles = []
    mg_2_min_angles = []

    cu_max_angles = []
    si_1_max_angles = []
    si_2_max_angles = []
    al_max_angles = []
    mg_1_max_angles = []
    mg_2_max_angles = []

    cu_intensities = []
    si_intensities = []
    al_intensities = []
    mg_intensities = []

    number_of_files = 0
    number_of_vertices = 0

    with open('Saves/validation_set/filenames_control.txt', mode='r') as f:
        for line in f:
            line = line.replace('\n', '')
            filename = line
            filename = 'Saves/validation_set/' + filename
            number_of_files += 1

            file = core.SuchSoftware.load(filename)
            set_mg_type(file.graph)
            number_of_vertices += file.num_columns

            cu_min_angles, si_1_min_angles, si_2_min_angles, al_min_angles, mg_1_min_angles, mg_2_min_angles, cu_max_angles, si_1_max_angles, si_2_max_angles,\
            al_max_angles, mg_1_max_angles, mg_2_max_angles, cu_intensities, si_intensities, al_intensities, mg_intensities = \
                accumulate_test_data(file, cu_min_angles, si_1_min_angles, si_2_min_angles, al_min_angles, mg_1_min_angles, mg_2_min_angles, cu_max_angles, si_1_max_angles, si_2_max_angles,
            al_max_angles, mg_1_max_angles, mg_2_max_angles, cu_intensities, si_intensities, al_intensities, mg_intensities)

    plot_test_data(cu_min_angles, si_1_min_angles, si_2_min_angles, al_min_angles, mg_1_min_angles, mg_2_min_angles, cu_max_angles, si_1_max_angles, si_2_max_angles,
            al_max_angles, mg_1_max_angles, mg_2_max_angles, cu_intensities, si_intensities, al_intensities, mg_intensities,
                                title='Summery of validation set. {} columns over {} images'.format(number_of_vertices,
                                                                                                    number_of_files))

    # plot_test_data_2(cu_min_angles, si_1_min_angles, si_2_min_angles, al_min_angles, mg_1_min_angles, mg_2_min_angles,
    #                cu_max_angles, si_1_max_angles, si_2_max_angles,
    #                al_max_angles, mg_1_max_angles, mg_2_max_angles, cu_intensities, si_intensities, al_intensities,
    #                mg_intensities,
    #                title='Summery of validation set. {} columns over {} images'.format(number_of_vertices,
    #                                                                                    number_of_files))


def accumulate_test_data(obj, cu_min_angles, si_1_min_angles, si_2_min_angles, al_min_angles, mg_1_min_angles, mg_2_min_angles, cu_max_angles, si_1_max_angles, si_2_max_angles,
            al_max_angles, mg_1_max_angles, mg_2_max_angles, cu_intensities, si_intensities, al_intensities, mg_intensities):
    for vertex in obj.graph.vertices:

        if not vertex.is_edge_column:

            max_, min_, *_ = graph_op.base_angle_score(obj.graph, vertex.i, apply=False)

            if vertex.species() == 'Cu':
                cu_min_angles.append(min_)
                cu_max_angles.append(max_)
                cu_intensities.append(vertex.peak_gamma)

            elif vertex.species() == 'Si':
                si_intensities.append(vertex.peak_gamma)
                if vertex.flag_2:
                    si_2_min_angles.append(min_)
                    si_2_max_angles.append(max_)
                else:
                    si_1_min_angles.append(min_)
                    si_1_max_angles.append(max_)

            elif vertex.species() == 'Al':
                al_min_angles.append(min_)
                al_max_angles.append(max_)
                al_intensities.append(vertex.peak_gamma)

            elif vertex.species() == 'Mg':
                mg_intensities.append(vertex.peak_gamma)
                si_intensities.append(vertex.peak_gamma)
                if vertex.flag_2:
                    mg_2_min_angles.append(min_)
                    mg_2_max_angles.append(max_)
                else:
                    mg_1_min_angles.append(min_)
                    mg_1_max_angles.append(max_)

            else:

                print('Error')

    return cu_min_angles, si_1_min_angles, si_2_min_angles, al_min_angles, mg_1_min_angles, mg_2_min_angles, cu_max_angles, si_1_max_angles, si_2_max_angles,\
            al_max_angles, mg_1_max_angles, mg_2_max_angles, cu_intensities, si_intensities, al_intensities, mg_intensities


def plot_test_data(cu_min_angles, si_1_min_angles, si_2_min_angles, al_min_angles, mg_1_min_angles, mg_2_min_angles, cu_max_angles, si_1_max_angles, si_2_max_angles,
            al_max_angles, mg_1_max_angles, mg_2_max_angles, cu_intensities, si_intensities, al_intensities, mg_intensities, title='Statistical summary'):

    cu_min_std = np.sqrt(utils.variance(cu_min_angles))
    cu_max_std = np.sqrt(utils.variance(cu_max_angles))
    si_1_min_std = np.sqrt(utils.variance(si_1_min_angles))
    si_1_max_std = np.sqrt(utils.variance(si_1_max_angles))
    si_2_min_std = np.sqrt(utils.variance(si_2_min_angles))
    si_2_max_std = np.sqrt(utils.variance(si_2_max_angles))
    al_min_std = np.sqrt(utils.variance(al_min_angles))
    al_max_std = np.sqrt(utils.variance(al_max_angles))
    mg_1_min_std = np.sqrt(utils.variance(mg_1_min_angles))
    mg_1_max_std = np.sqrt(utils.variance(mg_1_max_angles))
    mg_2_min_std = np.sqrt(utils.variance(mg_2_min_angles))
    mg_2_max_std = np.sqrt(utils.variance(mg_2_max_angles))

    std_min = [cu_min_std, si_1_min_std, si_2_min_std, al_min_std, mg_1_min_std, mg_2_min_std]
    std_max = [cu_max_std, si_1_max_std, si_2_max_std, al_max_std, mg_1_max_std, mg_2_max_std]

    cu_min_mean = utils.mean_val(cu_min_angles)
    cu_max_mean = utils.mean_val(cu_max_angles)
    si_1_min_mean = utils.mean_val(si_1_min_angles)
    si_1_max_mean = utils.mean_val(si_1_max_angles)
    si_2_min_mean = utils.mean_val(si_2_min_angles)
    si_2_max_mean = utils.mean_val(si_2_max_angles)
    al_min_mean = utils.mean_val(al_min_angles)
    al_max_mean = utils.mean_val(al_max_angles)
    mg_1_min_mean = utils.mean_val(mg_1_min_angles)
    mg_1_max_mean = utils.mean_val(mg_1_max_angles)
    mg_2_min_mean = utils.mean_val(mg_2_min_angles)
    mg_2_max_mean = utils.mean_val(mg_2_max_angles)

    mean_min = [cu_min_mean, si_1_min_mean, si_2_min_mean, al_min_mean, mg_1_min_mean, mg_2_min_mean]
    mean_max = [cu_max_mean, si_1_max_mean, si_2_max_mean, al_max_mean, mg_1_max_mean, mg_2_max_mean]

    cu_gamma_std = np.sqrt(utils.variance(cu_intensities))
    si_gamma_std = np.sqrt(utils.variance(si_intensities))
    al_gamma_std = np.sqrt(utils.variance(al_intensities))
    mg_gamma_std = np.sqrt(utils.variance(mg_intensities))

    cu_gamma_mean = utils.mean_val(cu_intensities)
    si_gamma_mean = utils.mean_val(si_intensities)
    al_gamma_mean = utils.mean_val(al_intensities)
    mg_gamma_mean = utils.mean_val(mg_intensities)

    alpha = np.linspace(1, 4, 1000)
    beta = np.linspace(1, 2.5, 1000)
    gamma = np.linspace(0, 1, 1000)

    const_1_cu = [0.8] * len(cu_max_angles)
    const_1_si_1 = [0.7] * len(si_1_max_angles)
    const_1_si_2 = [0.6] * len(si_2_max_angles)
    const_1_al = [0.5] * len(al_max_angles)
    const_1_mg_1 = [0.4] * len(mg_1_max_angles)
    const_1_mg_2 = [0.3] * len(mg_2_max_angles)

    const_2_cu = [2.0] * len(cu_min_angles)
    const_2_si_1 = [1.9] * len(si_1_min_angles)
    const_2_si_2 = [1.8] * len(si_2_min_angles)
    const_2_al = [1.7] * len(al_min_angles)
    const_2_mg_1 = [1.6] * len(mg_1_min_angles)
    const_2_mg_2 = [1.5] * len(mg_2_min_angles)

    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    ax_min = fig.add_subplot(gs[0, 0])
    ax_max = fig.add_subplot(gs[1, 0])
    ax_co = fig.add_subplot(gs[2, 0])
    ax_scatter = fig.add_subplot(gs[:, 1])

    ax_min.plot(alpha, utils.normal_dist(alpha, cu_min_mean, cu_min_std), 'y',
                label='Cu ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(cu_min_mean, cu_min_std))
    ax_min.plot(alpha, utils.normal_dist(alpha, si_1_min_mean, si_1_min_std), 'r',
                label='Si$_1$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(si_1_min_mean, si_1_min_std))
    ax_min.plot(alpha, utils.normal_dist(alpha, si_2_min_mean, si_2_min_std), 'k',
                label='Si$_2$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(si_2_min_mean, si_2_min_std))
    ax_min.plot(alpha, utils.normal_dist(alpha, al_min_mean, al_min_std), 'g',
                label='Al ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(al_min_mean, al_min_std))
    ax_min.plot(alpha, utils.normal_dist(alpha, mg_1_min_mean, mg_1_min_std), 'm',
                label='Mg$_1$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(mg_1_min_mean, mg_1_min_std))
    ax_min.plot(alpha, utils.normal_dist(alpha, mg_2_min_mean, mg_2_min_std), 'c',
                label='Mg$_2$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(mg_2_min_mean, mg_2_min_std))

    ax_min.set_title('Minimum central angles fitted density')
    ax_min.set_xlabel('Min angle (radians)')
    ax_min.legend()

    ax_max.plot(alpha, utils.normal_dist(alpha, cu_max_mean, cu_max_std), 'y',
                label='Cu ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(cu_max_mean, cu_max_std))
    ax_max.plot(alpha, utils.normal_dist(alpha, si_1_max_mean, si_1_max_std), 'r',
                label='Si$_1$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(si_1_max_mean, si_1_max_std))
    ax_max.plot(alpha, utils.normal_dist(alpha, si_2_max_mean, si_2_max_std), 'k',
                label='Si$_2$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(si_2_max_mean, si_2_max_std))
    ax_max.plot(alpha, utils.normal_dist(alpha, al_max_mean, al_max_std), 'g',
                label='Al ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(al_max_mean, al_max_std))
    ax_max.plot(alpha, utils.normal_dist(alpha, mg_1_max_mean, mg_1_max_std), 'm',
                label='Mg$_1$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(mg_1_max_mean, mg_1_max_std))
    ax_max.plot(alpha, utils.normal_dist(alpha, mg_2_max_mean, mg_2_max_std), 'c',
                label='Mg$_2$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(mg_2_max_mean, mg_2_max_std))

    ax_max.set_title('Maximum central angles fitted density')
    ax_max.set_xlabel('max angle (radians)')
    ax_max.legend()

    ax_co.plot(gamma, utils.normal_dist(gamma, cu_gamma_mean, cu_gamma_std), 'y',
               label='Cu')
    ax_co.plot(gamma, utils.normal_dist(gamma, si_gamma_mean, si_gamma_std), 'r',
               label='Si')
    ax_co.plot(gamma, utils.normal_dist(gamma, al_gamma_mean, al_gamma_std), 'g',
               label='Al')
    ax_co.plot(gamma, utils.normal_dist(gamma, mg_gamma_mean, mg_gamma_std), 'm',
               label='Mg')

    ax_co.set_title('peak z-contrast distributions')
    ax_co.set_xlabel('z-contrast (normalized $\in (0, 1)$)')
    ax_co.legend()

    ax_scatter.scatter(cu_min_angles, cu_max_angles, c='y', label='Cu', s=8)
    ax_scatter.scatter(si_1_min_angles, si_1_max_angles, c='r', label='Si$_1$', s=8)
    ax_scatter.scatter(si_2_min_angles, si_2_max_angles, c='k', label='Si$_2$', s=8)
    ax_scatter.scatter(al_min_angles, al_max_angles, c='g', label='Al', s=8)
    ax_scatter.scatter(mg_1_min_angles, mg_1_max_angles, c='m', label='Mg$_1$', s=8)
    ax_scatter.scatter(mg_2_min_angles, mg_2_max_angles, c='c', label='Mg$_2$', s=8)

    # Axis projections:
    ax_scatter.scatter(const_1_cu, cu_max_angles, c='y', s=3)
    ax_scatter.scatter(const_1_si_1, si_1_max_angles, c='r', s=3)
    ax_scatter.scatter(const_1_si_2, si_2_max_angles, c='k', s=3)
    ax_scatter.scatter(const_1_al, al_max_angles, c='g', s=3)
    ax_scatter.scatter(const_1_mg_1, mg_1_max_angles, c='m', s=3)
    ax_scatter.scatter(const_1_mg_2, mg_2_max_angles, c='c', s=3)

    ax_scatter.scatter(cu_min_angles, const_2_cu, c='y', s=3)
    ax_scatter.scatter(si_1_min_angles, const_2_si_1, c='r', s=3)
    ax_scatter.scatter(si_2_min_angles, const_2_si_2, c='k', s=3)
    ax_scatter.scatter(al_min_angles, const_2_al, c='g', s=3)
    ax_scatter.scatter(mg_1_min_angles, const_2_mg_1, c='m', s=3)
    ax_scatter.scatter(mg_2_min_angles, const_2_mg_2, c='c', s=3)

    # Surface intersection contours:
    colors = ['y', 'r', 'k', 'g', 'm', 'c']
    x, y = np.meshgrid(beta, alpha)
    for a in range(0, 6):
        for b in range(0, 6):

            if not a == b:

                sigma_a = std_min[a] * std_max[a]
                sigma_b = std_min[b] * std_max[b]

                A = (sigma_a ** 2) * std_max[b] ** 2 - (sigma_b ** 2) * std_max[a] ** 2
                B = 0
                C = (sigma_a ** 2) * std_min[b] ** 2 - (sigma_b ** 2) * std_min[a] ** 2
                D = 2 * (mean_min[a] * (sigma_b ** 2) * std_max[a] ** 2 - mean_min[b] * (sigma_a ** 2) * std_max[b] ** 2)
                E = 2 * (mean_max[a] * (sigma_b ** 2) * std_min[a] ** 2 - mean_max[b] * (sigma_a ** 2) * std_min[b] ** 2)
                F = (sigma_a ** 2) * ((mean_min[b] ** 2) * std_max[b] ** 2 + (mean_max[b] ** 2) * std_min[b] ** 2)
                F -= (sigma_b ** 2) * ((mean_min[a] ** 2) * std_max[a] ** 2 + (mean_max[a] ** 2) * std_min[a] ** 2)
                F += 2 * (sigma_a ** 2) * (sigma_b ** 2) * np.log(sigma_b / sigma_a)

                ax_scatter.contour(x, y, (A * x ** 2 + B * x * y + C * y ** 2 + D * x + E * y + F), [0], colors=colors[a])

    ax_scatter.set_title('Scatter-plot of min-max angles')
    ax_scatter.set_xlabel('Min angle (radians)')
    ax_scatter.set_ylabel('max angle (radians)')
    ax_scatter.legend()

    fig.suptitle(title)

    plt.show()


def plot_test_data_2(cu_min_angles, si_1_min_angles, si_2_min_angles, al_min_angles, mg_1_min_angles, mg_2_min_angles, cu_max_angles, si_1_max_angles, si_2_max_angles,
            al_max_angles, mg_1_max_angles, mg_2_max_angles, cu_intensities, si_intensities, al_intensities, mg_intensities, title='Statistical summary'):

    cu_min_std = np.sqrt(utils.variance(cu_min_angles))
    cu_max_std = np.sqrt(utils.variance(cu_max_angles))
    si_1_min_std = np.sqrt(utils.variance(si_1_min_angles))
    si_1_max_std = np.sqrt(utils.variance(si_1_max_angles))
    si_2_min_std = np.sqrt(utils.variance(si_2_min_angles))
    si_2_max_std = np.sqrt(utils.variance(si_2_max_angles))
    al_min_std = np.sqrt(utils.variance(al_min_angles))
    al_max_std = np.sqrt(utils.variance(al_max_angles))
    mg_1_min_std = np.sqrt(utils.variance(mg_1_min_angles))
    mg_1_max_std = np.sqrt(utils.variance(mg_1_max_angles))
    mg_2_min_std = np.sqrt(utils.variance(mg_2_min_angles))
    mg_2_max_std = np.sqrt(utils.variance(mg_2_max_angles))

    cu_min_mean = utils.mean_val(cu_min_angles)
    cu_max_mean = utils.mean_val(cu_max_angles)
    si_1_min_mean = utils.mean_val(si_1_min_angles)
    si_1_max_mean = utils.mean_val(si_1_max_angles)
    si_2_min_mean = utils.mean_val(si_2_min_angles)
    si_2_max_mean = utils.mean_val(si_2_max_angles)
    al_min_mean = utils.mean_val(al_min_angles)
    al_max_mean = utils.mean_val(al_max_angles)
    mg_1_min_mean = utils.mean_val(mg_1_min_angles)
    mg_1_max_mean = utils.mean_val(mg_1_max_angles)
    mg_2_min_mean = utils.mean_val(mg_2_min_angles)
    mg_2_max_mean = utils.mean_val(mg_2_max_angles)

    cu_gamma_std = np.sqrt(utils.variance(cu_intensities))
    si_gamma_std = np.sqrt(utils.variance(si_intensities))
    al_gamma_std = np.sqrt(utils.variance(al_intensities))
    mg_gamma_std = np.sqrt(utils.variance(mg_intensities))

    cu_gamma_mean = utils.mean_val(cu_intensities)
    si_gamma_mean = utils.mean_val(si_intensities)
    al_gamma_mean = utils.mean_val(al_intensities)
    mg_gamma_mean = utils.mean_val(mg_intensities)

    alpha = np.linspace(1, 4, 1000)
    gamma = np.linspace(0, 1, 1000)

    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    ax_min = fig.add_subplot(gs[0, 0])
    ax_max = fig.add_subplot(gs[1, 0])
    ax_co = fig.add_subplot(gs[2, 0])
    ax_scatter = fig.add_subplot(gs[:, 1])

    ax_min.plot(alpha, utils.normal_dist(alpha, cu_min_mean, cu_min_std), 'y',
                label='Cu ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(cu_min_mean, cu_min_std))
    ax_min.plot(alpha, 0.5 * utils.normal_dist(alpha, si_1_min_mean, si_1_min_std)
                + 0.5 * utils.normal_dist(alpha, si_2_min_mean, si_2_min_std), 'r',
                label='Si ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(si_1_min_mean, si_1_min_std))
    ax_min.plot(alpha, utils.normal_dist(alpha, al_min_mean, al_min_std), 'g',
                label='Al ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(al_min_mean, al_min_std))
    ax_min.plot(alpha, 0.5 * utils.normal_dist(alpha, mg_1_min_mean, mg_1_min_std) + 0.5 * utils.normal_dist(alpha, mg_2_min_mean, mg_2_min_std), 'm',
                label='Mg ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(mg_1_min_mean, mg_1_min_std))

    ax_min.set_title('Minimum central angles fitted density')
    ax_min.set_xlabel('Min angle (radians)')
    ax_min.legend()

    ax_max.plot(alpha, utils.normal_dist(alpha, cu_max_mean, cu_max_std), 'y',
                label='Cu ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(cu_max_mean, cu_max_std))
    ax_max.plot(alpha, 0.5 * utils.normal_dist(alpha, si_1_max_mean, si_1_max_std) + 0.5 * utils.normal_dist(alpha, si_2_max_mean, si_2_max_std), 'r',
                label='Si ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(si_1_max_mean, si_1_max_std))
    ax_max.plot(alpha, utils.normal_dist(alpha, al_max_mean, al_max_std), 'g',
                label='Al ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(al_max_mean, al_max_std))
    ax_max.plot(alpha, 0.5 * utils.normal_dist(alpha, mg_1_max_mean, mg_1_max_std) + 0.5 * utils.normal_dist(alpha, mg_2_max_mean, mg_2_max_std), 'm',
                label='Mg ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(mg_2_max_mean, mg_2_max_std))

    ax_max.set_title('Maximum central angles fitted density')
    ax_max.set_xlabel('max angle (radians)')
    ax_max.legend()

    ax_co.plot(gamma, utils.normal_dist(gamma, cu_gamma_mean, cu_gamma_std), 'y',
               label='Cu')
    ax_co.plot(gamma, utils.normal_dist(gamma, si_gamma_mean, si_gamma_std), 'r',
               label='Si')
    ax_co.plot(gamma, utils.normal_dist(gamma, al_gamma_mean, al_gamma_std), 'g',
               label='Al')
    ax_co.plot(gamma, utils.normal_dist(gamma, mg_gamma_mean, mg_gamma_std), 'm',
               label='Mg')

    ax_co.set_title('peak z-contrast distributions')
    ax_co.set_xlabel('z-contrast (normalized $\in (0, 1)$)')
    ax_co.legend()

    ax_scatter.scatter(cu_min_angles, cu_max_angles, c='y', label='Cu', s=8)
    ax_scatter.scatter(si_1_min_angles, si_1_max_angles, c='r', label='Si$_1$', s=8)
    ax_scatter.scatter(si_2_min_angles, si_2_max_angles, c='k', label='Si$_2$', s=8)
    ax_scatter.scatter(al_min_angles, al_max_angles, c='g', label='Al', s=8)
    ax_scatter.scatter(mg_1_min_angles, mg_1_max_angles, c='m', label='Mg$_1$', s=8)
    ax_scatter.scatter(mg_2_min_angles, mg_2_max_angles, c='c', label='Mg$_2$', s=8)

    ax_scatter.set_title('Scatter-plot of min-max angles')
    ax_scatter.set_xlabel('Min angle (radians)')
    ax_scatter.set_ylabel('max angle (radians)')
    ax_scatter.legend()

    fig.suptitle(title)

    plt.show()

