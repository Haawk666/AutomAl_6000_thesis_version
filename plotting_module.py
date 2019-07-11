import core
from matplotlib import pyplot as plt
import graph_op
import numpy as np
import utils
from matplotlib.gridspec import GridSpec
import logging

# Instantiate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AlphaMinMax:

    def __init__(self, files):

        self.files = files
        self.number_of_vertices = 0
        self.number_of_files = 0

        self.cu_min_angles = []
        self.si_1_min_angles = []
        self.si_2_min_angles = []
        self.al_min_angles = []
        self.mg_1_min_angles = []
        self.mg_2_min_angles = []
        self.mg_3_min_angles = []

        self.cu_max_angles = []
        self.si_1_max_angles = []
        self.si_2_max_angles = []
        self.al_max_angles = []
        self.mg_1_max_angles = []
        self.mg_2_max_angles = []
        self.mg_3_max_angles = []

        self.cu_min_std = 0
        self.cu_max_std = 0
        self.si_1_min_std = 0
        self.si_1_max_std = 0
        self.si_2_min_std = 0
        self.si_2_max_std = 0
        self.al_min_std = 0
        self.al_max_std = 0
        self.mg_1_min_std = 0
        self.mg_1_max_std = 0
        self.mg_2_min_std = 0
        self.mg_2_max_std = 0
        self.mg_3_min_std = 0
        self.mg_3_max_std = 0

        self.std_min = []
        self.std_max = []

        self.cu_min_mean = 0
        self.cu_max_mean = 0
        self.si_1_min_mean = 0
        self.si_1_max_mean = 0
        self.si_2_min_mean = 0
        self.si_2_max_mean = 0
        self.al_min_mean = 0
        self.al_max_mean = 0
        self.mg_1_min_mean = 0
        self.mg_1_max_mean = 0
        self.mg_2_min_mean = 0
        self.mg_2_max_mean = 0
        self.mg_3_min_mean = 0
        self.mg_3_max_mean = 0

        self.mean_min = []
        self.mean_max = []

    def accumulate_data(self, exclude_edges=True, exclude_matrix=False, exclude_hidden=False, exclude_1=False, exclude_2=False, exclude_3=False, exclude_4=False):

        logger.info('Accumulating data...')

        for file_ in iter(self.files.splitlines()):
            instance = core.SuchSoftware.load(file_)
            self.number_of_files += 1

            for vertex in instance.graph.vertices:
                if not (exclude_edges and vertex.is_edge_column):
                    if not(exclude_matrix and not vertex.is_in_precipitate):
                        if not (exclude_hidden and not vertex.show_in_overlay):
                            if not (exclude_1 and vertex.flag_1):
                                if not (exclude_2 and vertex.flag_2):
                                    if not (exclude_3 and vertex.flag_3):
                                        if not (exclude_4 and vertex.flag_4):

                                            self.number_of_vertices += 1
                                            max_, min_ = graph_op.base_angle_score(instance.graph, vertex.i, apply=False)

                                            if vertex.species() == 'Cu':
                                                self.cu_min_angles.append(min_)
                                                self.cu_max_angles.append(max_)

                                            elif vertex.species() == 'Si':
                                                if vertex.flag_2:
                                                    self.si_2_min_angles.append(min_)
                                                    self.si_2_max_angles.append(max_)
                                                else:
                                                    self.si_1_min_angles.append(min_)
                                                    self.si_1_max_angles.append(max_)

                                            elif vertex.species() == 'Al':
                                                self.al_min_angles.append(min_)
                                                self.al_max_angles.append(max_)

                                            elif vertex.species() == 'Mg':
                                                if not vertex.flag_3:
                                                    if max_ > 3.15:
                                                        self.mg_2_min_angles.append(min_)
                                                        self.mg_2_max_angles.append(max_)
                                                    else:
                                                        self.mg_1_min_angles.append(min_)
                                                        self.mg_1_max_angles.append(max_)
                                                else:
                                                    self.mg_3_min_angles.append(min_)
                                                    self.mg_3_max_angles.append(max_)

        self.cu_min_std = np.sqrt(utils.variance(self.cu_min_angles))
        self.cu_max_std = np.sqrt(utils.variance(self.cu_max_angles))
        self.si_1_min_std = np.sqrt(utils.variance(self.si_1_min_angles))
        self.si_1_max_std = np.sqrt(utils.variance(self.si_1_max_angles))
        self.si_2_min_std = np.sqrt(utils.variance(self.si_2_min_angles))
        self.si_2_max_std = np.sqrt(utils.variance(self.si_2_max_angles))
        self.al_min_std = np.sqrt(utils.variance(self.al_min_angles))
        self.al_max_std = np.sqrt(utils.variance(self.al_max_angles))
        self.mg_1_min_std = np.sqrt(utils.variance(self.mg_1_min_angles))
        self.mg_1_max_std = np.sqrt(utils.variance(self.mg_1_max_angles))
        self.mg_2_min_std = np.sqrt(utils.variance(self.mg_2_min_angles))
        self.mg_2_max_std = np.sqrt(utils.variance(self.mg_2_max_angles))
        self.mg_3_min_std = np.sqrt(utils.variance(self.mg_3_min_angles))
        self.mg_3_max_std = np.sqrt(utils.variance(self.mg_3_max_angles))

        self.std_min = [self.cu_min_std,
                        self.si_1_min_std,
                        self.si_2_min_std,
                        self.al_min_std,
                        self.mg_1_min_std,
                        self.mg_2_min_std,
                        self.mg_3_min_std]
        self.std_max = [self.cu_max_std,
                        self.si_1_max_std,
                        self.si_2_max_std,
                        self.al_max_std,
                        self.mg_1_max_std,
                        self.mg_2_max_std,
                        self.mg_3_max_std]

        self.cu_min_mean = utils.mean_val(self.cu_min_angles)
        self.cu_max_mean = utils.mean_val(self.cu_max_angles)
        self.si_1_min_mean = utils.mean_val(self.si_1_min_angles)
        self.si_1_max_mean = utils.mean_val(self.si_1_max_angles)
        self.si_2_min_mean = utils.mean_val(self.si_2_min_angles)
        self.si_2_max_mean = utils.mean_val(self.si_2_max_angles)
        self.al_min_mean = utils.mean_val(self.al_min_angles)
        self.al_max_mean = utils.mean_val(self.al_max_angles)
        self.mg_1_min_mean = utils.mean_val(self.mg_1_min_angles)
        self.mg_1_max_mean = utils.mean_val(self.mg_1_max_angles)
        self.mg_2_min_mean = utils.mean_val(self.mg_2_min_angles)
        self.mg_2_max_mean = utils.mean_val(self.mg_2_max_angles)
        self.mg_3_min_mean = utils.mean_val(self.mg_3_min_angles)
        self.mg_3_max_mean = utils.mean_val(self.mg_3_max_angles)

        self.mean_min = [self.cu_min_mean,
                         self.si_1_min_mean,
                         self.si_2_min_mean,
                         self.al_min_mean,
                         self.mg_1_min_mean,
                         self.mg_2_min_mean,
                         self.mg_3_min_mean]
        self.mean_max = [self.cu_max_mean,
                         self.si_1_max_mean,
                         self.si_2_max_mean,
                         self.al_max_mean,
                         self.mg_1_max_mean,
                         self.mg_2_max_mean,
                         self.mg_3_max_mean]

    def plot(self):

        logger.info('Generating plot(s)...')

        alpha = np.linspace(1, 4, 1000)

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig)
        ax_min = fig.add_subplot(gs[0, 0])
        ax_max = fig.add_subplot(gs[1, 0])
        ax_scatter = fig.add_subplot(gs[:, 1])

        ax_min.plot(alpha, utils.normal_dist(alpha, self.cu_min_mean, self.cu_min_std), 'y',
                    label='Cu ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.cu_min_mean, self.cu_min_std))
        ax_min.plot(alpha, utils.normal_dist(alpha, self.si_1_min_mean, self.si_1_min_std), 'r',
                    label='Si$_1$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.si_1_min_mean, self.si_1_min_std))
        ax_min.plot(alpha, utils.normal_dist(alpha, self.si_2_min_mean, self.si_2_min_std), 'k',
                    label='Si$_2$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.si_2_min_mean, self.si_2_min_std))
        ax_min.plot(alpha, utils.normal_dist(alpha, self.al_min_mean, self.al_min_std), 'g',
                    label='Al ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.al_min_mean, self.al_min_std))
        ax_min.plot(alpha, utils.normal_dist(alpha, self.mg_1_min_mean, self.mg_1_min_std), 'm',
                    label='Mg$_1$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.mg_1_min_mean, self.mg_1_min_std))
        ax_min.plot(alpha, utils.normal_dist(alpha, self.mg_2_min_mean, self.mg_2_min_std), 'c',
                    label='Mg$_2$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.mg_2_min_mean, self.mg_2_min_std))
        ax_min.plot(alpha, utils.normal_dist(alpha, self.mg_3_min_mean, self.mg_3_min_std), 'b',
                    label='Mg$_3$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.mg_3_min_mean,
                                                                                   self.mg_3_min_std))

        ax_min.set_title('Minimum central angles fitted density')
        ax_min.set_xlabel('Min angle (radians)')
        ax_min.legend()

        ax_max.plot(alpha, utils.normal_dist(alpha, self.cu_max_mean, self.cu_max_std), 'y',
                    label='Cu ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.cu_max_mean, self.cu_max_std))
        ax_max.plot(alpha, utils.normal_dist(alpha, self.si_1_max_mean, self.si_1_max_std), 'r',
                    label='Si$_1$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.si_1_max_mean, self.si_1_max_std))
        ax_max.plot(alpha, utils.normal_dist(alpha, self.si_2_max_mean, self.si_2_max_std), 'k',
                    label='Si$_2$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.si_2_max_mean, self.si_2_max_std))
        ax_max.plot(alpha, utils.normal_dist(alpha, self.al_max_mean, self.al_max_std), 'g',
                    label='Al ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.al_max_mean, self.al_max_std))
        ax_max.plot(alpha, utils.normal_dist(alpha, self.mg_1_max_mean, self.mg_1_max_std), 'm',
                    label='Mg$_1$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.mg_1_max_mean, self.mg_1_max_std))
        ax_max.plot(alpha, utils.normal_dist(alpha, self.mg_2_max_mean, self.mg_2_max_std), 'c',
                    label='Mg$_2$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.mg_2_max_mean, self.mg_2_max_std))
        ax_max.plot(alpha, utils.normal_dist(alpha, self.mg_3_max_mean, self.mg_3_max_std), 'b',
                    label='Mg$_3$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.mg_3_max_mean,
                                                                                   self.mg_3_max_std))

        ax_max.set_title('Maximum central angles fitted density')
        ax_max.set_xlabel('max angle (radians)')
        ax_max.legend()

        ax_scatter.scatter(self.cu_min_angles, self.cu_max_angles, c='y', label='Cu', s=8)
        ax_scatter.scatter(self.si_1_min_angles, self.si_1_max_angles, c='r', label='Si$_1$', s=8)
        ax_scatter.scatter(self.si_2_min_angles, self.si_2_max_angles, c='k', label='Si$_2$', s=8)
        ax_scatter.scatter(self.al_min_angles, self.al_max_angles, c='g', label='Al', s=8)
        ax_scatter.scatter(self.mg_1_min_angles, self.mg_1_max_angles, c='m', label='Mg$_1$', s=8)
        ax_scatter.scatter(self.mg_2_min_angles, self.mg_2_max_angles, c='c', label='Mg$_2$', s=8)
        ax_scatter.scatter(self.mg_3_min_angles, self.mg_3_max_angles, c='b', label='Mg$_3$', s=8)

        ax_scatter.set_title('Scatter-plot of min-max angles')
        ax_scatter.set_xlabel('Min angle (radians)')
        ax_scatter.set_ylabel('max angle (radians)')
        ax_scatter.legend()

        fig.suptitle('Alpha min/max summary')

        logger.info('Plotted min/max alpha over {} files and {} vertices!'.format(self.number_of_files, self.number_of_vertices))

        plt.show()


