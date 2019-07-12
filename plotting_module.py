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


class InterAtomicDistances:

    def __init__(self, files, distance_mode='spatial'):

        self.files = files
        self.distance_mode = distance_mode
        self.number_of_edges = 0
        self.number_of_files = 0

        self.si_si = []
        self.si_cu = []
        self.si_al = []
        self.si_mg = []
        self.cu_cu = []
        self.cu_al = []
        self.cu_mg = []
        self.al_al = []
        self.al_mg = []
        self.mg_mg = []

        self.si_si_mean = 0
        self.si_cu_mean = 0
        self.si_al_mean = 0
        self.si_mg_mean = 0
        self.cu_cu_mean = 0
        self.cu_al_mean = 0
        self.cu_mg_mean = 0
        self.al_al_mean = 0
        self.al_mg_mean = 0
        self.mg_mg_mean = 0

        self.si_si_std = 0
        self.si_cu_std = 0
        self.si_al_std = 0
        self.si_mg_std = 0
        self.cu_cu_std = 0
        self.cu_al_std = 0
        self.cu_mg_std = 0
        self.al_al_std = 0
        self.al_mg_std = 0
        self.mg_mg_std = 0

    def accumulate_data(self, exclude_edges=True, exclude_matrix=False, exclude_hidden=False, exclude_1=False, exclude_2=False, exclude_3=False, exclude_4=False):

        logger.info('Accumulating data...')

        for file_ in iter(self.files.splitlines()):
            instance = core.SuchSoftware.load(file_)
            self.number_of_files += 1

            lattice_const = core.SuchSoftware.al_lattice_const

            for vertex_a in instance.graph.vertices:
                for partner_index in vertex_a.partner_indices:

                    vertex_b = instance.graph.vertices[partner_index]

                    if not (exclude_edges and (vertex_a.is_edge_column or vertex_b.is_edge_column)):
                        if not(exclude_matrix and not (vertex_a.is_in_precipitate or vertex_b.is_in_precipitate)):
                            if not (exclude_hidden and not (vertex_a.show_in_overlay or vertex_b.show_in_overlay)):
                                if not (exclude_1 and (vertex_a.flag_1 or vertex_b.flag_1)):
                                    if not (exclude_2 and (vertex_a.flag_2 or vertex_b.flag_2)):
                                        if not (exclude_3 and (vertex_a.flag_3 or vertex_b.flag_3)):
                                            if not (exclude_4 and (vertex_a.flag_4 or vertex_b.flag_4)):

                                                x = vertex_a.real_coor_x - vertex_b.real_coor_x
                                                x *= instance.scale
                                                y = vertex_a.real_coor_y - vertex_b.real_coor_y
                                                y *= instance.scale
                                                projected_distance = np.sqrt(x ** 2 + y ** 2)
                                                if vertex_a.level == vertex_b.level:
                                                    spatial_distance = projected_distance
                                                else:
                                                    spatial_distance = np.sqrt(projected_distance ** 2 + (lattice_const / 2) ** 2)

                                                if self.distance_mode == 'spatial':
                                                    pass
                                                elif self.distance_mode == 'projected':
                                                    spatial_distance = projected_distance

                                                if vertex_a.h_index == 0 and vertex_b.h_index == 0:
                                                    self.si_si.append(spatial_distance)
                                                elif (vertex_a.h_index == 0 and vertex_b.h_index == 1) or (vertex_b.h_index == 0 and vertex_a.h_index == 1):
                                                    self.si_cu.append(spatial_distance)
                                                elif (vertex_a.h_index == 0 and vertex_b.h_index == 3) or (vertex_b.h_index == 0 and vertex_a.h_index == 3):
                                                    self.si_al.append(spatial_distance)
                                                elif (vertex_a.h_index == 0 and vertex_b.h_index == 5) or (vertex_b.h_index == 0 and vertex_a.h_index == 5):
                                                    self.si_mg.append(spatial_distance)
                                                elif vertex_a.h_index == 1 and vertex_b.h_index == 1:
                                                    self.cu_cu.append(spatial_distance)
                                                elif (vertex_a.h_index == 1 and vertex_b.h_index == 3) or (vertex_b.h_index == 1 and vertex_a.h_index == 3):
                                                    self.cu_al.append(spatial_distance)
                                                elif (vertex_a.h_index == 1 and vertex_b.h_index == 5) or (vertex_b.h_index == 1 and vertex_a.h_index == 5):
                                                    self.cu_mg.append(spatial_distance)
                                                elif vertex_a.h_index == 3 and vertex_b.h_index == 3:
                                                    self.al_al.append(spatial_distance)
                                                elif (vertex_a.h_index == 3 and vertex_b.h_index == 5) or (vertex_b.h_index == 3 and vertex_a.h_index == 5):
                                                    self.al_mg.append(spatial_distance)
                                                elif vertex_a.h_index == 5 and vertex_b.h_index == 5:
                                                    self.mg_mg.append(spatial_distance)

        self.si_si_mean = utils.mean_val(self.si_si)
        self.si_cu_mean = utils.mean_val(self.si_cu)
        self.si_al_mean = utils.mean_val(self.si_al)
        self.si_mg_mean = utils.mean_val(self.si_mg)
        self.cu_cu_mean = utils.mean_val(self.cu_cu)
        self.cu_al_mean = utils.mean_val(self.cu_al)
        self.cu_mg_mean = utils.mean_val(self.cu_mg)
        self.al_al_mean = utils.mean_val(self.al_al)
        self.al_mg_mean = utils.mean_val(self.al_mg)
        self.mg_mg_mean = utils.mean_val(self.mg_mg)

        self.si_si_std = np.sqrt(utils.variance(self.si_si))
        self.si_cu_std = np.sqrt(utils.variance(self.si_cu))
        self.si_al_std = np.sqrt(utils.variance(self.si_al))
        self.si_mg_std = np.sqrt(utils.variance(self.si_mg))
        self.cu_cu_std = np.sqrt(utils.variance(self.cu_cu))
        self.cu_al_std = np.sqrt(utils.variance(self.cu_al))
        self.cu_mg_std = np.sqrt(utils.variance(self.cu_mg))
        self.al_al_std = np.sqrt(utils.variance(self.al_al))
        self.al_mg_std = np.sqrt(utils.variance(self.al_mg))
        self.mg_mg_std = np.sqrt(utils.variance(self.mg_mg))

    def plot(self):

        logger.info('Generating plots...')

        distance = np.linspace(200, 400, 1000)

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, 1, figure=fig)
        ax_same = fig.add_subplot(gs[0, 0])
        ax_pairs_1 = fig.add_subplot(gs[1, 0])

        ax_same.plot(distance, utils.normal_dist(distance, self.si_si_mean, self.si_si_std), 'r', label='Si <-> Si')
        ax_same.plot(distance, utils.normal_dist(distance, self.cu_cu_mean, self.cu_cu_std), 'y', label='Cu <-> Cu')
        ax_same.plot(distance, utils.normal_dist(distance, self.al_al_mean, self.al_al_std), 'g', label='Al <-> Al')
        ax_same.plot(distance, utils.normal_dist(distance, self.mg_mg_mean, self.mg_mg_std), 'm', label='Mg <-> Mg')

        ax_same.axvline(x=2 * core.SuchSoftware.si_radii, c='r')
        ax_same.axvline(x=2 * core.SuchSoftware.cu_radii, c='y')
        ax_same.axvline(x=2 * core.SuchSoftware.al_radii, c='g')
        ax_same.axvline(x=2 * core.SuchSoftware.mg_radii, c='m')

        ax_same.set_title('Similar species pairs')
        ax_same.set_xlabel('Inter-atomic distance (pm)')
        ax_same.legend()

        ax_pairs_1.plot(distance, utils.normal_dist(distance, self.si_cu_mean, self.si_cu_std), 'r', label='Si <-> Cu')
        ax_pairs_1.plot(distance, utils.normal_dist(distance, self.si_al_mean, self.si_al_std), 'k', label='Si <-> Al')
        ax_pairs_1.plot(distance, utils.normal_dist(distance, self.si_mg_mean, self.si_mg_std), 'c', label='Si <-> Mg')
        ax_pairs_1.plot(distance, utils.normal_dist(distance, self.cu_al_mean, self.cu_al_std), 'y', label='Cu <-> Al')
        ax_pairs_1.plot(distance, utils.normal_dist(distance, self.cu_al_mean, self.cu_al_std), 'b', label='Cu <-> Mg')
        ax_pairs_1.plot(distance, utils.normal_dist(distance, self.al_mg_mean, self.al_mg_std), 'g', label='Al <-> Mg')

        ax_pairs_1.axvline(x=core.SuchSoftware.si_radii + core.SuchSoftware.cu_radii, c='r')
        ax_pairs_1.axvline(x=core.SuchSoftware.si_radii + core.SuchSoftware.al_radii, c='k')
        ax_pairs_1.axvline(x=core.SuchSoftware.si_radii + core.SuchSoftware.mg_radii, c='c')
        ax_pairs_1.axvline(x=core.SuchSoftware.cu_radii + core.SuchSoftware.al_radii, c='y')
        ax_pairs_1.axvline(x=core.SuchSoftware.cu_radii + core.SuchSoftware.mg_radii, c='b')
        ax_pairs_1.axvline(x=core.SuchSoftware.al_radii + core.SuchSoftware.mg_radii, c='g')

        ax_pairs_1.set_title('Un-similar species pairs')
        ax_pairs_1.set_xlabel('Inter-atomic distance (pm)')
        ax_pairs_1.legend()

        fig.suptitle('Fitted distributions of inter-atomic distances\n'
                     '(Vertical lines represent hard sphere model values)')

        plt.show()


class Gamma:

    def __init__(self, files):

        self.files = files
        self.number_of_vertices = 0
        self.number_of_files = 0

        self.cu_avg_intensities = []
        self.si_avg_intensities = []
        self.al_avg_intensities = []
        self.mg_avg_intensities = []

        self.cu_avg_gamma_std = 0
        self.si_avg_gamma_std = 0
        self.al_avg_gamma_std = 0
        self.mg_avg_gamma_std = 0

        self.cu_avg_gamma_mean = 0
        self.si_avg_gamma_mean = 0
        self.al_avg_gamma_mean = 0
        self.mg_avg_gamma_mean = 0

        self.cu_peak_intensities = []
        self.si_peak_intensities = []
        self.al_peak_intensities = []
        self.mg_peak_intensities = []

        self.cu_peak_gamma_std = 0
        self.si_peak_gamma_std = 0
        self.al_peak_gamma_std = 0
        self.mg_peak_gamma_std = 0

        self.cu_peak_gamma_mean = 0
        self.si_peak_gamma_mean = 0
        self.al_peak_gamma_mean = 0
        self.mg_peak_gamma_mean = 0

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

                                            if vertex.species() == 'Si':
                                                self.si_avg_intensities.append(vertex.peak_gamma)
                                                self.si_peak_intensities.append(vertex.avg_gamma)
                                            elif vertex.species() == 'Cu':
                                                self.cu_avg_intensities.append(vertex.peak_gamma)
                                                self.cu_peak_intensities.append(vertex.avg_gamma)
                                            elif vertex.species() == 'Al':
                                                self.al_avg_intensities.append(vertex.peak_gamma)
                                                self.al_peak_intensities.append(vertex.avg_gamma)
                                            elif vertex.species() == 'Mg':
                                                self.mg_avg_intensities.append(vertex.peak_gamma)
                                                self.mg_peak_intensities.append(vertex.avg_gamma)

        self.cu_avg_gamma_std = np.sqrt(utils.variance(self.cu_avg_intensities))
        self.si_avg_gamma_std = np.sqrt(utils.variance(self.si_avg_intensities))
        self.al_avg_gamma_std = np.sqrt(utils.variance(self.al_avg_intensities))
        self.mg_avg_gamma_std = np.sqrt(utils.variance(self.mg_avg_intensities))

        self.cu_avg_gamma_mean = utils.mean_val(self.cu_avg_intensities)
        self.si_avg_gamma_mean = utils.mean_val(self.si_avg_intensities)
        self.al_avg_gamma_mean = utils.mean_val(self.al_avg_intensities)
        self.mg_avg_gamma_mean = utils.mean_val(self.mg_avg_intensities)

        self.cu_peak_gamma_std = np.sqrt(utils.variance(self.cu_avg_intensities))
        self.si_peak_gamma_std = np.sqrt(utils.variance(self.si_peak_intensities))
        self.al_peak_gamma_std = np.sqrt(utils.variance(self.al_peak_intensities))
        self.mg_peak_gamma_std = np.sqrt(utils.variance(self.mg_peak_intensities))

        self.cu_peak_gamma_mean = utils.mean_val(self.cu_peak_intensities)
        self.si_peak_gamma_mean = utils.mean_val(self.si_peak_intensities)
        self.al_peak_gamma_mean = utils.mean_val(self.al_peak_intensities)
        self.mg_peak_gamma_mean = utils.mean_val(self.mg_peak_intensities)

    def plot(self):

        logger.info('Generating plots...')

        gamma = np.linspace(0, 1, 1000)

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig)
        ax_peak = fig.add_subplot(gs[0, 0])
        ax_avg = fig.add_subplot(gs[1, 0])
        ax_scatter = fig.add_subplot(gs[:, 1])

        ax_peak.plot(gamma, utils.normal_dist(gamma, self.cu_peak_gamma_mean, self.cu_peak_gamma_std), 'y', label='Cu')
        ax_peak.plot(gamma, utils.normal_dist(gamma, self.si_peak_gamma_mean, self.si_peak_gamma_std), 'r', label='Si')
        ax_peak.plot(gamma, utils.normal_dist(gamma, self.al_peak_gamma_mean, self.al_peak_gamma_std), 'g', label='Al')
        ax_peak.plot(gamma, utils.normal_dist(gamma, self.mg_peak_gamma_mean, self.mg_peak_gamma_std), 'm', label='Mg')

        ax_peak.set_title('peak z-contrast fitted distributions')
        ax_peak.set_xlabel('peak z-contrast (normalized $\in (0, 1)$)')
        ax_peak.legend()

        ax_avg.plot(gamma, utils.normal_dist(gamma, self.cu_avg_gamma_mean, self.cu_avg_gamma_std), 'y', label='Cu')
        ax_avg.plot(gamma, utils.normal_dist(gamma, self.si_avg_gamma_mean, self.si_avg_gamma_std), 'r', label='Si')
        ax_avg.plot(gamma, utils.normal_dist(gamma, self.al_avg_gamma_mean, self.al_avg_gamma_std), 'g', label='Al')
        ax_avg.plot(gamma, utils.normal_dist(gamma, self.mg_avg_gamma_mean, self.mg_avg_gamma_std), 'm', label='Mg')

        ax_avg.set_title('average z-contrast fitted distributions')
        ax_avg.set_xlabel('average z-contrast (normalized $\in (0, 1)$)')
        ax_avg.legend()

        ax_scatter.scatter(self.cu_peak_intensities, self.cu_avg_intensities, c='y', label='Cu', s=8)
        ax_scatter.scatter(self.si_peak_intensities, self.si_avg_intensities, c='r', label='Si', s=8)
        ax_scatter.scatter(self.al_peak_intensities, self.al_avg_intensities, c='g', label='Al', s=8)
        ax_scatter.scatter(self.mg_peak_intensities, self.mg_avg_intensities, c='m', label='Mg', s=8)

        ax_scatter.set_title('Scatter-plot of peak-avg contrast')
        ax_scatter.set_xlabel('peak z-contrast (normalized $\in (0, 1)$)')
        ax_scatter.set_ylabel('average z-contrast (normalized $\in (0, 1)$)')
        ax_scatter.legend()

        fig.suptitle('Scatter plot of peak-avg contrasts')

        plt.show()


class MinMax:

    def __init__(self, files, angle_mode='alpha'):

        self.files = files
        self.angle_mode = angle_mode
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

                                            if self.angle_mode == 'alpha':
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

                                            elif self.angle_mode == 'theta':
                                                sub_graph = instance.graph.get_atomic_configuration(vertex.i)
                                                theta_angles = []
                                                for mesh in sub_graph.meshes:
                                                    theta_angles.append(mesh.angles[0])
                                                max_ = max(theta_angles)
                                                min_ = min(theta_angles)

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

