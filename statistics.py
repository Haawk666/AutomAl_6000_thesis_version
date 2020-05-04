"""This module contains the different statistical models used in the column characterization algorithm. It also has
methods that can gather model parameters from sets of overlay files. The default model parameters used here are
gathered from the images in the validation data set. See automal.org for more details."""


import core
import utils
import default_models

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pickle
import logging
# Instantiate logger:
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def alpha_model(alpha_max, alpha_min):
    x = [alpha_max, alpha_min]
    params = default_models.alpha_model
    advanced_model = [
        utils.multivariate_normal_dist(x, params[0][0], params[0][3], params[0][4]),
        utils.multivariate_normal_dist(x, params[1][0], params[1][3], params[1][4]),
        utils.multivariate_normal_dist(x, params[2][0], params[2][3], params[2][4]),
        utils.multivariate_normal_dist(x, params[3][0], params[3][3], params[3][4]),
        utils.multivariate_normal_dist(x, params[4][0], params[4][3], params[4][4]),
        utils.multivariate_normal_dist(x, params[5][0], params[5][3], params[5][4]),
        utils.multivariate_normal_dist(x, params[6][0], params[6][3], params[6][4])
    ]
    advanced_model = utils.normalize_list(advanced_model, 1)
    simple_model = [
        advanced_model[0] + advanced_model[1],
        advanced_model[2],
        0,
        advanced_model[3] + advanced_model[4],
        0,
        advanced_model[5] + advanced_model[6],
        0
    ]
    return simple_model, advanced_model


def theta_model(theta_max, theta_min, theta_avg):
    x = [theta_max, theta_min, theta_avg]
    params = default_models.theta_model
    advanced_model = [
        utils.multivariate_normal_dist(x, params[0][0], params[0][3], params[0][4]),
        utils.multivariate_normal_dist(x, params[1][0], params[1][3], params[1][4]),
        utils.multivariate_normal_dist(x, params[2][0], params[2][3], params[2][4]),
        utils.multivariate_normal_dist(x, params[3][0], params[3][3], params[3][4]),
        utils.multivariate_normal_dist(x, params[4][0], params[4][3], params[4][4]),
        utils.multivariate_normal_dist(x, params[5][0], params[5][3], params[5][4]),
        utils.multivariate_normal_dist(x, params[6][0], params[6][3], params[6][4])
    ]
    advanced_model = utils.normalize_list(advanced_model, 1)
    simple_model = [
        advanced_model[0] + advanced_model[1],
        advanced_model[2],
        0,
        advanced_model[3] + advanced_model[4],
        0,
        advanced_model[5] + advanced_model[6],
        0
    ]
    return simple_model, advanced_model


def normalized_gamma_model(normalized_peak_gamma, normalized_avg_gamma):
    x = [normalized_peak_gamma, normalized_avg_gamma]
    params = default_models.gamma_model
    advanced_model = [
        utils.multivariate_normal_dist(x, params[0][0], params[0][3], params[0][4]),
        utils.multivariate_normal_dist(x, params[1][0], params[1][3], params[1][4]),
        utils.multivariate_normal_dist(x, params[2][0], params[2][3], params[2][4]),
        utils.multivariate_normal_dist(x, params[3][0], params[3][3], params[3][4]),
        utils.multivariate_normal_dist(x, params[4][0], params[4][3], params[4][4]),
        utils.multivariate_normal_dist(x, params[5][0], params[5][3], params[5][4]),
        utils.multivariate_normal_dist(x, params[6][0], params[6][3], params[6][4])
    ]
    advanced_model = utils.normalize_list(advanced_model, 1)
    simple_model = [
        advanced_model[0] + advanced_model[1],
        advanced_model[2],
        0,
        advanced_model[3] + advanced_model[4],
        0,
        advanced_model[5] + advanced_model[6],
        0
    ]
    return simple_model, advanced_model


def composite_model(alpha_max, alpha_min, theta_max, theta_min, theta_avg, gamma_avg, gamma_peak):
    x = [alpha_max, alpha_min, theta_max, theta_min, theta_avg, gamma_avg, gamma_peak]
    params = default_models.composite_model
    advanced_model = [
        utils.multivariate_normal_dist(x, params[0][0], params[0][3], params[0][4]),
        utils.multivariate_normal_dist(x, params[1][0], params[1][3], params[1][4]),
        utils.multivariate_normal_dist(x, params[2][0], params[2][3], params[2][4]),
        utils.multivariate_normal_dist(x, params[3][0], params[3][3], params[3][4]),
        utils.multivariate_normal_dist(x, params[4][0], params[4][3], params[4][4]),
        utils.multivariate_normal_dist(x, params[5][0], params[5][3], params[5][4]),
        utils.multivariate_normal_dist(x, params[6][0], params[6][3], params[6][4])
    ]
    advanced_model = utils.normalize_list(advanced_model, 1)
    simple_model = [
        advanced_model[0] + advanced_model[1],
        advanced_model[2],
        0,
        advanced_model[3] + advanced_model[4],
        0,
        advanced_model[5] + advanced_model[6],
        0
    ]
    return simple_model, advanced_model


def product_model(vertex):
    pass


def weighted_model(vertex):
    pass


def calculate_models(files, filter_=None, recalc_properties=False, savefile='default_models'):

    data = []
    keys = ['advanced_category_index', 'alpha_max', 'alpha_min', 'theta_max', 'theta_min',
        'theta_angle_mean', 'normalized_peak_gamma', 'normalized_avg_gamma', 'avg_central_separation'
    ]
    for file in files.splitlines(keepends=False):
        instance = core.SuchSoftware.load(file)
        image_data = instance.graph.calc_condensed_property_data(filter_=filter_, recalc=recalc_properties,
                                                                 evaluate_category=True, keys=keys)
        data += image_data

    category_titles = ['Si_1', 'Si_2', 'Cu', 'Al_1', 'Al_2', 'Mg_1', 'Mg_2']
    reformatted_data = [
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []]
    ]
    for data_item in data:
        data_item['category_title'] = category_titles[data_item['advanced_category_index']]
        reformatted_data[data_item['advanced_category_index']][0].append(data_item['alpha_max'])
        reformatted_data[data_item['advanced_category_index']][1].append(data_item['alpha_min'])
        reformatted_data[data_item['advanced_category_index']][2].append(data_item['theta_max'])
        reformatted_data[data_item['advanced_category_index']][3].append(data_item['theta_min'])
        reformatted_data[data_item['advanced_category_index']][4].append(data_item['theta_angle_mean'])
        reformatted_data[data_item['advanced_category_index']][5].append(data_item['normalized_peak_gamma'])
        reformatted_data[data_item['advanced_category_index']][6].append(data_item['normalized_avg_gamma'])
        reformatted_data[data_item['advanced_category_index']][7].append(data_item['avg_central_separation'])

    model = Model(reformatted_data, category_titles)
    with open(savefile, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    return model


class MultivariateNormalDist:

    def __init__(self, data, category_title):

        # k is the dimensionality of the data, n is the number of data-points
        (self.k, self.n) = data.shape
        # Title of category is mandatory
        self.category_title = category_title

        # The mean of each dimension
        self.means = []
        # The variance of each dimension
        self.variances = []
        # The covariance matrix
        self.covar_matrix = np.zeros([self.k, self.k], dtype=np.float64)
        # The inverse of the covariance matrix
        self.inverse_covar_matrix = np.zeros([self.k, self.k], dtype=np.float64)
        # Determinant of the covariance matrix
        self.covar_matrix_determinant = 0
        # Eigen-things of the covariance matrix
        self.covar_matrix_eigenvalues = []
        self.covar_matrix_eigenvectors = []

        self.calc_params(data)

    def calc_params(self, data):
        """Two pass algorithm"""
        self.means = []
        self.variances = []
        for dimension in range(0, self.k):
            if not self.n == 0:
                mean = np.sum(data[dimension, :]) / self.n
            else:
                mean = 0
            var = 0
            for data_item in data[dimension, :]:
                var += (data_item - mean)**2
            if not self.n < 2:
                var = var / (self.n - 1)
            else:
                var = np.infty
            self.means.append(mean)
            self.variances.append(var)

        for dimension_1 in range(0, self.k):
            for dimension_2 in range(0, self.k):

                for data_index in range(0, self.n):
                    factor_1 = data[dimension_1, data_index] - self.means[dimension_1]
                    factor_2 = data[dimension_2, data_index] - self.means[dimension_2]
                    self.covar_matrix[dimension_1, dimension_2] += factor_1 * factor_2
                self.covar_matrix[dimension_1, dimension_2] /= self.n

        self.covar_matrix_determinant = np.linalg.det(self.covar_matrix)
        self.inverse_covar_matrix = np.linalg.inv(self.covar_matrix)
        self.covar_matrix_eigenvalues, self.covar_matrix_eigenvectors = np.linalg.eig(self.covar_matrix)
        idx = np.argsort(self.covar_matrix_eigenvalues)[::-1]
        self.covar_matrix_eigenvalues = self.covar_matrix_eigenvalues[idx]
        self.covar_matrix_eigenvectors = self.covar_matrix_eigenvectors[:, idx]


class Model:

    def __init__(self, data, category_titles=None):

        self.num_data_categories = len(data)
        self.k = len(data[0])

        if category_titles is None:
            self.category_titles = []
            for i in range(1, self.num_data_categories + 1):
                self.category_titles.append('Category {}'.format(i))
        else:
            self.category_titles = category_titles
        self.data = data
        self.prepare_data()
        self.concatenated_data = self.concatenate_categories()

        self.uncategorized_normal_dist = MultivariateNormalDist(self.concatenated_data, 'All categories')

        self.composite_model = []
        self.alpha_model = []
        self.separation_model = []

        for i, category in enumerate(self.data):
            self.composite_model.append(MultivariateNormalDist(category, self.category_titles[i]))
            self.alpha_model.append(MultivariateNormalDist(category[0:1, :], ['alpha_max', 'alpha_min']))
            self.separation_model.append(MultivariateNormalDist(category[7, :], ['avg_central_separation']))

    def prepare_data(self):
        matrix_data = []
        for category in self.data:
            matrix = np.array(category)
            matrix_data.append(matrix)
        self.data = matrix_data

    def concatenate_categories(self):
        concatenated_data = self.data[0]
        for i, category in enumerate(self.category_titles):
            if not i == 0:
                concatenated_data = np.concatenate((concatenated_data, self.data[i]), axis=1)
        return concatenated_data

    def plot_data(self):

    def plot_pca(self):
        logger.info('Generating plot...')

        alpha = np.linspace(-10, 10, 1000)

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig)
        ax_pc_1 = fig.add_subplot(gs[0, 0])
        ax_pc_2 = fig.add_subplot(gs[1, 0])
        ax_scatter = fig.add_subplot(gs[:, 1])

        ax_pc_1.plot(alpha, utils.normal_dist(alpha, utils.mean_val(self.cu_pca_data[:, 2]),
                                              np.sqrt(utils.variance(self.cu_pca_data[:, 2]))),
                     'y',
                     label='Cu ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(utils.mean_val(self.cu_pca_data[:, 2]),
                                                                                np.sqrt(utils.variance(
                                                                                    self.cu_pca_data[:, 2]))))
        ax_pc_1.plot(alpha, utils.normal_dist(alpha, utils.mean_val(self.si_pca_data[:, 2]),
                                              np.sqrt(utils.variance(self.si_pca_data[:, 2]))),
                     'r',
                     label='Si ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(utils.mean_val(self.si_pca_data[:, 2]),
                                                                                np.sqrt(utils.variance(
                                                                                    self.si_pca_data[:, 2]))))
        ax_pc_1.plot(alpha, utils.normal_dist(alpha, utils.mean_val(self.al_pca_data[:, 2]),
                                              np.sqrt(utils.variance(self.al_pca_data[:, 2]))),
                     'g',
                     label='Al ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(utils.mean_val(self.al_pca_data[:, 2]),
                                                                                np.sqrt(utils.variance(
                                                                                    self.al_pca_data[:, 2]))))
        ax_pc_1.plot(alpha, utils.normal_dist(alpha, utils.mean_val(self.mg_pca_data[:, 2]),
                                              np.sqrt(utils.variance(self.mg_pca_data[:, 2]))),
                     'm',
                     label='Mg ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(utils.mean_val(self.mg_pca_data[:, 2]),
                                                                                np.sqrt(utils.variance(
                                                                                    self.mg_pca_data[:, 2]))))

        ax_pc_1.set_title('Principle component 1 fitted density')
        ax_pc_1.set_xlabel('Principle component 1')
        ax_pc_1.legend()

        ax_pc_2.plot(alpha, utils.normal_dist(alpha, utils.mean_val(self.cu_pca_data[:, 3]),
                                              np.sqrt(utils.variance(self.cu_pca_data[:, 3]))),
                     'y',
                     label='Cu ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(utils.mean_val(self.cu_pca_data[:, 3]),
                                                                                np.sqrt(utils.variance(
                                                                                    self.cu_pca_data[:, 3]))))
        ax_pc_2.plot(alpha, utils.normal_dist(alpha, utils.mean_val(self.si_pca_data[:, 3]),
                                              np.sqrt(utils.variance(self.si_pca_data[:, 3]))),
                     'r',
                     label='Si ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(utils.mean_val(self.si_pca_data[:, 3]),
                                                                                np.sqrt(utils.variance(
                                                                                    self.si_pca_data[:, 3]))))
        ax_pc_2.plot(alpha, utils.normal_dist(alpha, utils.mean_val(self.al_pca_data[:, 3]),
                                              np.sqrt(utils.variance(self.al_pca_data[:, 3]))),
                     'g',
                     label='Al ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(utils.mean_val(self.al_pca_data[:, 3]),
                                                                                np.sqrt(utils.variance(
                                                                                    self.al_pca_data[:, 3]))))
        ax_pc_2.plot(alpha, utils.normal_dist(alpha, utils.mean_val(self.mg_pca_data[:, 3]),
                                              np.sqrt(utils.variance(self.mg_pca_data[:, 3]))),
                     'm',
                     label='Mg ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(utils.mean_val(self.mg_pca_data[:, 3]),
                                                                                np.sqrt(utils.variance(
                                                                                    self.mg_pca_data[:, 3]))))

        ax_pc_2.set_title('Principle component 2 fitted density')
        ax_pc_2.set_xlabel('Principle component 2')
        ax_pc_2.legend()

        ax_scatter.scatter(self.cu_pca_data[:, 2], self.cu_pca_data[:, 3], c='y', label='Cu', s=4)
        ax_scatter.scatter(self.si_pca_data[:, 2], self.si_pca_data[:, 3], c='r', label='Si', s=4)
        ax_scatter.scatter(self.al_pca_data[:, 2], self.al_pca_data[:, 3], c='g', label='Al', s=4)
        ax_scatter.scatter(self.mg_pca_data[:, 2], self.mg_pca_data[:, 3], c='m', label='Mg', s=4)

        ax_scatter.set_title('Scatter-plot of two first principle components')
        ax_scatter.set_xlabel('PC 1')
        ax_scatter.set_ylabel('PC 2')
        ax_scatter.legend()

        fig.suptitle('Principle component analysis')

        logger.info('Plotted PCA alpha over {} files and {} vertices!'.format(self.num_files, self.num_vertices))

        plt.show()



