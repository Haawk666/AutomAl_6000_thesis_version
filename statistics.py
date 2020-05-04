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

    model = DataManager(reformatted_data, category_titles)
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
                self.covar_matrix[dimension_1, dimension_2] /= self.n - 1

        self.covar_matrix_determinant = np.linalg.det(self.covar_matrix)
        self.inverse_covar_matrix = np.linalg.inv(self.covar_matrix)
        self.covar_matrix_eigenvalues, self.covar_matrix_eigenvectors = np.linalg.eig(self.covar_matrix)
        idx = np.argsort(self.covar_matrix_eigenvalues)[::-1]
        self.covar_matrix_eigenvalues = self.covar_matrix_eigenvalues[idx]
        self.covar_matrix_eigenvectors = self.covar_matrix_eigenvectors[:, idx]


class DataManager:
    """A class designed to gather, handle, transform, plot, model and export data from multiple images for AutomAl 6000

    The :code:`filter_` argument can be used to access particular sub-sets of the available data. The filter is passed
    as a list of 7 boolean values. The table below shows the effect of each value togheter with the default behavior if
    no filter is passed

    ======= =========================== ===============================
    x       default :code:`filter_[x]`  Effect
    ======= =========================== ===============================
    0       False                       Include edge columns
    1       True                        Include matrix columns
    2       True
    3       True
    4       True
    5       True
    6       True
    ======= =========================== ===============================

    The :code:`keys` argument is a list of strings that determine which of the data attributes that the data manager
    should include. The available keys are displayed in the table below, as well as an indication of which attributes
    will be included if no keys are passed. Only attributes that are singular real-valued will be used for statistical
    analysis, indicated by the \'physical\' column.

    =========================================== =========================================================== ======================= ============
    Key                                         Attribute - type                                            Included by default     Physical
    =========================================== =========================================================== ======================= ============
    :code:`'i'`                                 :code:`vertex.i` - int                                      No                      No
    :code:`'r'`                                 :code:`vertex.r` - int                                      No                      No
    :code:`'species_index'`                     :code:`vertex.species_index` - int                          No                      No
    :code:`'species_variant'`                   :code:`vertex.species_variant` - int                        No                      No
    :code:`'advanced_category_index'`           :code:`vertex.advanced_category_index` - int                Yes                     No
    :code:`'alpha_angles'`                      :code:`vertex.alpha_angles` - [float]                       No                      No
    :code:`'alpha_max'`                         :code:`vertex.alpha_max` - float                            Yes                     Yes
    :code:`'alpha_min'`                         :code:`vertex.alpha_min` - float                            Yes                     Yes
    :code:`'theta_angles'`                      :code:`vertex.theta_angles` - [float]                       No                      No
    :code:`'theta_max'`                         :code:`vertex.theta_max` - float                            Yes                     Yes
    :code:`'theta_min'`                         :code:`vertex.theta_min` - float                            Yes                     Yes
    :code:`'theta_angle_variance'`              :code:`vertex.theta_angle_variance` - float                 No                      No
    :code:`'theta_angle_mean'`                  :code:`vertex.theta_angle_mean` - float                     Yes                     Yes
    :code:`'peak_gamma'`                        :code:`vertex.peak_gamma`                                   No                      No
    :code:`'avg_gamma'`                         :code:`vertex.avg_gamma`                                    No                      No
    :code:`'normalized_peak_gamma'`             :code:`vertex.normalized_peak_gamma` - float                Yes                     Yes
    :code:`'normalized_avg_gamma'`              :code:`vertex.normalized_avg_gamma` - float                 Yes                     Yes
    :code:`'redshift'`                          :code:`vertex.redshift` - float                             No                      No
    :code:`'avg_central_separation'`            :code:`vertex.avg_central_separation` - float               Yes                     Yes
    :code:`'zeta'`                              :code:`vertex.zeta` - bool                                  No                      No
    :code:`'im_coor_x'`                         :code:`vertex.im_coor_x` - float                            No                      No
    :code:`'im_coor_y'`                         :code:`vertex.im_coor_y` - float                            No                      No
    :code:`'im_coor_z'`                         :code:`vertex.im_coor_z` - float                            No                      No
    :code:`'spatial_coor_x'`                    :code:`vertex.spatial_coor_x` - float                       No                      No
    :code:`'spatial_coor_y'`                    :code:`vertex.spatial_coor_y` - float                       No                      No
    :code:`'spatial_coor_z'`                    :code:`vertex.spatial_coor_z` - float                       No                      No
    ========================================== =========================================================== ======================= ============

    The :code:`categorization` keyword determines how the data will be categorized. The options are

    ======================= ===========================================================================
    :code:`categorization`  Explanation
    ======================= ===========================================================================
    :code:`'advanced'`      Categorize data by the :code:`vertex.advanced_category_index` attribute
    :code:`'simple'`        Categorize data by the :code:`vertex.species_index` attribute
    ======================= ===========================================================================

    note
    ----------
    This dataManager is designed to handle vertex-based data. To analyse Arc-centered data, use the 'ArcDataManager' class


    :param files: String of full filenames separated by newline character. Data will be gathered from each of the files.
    :param filter_: (Optional, default: None) Filter inclusion of data by the tokens in the above table.
    :param keys: (Optional, default: None) The keys of the data attributes to include
    :param save_filename: (Optional, default: 'model') Pickle the model data with this filename
    :param categorization: (Optional, default: 'advanced') Categorization keyword

    :type files: string
    :type filter_: [bool]
    :type keys: [string]
    :type save_filename: string
    :type categorization: string

    :returns DataManager object:
    :rtype :code:`<statistics.DataManager>`

    """

    def __init__(self, files, filter_=None, keys=None, save_filename='model', categorization='advanced'):

        if filter_ is None:
            self.filter_ = [False, True, True, True, True, True, True]
        else:
            self.filter_ = filter_

        if keys is None:
            self.keys = ['advanced_category_index', 'alpha_max', 'alpha_min', 'theta_max', 'theta_min',
                         'theta_angle_mean', 'normalized_peak_gamma', 'normalized_avg_gamma', 'avg_central_separation']
        else:
            self.keys = keys
        self.attribute_keys = self.determine_attribute_keys()
        self.k = len(self.attribute_keys)

        self.files = files
        self.save_filename = save_filename

        self.original_dict_data = self.collect_data()
        self.n = len(self.original_dict_data)

        self.categorization = categorization
        if self.categorization == 'advanced':
            self.category_titles = ['Si_1', 'Si_2', 'Cu', 'Al_1', 'Al_2', 'Mg_1', 'Mg_2']
        elif self.categorization == 'simple':
            self.category_titles = ['Si', 'Cu', 'Al', 'Mg']
        else:
            self.category_titles = ['Si', 'Cu', 'Al', 'Mg']
            logger.warning('Unrecognized categorization. Using \'simple\'...')
        self.num_data_categories = len(self.category_titles)

        self.matrix_data, self.category_n = self.vectorize_data()
        self.concatenated_matrix_data = self.concatenate_categories()

        self.uncategorized_normal_dist = MultivariateNormalDist(self.concatenated_matrix_data, 'All categories')






        self.mean_shifted_concatenated_data = self.concatenated_data
        self.mean_shifted_data = []

        self.uncategorized_normal_dist = MultivariateNormalDist(self.concatenated_data, 'All categories')

        self.shift_data()

        self.composite_model = []
        self.alpha_model = []
        self.separation_model = []

        for i, category in enumerate(self.data):
            self.composite_model.append(MultivariateNormalDist(category, self.category_titles[i]))
            self.alpha_model.append(MultivariateNormalDist(category[0:1, :], ['alpha_max', 'alpha_min']))
            self.separation_model.append(MultivariateNormalDist(category[7, :], ['avg_central_separation']))

    def determine_attribute_keys(self):
        attributes = []
        if 'alpha_max' in self.keys:
            attributes.append('alpha_max')
        if 'alpha_min' in self.keys:
            attributes.append('alpha_min')
        if 'theta_max' in self.keys:
            attributes.append('theta_max')
        if 'theta_min' in self.keys:
            attributes.append('theta_min')
        if 'theta_angle_mean' in self.keys:
            attributes.append('theta_angle_mean')
        if 'normalized_peak_gamma' in self.keys:
            attributes.append('normalized_peak_gamma')
        if 'normalized_avg_gamma' in self.keys:
            attributes.append('normalized_avg_gamma')
        if 'avg_central_separation' in self.keys:
            attributes.append('avg_central_separation')
        return attributes

    def collect_data(self):
        data = []
        for file in self.files.splitlines(keepends=False):
            instance = core.SuchSoftware.load(file)
            image_data = instance.graph.calc_condensed_property_data(filter_=self.filter_, recalc=True, keys=self.keys)
            data += image_data
        return data

    def vectorize_data(self):
        data = []
        for category in range(0, self.num_data_categories):
            data.append([])
            for attribute in range(0, self.k):
                data[category].append([])
        for data_item in self.original_dict_data:
            for h, attribute in enumerate(self.attribute_keys):
                if self.categorization == 'advanced':
                    data[data_item['advanced_category_index']][h] = data_item[attribute]
                else:
                    if data_item['species_index'] == 0:
                        category_index = 0
                    elif data_item['species_index'] == 1:
                        category_index = 1
                    elif data_item['species_index'] == 3:
                        category_index = 2
                    elif data_item['species_index'] == 5:
                        category_index = 3
                    else:
                        logger.error('Error in category assignment!')
                        category_index = 4
                    data[category_index][h] = data_item[attribute]
        matrix_data = []
        category_n = []
        for category_data in data:
            matrix_data.append(np.array(category_data))
            category_n.append(matrix_data[-1].shape[1])
        return matrix_data, category_n

    def prepare_data(self, data):
        matrix_data = []
        for category in data:
            matrix = np.array(category)
            matrix_data.append(matrix)
        self.data = matrix_data

    def concatenate_categories(self):
        concatenated_data = self.data[0]
        for i, category in enumerate(self.category_titles):
            if not i == 0:
                concatenated_data = np.concatenate((concatenated_data, self.data[i]), axis=1)
        return concatenated_data

    def shift_data(self):
        for attribute in range(0, 8):
            for data_item in self.mean_shifted_data[attribute, :]:
                data_item -= self.uncategorized_normal_dist.means[attribute]
        self.mean_shifted_data = []
        for category in self.data:
            self.mean_shifted_data.append(category)
            for attribute in range(0, 8):
                for h, data_item in enumerate(category[attribute, :]):
                    self.mean_shifted_data[-1][attribute, h] -= self.uncategorized_normal_dist.means[attribute]

    def plot_data(self):
        pass

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

    def save(self):
        with open(self.savefile, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename_full):
        """Load a DataManager instance from a pickle-file.

        :param filename_full: Path-name of the file to be loaded.
        :type filename_full: string

        :returns DataManager object:
        :rtype :code:`<statistics.DataManager>`

        """
        with open(filename_full, 'rb') as f:
            obj = pickle.load(f)
        return obj


class Plotting:

    def __init__(self, data):

        self.data = data

    def plot_interatomic_separations(self, type_='distribution'):
        logger.info('Generating plots...')

        if type_ == 'distribution':

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

        elif type_ == 'box':

            fig, ax = plt.subplots()

            tick_labels = ['Si <-> Si', 'Cu <-> Cu', 'Al <-> Al', 'Mg <-> Mg', 'Si <-> Cu', 'Si <-> Al', 'Si <-> Mg',
                           'Cu <-> Al', 'Cu <-> Mg', 'Al <-> Mg']

            ax.boxplot([self.si_si, self.cu_cu, self.al_al, self.mg_mg, self.si_cu, self.si_al, self.si_mg, self.cu_al, self.cu_mg, self.al_mg])

            ax.set_xticklabels(tick_labels, rotation=45, fontsize=12, fontdict={'horizontalalignment': 'right'})

            ax.plot(1, 2 * core.SuchSoftware.si_radii, color='r', marker='*', markeredgecolor='k', markersize=12,
                    linestyle='', label='Hard-sphere model values')
            ax.plot(2, 2 * core.SuchSoftware.cu_radii, color='r', marker='*', markeredgecolor='k', markersize=12)
            ax.plot(3, 2 * core.SuchSoftware.al_radii, color='r', marker='*', markeredgecolor='k', markersize=12)
            ax.plot(4, 2 * core.SuchSoftware.mg_radii, color='r', marker='*', markeredgecolor='k', markersize=12)
            ax.plot(5, core.SuchSoftware.si_radii + core.SuchSoftware.cu_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax.plot(6, core.SuchSoftware.si_radii + core.SuchSoftware.al_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax.plot(7, core.SuchSoftware.si_radii + core.SuchSoftware.mg_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax.plot(8, core.SuchSoftware.cu_radii + core.SuchSoftware.al_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax.plot(9, core.SuchSoftware.cu_radii + core.SuchSoftware.mg_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax.plot(10, core.SuchSoftware.al_radii + core.SuchSoftware.mg_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)

            ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
            ax.set_axisbelow(True)

            ax.set_ylabel('Inter-atomic distance (pm)')
            ax.legend(loc='upper left')

            fig.subplots_adjust(bottom=0.20)
            fig.suptitle('Box plot of inter-atomic distance data')

            plt.show()

        elif type_ == 'scatter':

            fig, ax = plt.subplots()

            tick_labels = ['', 'Si <-> Si', 'Cu <-> Cu', 'Al <-> Al', 'Mg <-> Mg', 'Si <-> Cu', 'Si <-> Al', 'Si <-> Mg',
                           'Cu <-> Al', 'Cu <-> Mg', 'Al <-> Mg']

            ax.plot([0] * len(self.si_si), self.si_si, marker='o', markeredgecolor='k', markersize=4, linestyle='', fillstyle='none')
            ax.plot([1] * len(self.cu_cu), self.cu_cu, marker='o', markeredgecolor='k', markersize=4, linestyle='', fillstyle='none')
            ax.plot([2] * len(self.al_al), self.al_al, marker='o', markeredgecolor='k', markersize=4, linestyle='', fillstyle='none')
            ax.plot([3] * len(self.mg_mg), self.mg_mg, marker='o', markeredgecolor='k', markersize=4, linestyle='', fillstyle='none')
            ax.plot([4] * len(self.si_cu), self.si_cu, marker='o', markeredgecolor='k', markersize=4, linestyle='', fillstyle='none')
            ax.plot([5] * len(self.si_al), self.si_al, marker='o', markeredgecolor='k', markersize=4, linestyle='', fillstyle='none')
            ax.plot([6] * len(self.si_mg), self.si_mg, marker='o', markeredgecolor='k', markersize=4, linestyle='', fillstyle='none')
            ax.plot([7] * len(self.cu_al), self.cu_al, marker='o', markeredgecolor='k', markersize=4, linestyle='', fillstyle='none')
            ax.plot([8] * len(self.cu_mg), self.cu_mg, marker='o', markeredgecolor='k', markersize=4, linestyle='', fillstyle='none')
            ax.plot([9] * len(self.al_mg), self.al_mg, marker='o', markeredgecolor='k', markersize=4, linestyle='', fillstyle='none')

            ax.plot(0, 2 * core.SuchSoftware.si_radii, color='r', marker='*', markeredgecolor='k', markersize=12,
                    linestyle='', label='Hard-sphere model values')
            ax.plot(1, 2 * core.SuchSoftware.cu_radii, color='r', marker='*', markeredgecolor='k', markersize=12)
            ax.plot(2, 2 * core.SuchSoftware.al_radii, color='r', marker='*', markeredgecolor='k', markersize=12)
            ax.plot(3, 2 * core.SuchSoftware.mg_radii, color='r', marker='*', markeredgecolor='k', markersize=12)
            ax.plot(4, core.SuchSoftware.si_radii + core.SuchSoftware.cu_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax.plot(5, core.SuchSoftware.si_radii + core.SuchSoftware.al_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax.plot(6, core.SuchSoftware.si_radii + core.SuchSoftware.mg_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax.plot(7, core.SuchSoftware.cu_radii + core.SuchSoftware.al_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax.plot(8, core.SuchSoftware.cu_radii + core.SuchSoftware.mg_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax.plot(9, core.SuchSoftware.al_radii + core.SuchSoftware.mg_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)

            ax.xaxis.set_major_locator(tick.MultipleLocator(1))
            ax.set_xticklabels(tick_labels, rotation=45, fontsize=12, fontdict={'horizontalalignment': 'right'})

            ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
            ax.set_axisbelow(True)

            ax.set_ylabel('Inter-atomic distance (pm)')
            ax.legend(loc='upper left')

            fig.subplots_adjust(bottom=0.20)
            fig.suptitle('Scatter-plot of inter-atomic distance data')

            plt.show()

        elif type_ == 'all':

            distance = np.linspace(200, 400, 1000)

            fig = plt.figure(constrained_layout=True)
            gs = GridSpec(2, 2, figure=fig)
            ax_same = fig.add_subplot(gs[0, 0])
            ax_pairs_1 = fig.add_subplot(gs[1, 0])
            ax_box = fig.add_subplot(gs[0, 1])
            ax_scatter = fig.add_subplot(gs[1, 1])

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

            ax_pairs_1.plot(distance, utils.normal_dist(distance, self.si_cu_mean, self.si_cu_std), 'r',
                            label='Si <-> Cu')
            ax_pairs_1.plot(distance, utils.normal_dist(distance, self.si_al_mean, self.si_al_std), 'k',
                            label='Si <-> Al')
            ax_pairs_1.plot(distance, utils.normal_dist(distance, self.si_mg_mean, self.si_mg_std), 'c',
                            label='Si <-> Mg')
            ax_pairs_1.plot(distance, utils.normal_dist(distance, self.cu_al_mean, self.cu_al_std), 'y',
                            label='Cu <-> Al')
            ax_pairs_1.plot(distance, utils.normal_dist(distance, self.cu_al_mean, self.cu_al_std), 'b',
                            label='Cu <-> Mg')
            ax_pairs_1.plot(distance, utils.normal_dist(distance, self.al_mg_mean, self.al_mg_std), 'g',
                            label='Al <-> Mg')

            ax_pairs_1.axvline(x=core.SuchSoftware.si_radii + core.SuchSoftware.cu_radii, c='r')
            ax_pairs_1.axvline(x=core.SuchSoftware.si_radii + core.SuchSoftware.al_radii, c='k')
            ax_pairs_1.axvline(x=core.SuchSoftware.si_radii + core.SuchSoftware.mg_radii, c='c')
            ax_pairs_1.axvline(x=core.SuchSoftware.cu_radii + core.SuchSoftware.al_radii, c='y')
            ax_pairs_1.axvline(x=core.SuchSoftware.cu_radii + core.SuchSoftware.mg_radii, c='b')
            ax_pairs_1.axvline(x=core.SuchSoftware.al_radii + core.SuchSoftware.mg_radii, c='g')

            ax_pairs_1.set_title('Un-similar species pairs')
            ax_pairs_1.set_xlabel('Inter-atomic distance (pm)')
            ax_pairs_1.legend()

            tick_labels = ['Si <-> Si', 'Cu <-> Cu', 'Al <-> Al', 'Mg <-> Mg', 'Si <-> Cu', 'Si <-> Al', 'Si <-> Mg',
                           'Cu <-> Al', 'Cu <-> Mg', 'Al <-> Mg']

            ax_box.boxplot([self.si_si, self.cu_cu, self.al_al, self.mg_mg, self.si_cu, self.si_al, self.si_mg, self.cu_al,
                        self.cu_mg, self.al_mg])

            ax_box.set_xticklabels(tick_labels, rotation=45, fontsize=10, fontdict={'horizontalalignment': 'right'})

            ax_box.plot(1, 2 * core.SuchSoftware.si_radii, color='r', marker='*', markeredgecolor='k', markersize=12,
                    linestyle='', label='Hard-sphere model values')
            ax_box.plot(2, 2 * core.SuchSoftware.cu_radii, color='r', marker='*', markeredgecolor='k', markersize=12)
            ax_box.plot(3, 2 * core.SuchSoftware.al_radii, color='r', marker='*', markeredgecolor='k', markersize=12)
            ax_box.plot(4, 2 * core.SuchSoftware.mg_radii, color='r', marker='*', markeredgecolor='k', markersize=12)
            ax_box.plot(5, core.SuchSoftware.si_radii + core.SuchSoftware.cu_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax_box.plot(6, core.SuchSoftware.si_radii + core.SuchSoftware.al_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax_box.plot(7, core.SuchSoftware.si_radii + core.SuchSoftware.mg_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax_box.plot(8, core.SuchSoftware.cu_radii + core.SuchSoftware.al_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax_box.plot(9, core.SuchSoftware.cu_radii + core.SuchSoftware.mg_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax_box.plot(10, core.SuchSoftware.al_radii + core.SuchSoftware.mg_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)

            ax_box.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
            ax_box.set_axisbelow(True)

            ax_box.set_ylabel('Inter-atomic distance (pm)')
            ax_box.legend(loc='upper left')
            ax_box.set_title('Box-plot of inter-atomic distance data')

            tick_labels = ['', 'Si <-> Si', 'Cu <-> Cu', 'Al <-> Al', 'Mg <-> Mg', 'Si <-> Cu', 'Si <-> Al', 'Si <-> Mg',
                           'Cu <-> Al', 'Cu <-> Mg', 'Al <-> Mg']

            ax_scatter.plot([1] * len(self.si_si), self.si_si, marker='o', markeredgecolor='k', markersize=4, linestyle='',
                    fillstyle='none')
            ax_scatter.plot([2] * len(self.cu_cu), self.cu_cu, marker='o', markeredgecolor='k', markersize=4, linestyle='',
                    fillstyle='none')
            ax_scatter.plot([3] * len(self.al_al), self.al_al, marker='o', markeredgecolor='k', markersize=4, linestyle='',
                    fillstyle='none')
            ax_scatter.plot([4] * len(self.mg_mg), self.mg_mg, marker='o', markeredgecolor='k', markersize=4, linestyle='',
                    fillstyle='none')
            ax_scatter.plot([5] * len(self.si_cu), self.si_cu, marker='o', markeredgecolor='k', markersize=4, linestyle='',
                    fillstyle='none')
            ax_scatter.plot([6] * len(self.si_al), self.si_al, marker='o', markeredgecolor='k', markersize=4, linestyle='',
                    fillstyle='none')
            ax_scatter.plot([7] * len(self.si_mg), self.si_mg, marker='o', markeredgecolor='k', markersize=4, linestyle='',
                    fillstyle='none')
            ax_scatter.plot([8] * len(self.cu_al), self.cu_al, marker='o', markeredgecolor='k', markersize=4, linestyle='',
                    fillstyle='none')
            ax_scatter.plot([9] * len(self.cu_mg), self.cu_mg, marker='o', markeredgecolor='k', markersize=4, linestyle='',
                    fillstyle='none')
            ax_scatter.plot([10] * len(self.al_mg), self.al_mg, marker='o', markeredgecolor='k', markersize=4, linestyle='',
                    fillstyle='none')

            ax_scatter.plot(1, 2 * core.SuchSoftware.si_radii, color='r', marker='*', markeredgecolor='k', markersize=12,
                    linestyle='', label='Hard-sphere model values')
            ax_scatter.plot(2, 2 * core.SuchSoftware.cu_radii, color='r', marker='*', markeredgecolor='k', markersize=12)
            ax_scatter.plot(3, 2 * core.SuchSoftware.al_radii, color='r', marker='*', markeredgecolor='k', markersize=12)
            ax_scatter.plot(4, 2 * core.SuchSoftware.mg_radii, color='r', marker='*', markeredgecolor='k', markersize=12)
            ax_scatter.plot(5, core.SuchSoftware.si_radii + core.SuchSoftware.cu_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax_scatter.plot(6, core.SuchSoftware.si_radii + core.SuchSoftware.al_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax_scatter.plot(7, core.SuchSoftware.si_radii + core.SuchSoftware.mg_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax_scatter.plot(8, core.SuchSoftware.cu_radii + core.SuchSoftware.al_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax_scatter.plot(9, core.SuchSoftware.cu_radii + core.SuchSoftware.mg_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)
            ax_scatter.plot(10, core.SuchSoftware.al_radii + core.SuchSoftware.mg_radii, color='r', marker='*',
                    markeredgecolor='k', markersize=12)

            ax_scatter.xaxis.set_major_locator(tick.MultipleLocator(1))
            ax_scatter.set_xticklabels(tick_labels, rotation=45, fontsize=10, fontdict={'horizontalalignment': 'right'})

            ax_scatter.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
            ax_scatter.set_axisbelow(True)

            ax_scatter.set_title('Scatter-plot of inter-atomic distance data')
            ax_scatter.set_ylabel('Inter-atomic distance (pm)')
            ax_scatter.legend(loc='upper left')

            fig.suptitle('All inter-atomic distance plots')

            plt.show()

        else:

            logger.error('Unkonwn plot-type!')

    def plot_gamma(self):
        logger.info('Generating plots...')

        gamma = np.linspace(0, 1, 1000)

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig)
        ax_peak = fig.add_subplot(gs[0, 0])
        ax_avg = fig.add_subplot(gs[1, 0])
        ax_scatter = fig.add_subplot(gs[:, 1])

        ax_peak.plot(gamma, utils.normal_dist(gamma, self.cu_peak_gamma_mean, self.cu_peak_gamma_std), 'y', label='Cu ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.cu_peak_gamma_mean, self.cu_peak_gamma_std))
        ax_peak.plot(gamma, utils.normal_dist(gamma, self.si_peak_gamma_mean, self.si_peak_gamma_std), 'r', label='Si ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.si_peak_gamma_mean, self.si_peak_gamma_std))
        ax_peak.plot(gamma, utils.normal_dist(gamma, self.al_peak_gamma_mean, self.al_peak_gamma_std), 'g', label='Al ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.al_peak_gamma_mean, self.al_peak_gamma_std))
        ax_peak.plot(gamma, utils.normal_dist(gamma, self.mg_peak_gamma_mean, self.mg_peak_gamma_std), 'm', label='Mg ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.mg_peak_gamma_mean, self.mg_peak_gamma_std))

        ax_peak.set_title('peak z-contrast fitted distributions')
        ax_peak.set_xlabel('peak z-contrast (normalized $\in (0, 1)$)')
        ax_peak.legend()

        ax_avg.plot(gamma, utils.normal_dist(gamma, self.cu_avg_gamma_mean, self.cu_avg_gamma_std), 'y', label='Cu ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.cu_avg_gamma_mean, self.cu_avg_gamma_std))
        ax_avg.plot(gamma, utils.normal_dist(gamma, self.si_avg_gamma_mean, self.si_avg_gamma_std), 'r', label='Si ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.si_avg_gamma_mean, self.si_avg_gamma_std))
        ax_avg.plot(gamma, utils.normal_dist(gamma, self.al_avg_gamma_mean, self.al_avg_gamma_std), 'g', label='Al ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.al_avg_gamma_mean, self.al_avg_gamma_std))
        ax_avg.plot(gamma, utils.normal_dist(gamma, self.mg_avg_gamma_mean, self.mg_avg_gamma_std), 'm', label='Mg ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.mg_avg_gamma_mean, self.mg_avg_gamma_std))

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
        ax_scatter.set_xlim([0, 1])
        ax_scatter.set_ylim([0, 1])
        ax_scatter.legend()

        fig.suptitle('Scatter plot of peak-avg contrasts')

        plt.show()

    def plot_alpha(self):

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
        ax_min.plot(alpha, utils.normal_dist(alpha, self.al_1_min_mean, self.al_1_min_std), 'g',
                    label='Al$_1$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.al_1_min_mean, self.al_1_min_std))
        ax_min.plot(alpha, utils.normal_dist(alpha, self.al_2_min_mean, self.al_2_min_std), 'g',
                    label='Al$_2$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.al_2_min_mean, self.al_2_min_std))
        ax_min.plot(alpha, utils.normal_dist(alpha, self.mg_1_min_mean, self.mg_1_min_std), 'm',
                    label='Mg$_1$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.mg_1_min_mean, self.mg_1_min_std))
        ax_min.plot(alpha, utils.normal_dist(alpha, self.mg_2_min_mean, self.mg_2_min_std), 'c',
                    label='Mg$_2$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.mg_2_min_mean, self.mg_2_min_std))

        ax_min.set_title('Minimum central angles fitted density')
        ax_min.set_xlabel('Min angle (radians)')
        ax_min.legend()

        ax_max.plot(alpha, utils.normal_dist(alpha, self.cu_max_mean, self.cu_max_std), 'y',
                    label='Cu ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.cu_max_mean, self.cu_max_std))
        ax_max.plot(alpha, utils.normal_dist(alpha, self.si_1_max_mean, self.si_1_max_std), 'r',
                    label='Si$_1$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.si_1_max_mean, self.si_1_max_std))
        ax_max.plot(alpha, utils.normal_dist(alpha, self.si_2_max_mean, self.si_2_max_std), 'k',
                    label='Si$_2$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.si_2_max_mean, self.si_2_max_std))
        ax_max.plot(alpha, utils.normal_dist(alpha, self.al_1_max_mean, self.al_1_max_std), 'g',
                    label='Al$_1$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.al_1_max_mean, self.al_1_max_std))
        ax_max.plot(alpha, utils.normal_dist(alpha, self.al_2_max_mean, self.al_2_max_std), 'g',
                    label='Al$_2$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.al_2_max_mean, self.al_2_max_std))
        ax_max.plot(alpha, utils.normal_dist(alpha, self.mg_1_max_mean, self.mg_1_max_std), 'm',
                    label='Mg$_1$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.mg_1_max_mean, self.mg_1_max_std))
        ax_max.plot(alpha, utils.normal_dist(alpha, self.mg_2_max_mean, self.mg_2_max_std), 'c',
                    label='Mg$_2$ ($\mu$ = ' + '{:.2f}, $\sigma$ = {:.2f})'.format(self.mg_2_max_mean, self.mg_2_max_std))

        ax_max.set_title('Maximum central angles fitted density')
        ax_max.set_xlabel('max angle (radians)')
        ax_max.legend()

        ax_scatter.scatter(self.cu_min_angles, self.cu_max_angles, c='y', label='Cu', s=8)
        ax_scatter.scatter(self.si_1_min_angles, self.si_1_max_angles, c='r', label='Si$_1$', s=8)
        ax_scatter.scatter(self.si_2_min_angles, self.si_2_max_angles, c='k', label='Si$_2$', s=8)
        ax_scatter.scatter(self.al_1_min_angles, self.al_1_max_angles, c='g', label='Al', s=8)
        ax_scatter.scatter(self.al_2_min_angles, self.al_2_max_angles, c='g', label='Al', s=8)
        ax_scatter.scatter(self.mg_1_min_angles, self.mg_1_max_angles, c='m', label='Mg$_1$', s=8)
        ax_scatter.scatter(self.mg_2_min_angles, self.mg_2_max_angles, c='c', label='Mg$_2$', s=8)

        ax_scatter.set_title('Scatter-plot of min-max angles')
        ax_scatter.set_xlabel('Min angle (radians)')
        ax_scatter.set_ylabel('max angle (radians)')
        ax_scatter.legend()

        if self.angle_mode == 'alpha':
            fig.suptitle('Alpha min/max summary')
        else:
            fig.suptitle('Theta min/max summary')

        logger.info('Plotted min/max over {} files and {} vertices!'.format(self.number_of_files, self.number_of_vertices))

        plt.show()

    def plot_theta(self):
        pass

    def plot_avg_central_separation(self):
        pass



