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

    def prediction(self, args):
        prob = utils.multivariate_normal_dist(args, self.means, self.covar_matrix_determinant, self.inverse_covar_matrix)
        return prob


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

    def __init__(self, files, filter_=None, keys=None, save_filename='model', categorization='advanced', recalc=False):

        if filter_ is None:
            self.filter_ = [False, True, True, True, True, True, True]
        else:
            self.filter_ = filter_

        if keys is None:
            self.keys = ['advanced_category_index', 'alpha_max', 'alpha_min', 'theta_max', 'theta_min',
                         'theta_angle_mean', 'normalized_peak_gamma', 'normalized_avg_gamma', 'avg_central_separation']
        else:
            self.keys = keys
        self.attribute_keys, self.attribute_units = self.determine_attribute_keys()
        self.k = len(self.attribute_keys)

        self.files = files
        self.save_filename = save_filename
        self.recalc = recalc

        self.original_dict_data = self.collect_data()
        self.n = len(self.original_dict_data)

        self.categorization = categorization
        if self.categorization == 'advanced':
            self.category_titles = ['Si_1', 'Si_2', 'Cu', 'Al_1', 'Al_2', 'Mg_1', 'Mg_2']
            self.colours = ['r', 'lightsalmon', 'y', 'g', 'mediumseagreen', 'm', 'plum']
        elif self.categorization == 'simple':
            self.category_titles = ['Si', 'Cu', 'Al', 'Mg']
            self.colours = ['r', 'y', 'g', 'm']
        else:
            self.category_titles = ['Si', 'Cu', 'Al', 'Mg']
            self.colours = ['r', 'y', 'g', 'm']
            logger.warning('Unrecognized categorization. Using \'simple\'...')
        self.num_data_categories = len(self.category_titles)

        self.matrix_data = self.vectorize_data()
        self.concatenated_matrix_data = self.concatenate_categories()
        self.normalized_concatenated_matrix_data = np.array(self.concatenated_matrix_data)

        self.uncategorized_normal_dist = MultivariateNormalDist(self.concatenated_matrix_data, 'All categories')

        self.norm_data()
        self.normalized_uncategorized_normal_dist = MultivariateNormalDist(self.normalized_concatenated_matrix_data, 'All categories')

        self.composite_model = []
        self.alpha_model = []
        self.avg_central_separation_model = []

        for c, category_data in enumerate(self.matrix_data):
            self.composite_model.append(MultivariateNormalDist(category_data, self.category_titles[c]))
            if 'alpha_max' in self.attribute_keys and 'alpha_min' in self.attribute_keys:
                max_index = self.attribute_keys.index('alpha_max')
                min_index = self.attribute_keys.index('alpha_min')
                self.alpha_model.append(MultivariateNormalDist(category_data[np.array([max_index, min_index]), :], self.category_titles[c]))
            if 'avg_central_separation' in self.attribute_keys:
                attr_index = self.attribute_keys.index('avg_central_separation')
                self.avg_central_separation_model.append(MultivariateNormalDist(category_data[attr_index, :].reshape(1, category_data[attr_index, :].shape[0]), self.category_titles[c]))

    def determine_attribute_keys(self):
        attributes = []
        units = []
        if 'alpha_max' in self.keys:
            attributes.append('alpha_max')
            units.append('(radians)')
        if 'alpha_min' in self.keys:
            attributes.append('alpha_min')
            units.append('(radians)')
        if 'theta_max' in self.keys:
            attributes.append('theta_max')
            units.append('(radians)')
        if 'theta_min' in self.keys:
            attributes.append('theta_min')
            units.append('(radians)')
        if 'theta_angle_mean' in self.keys:
            attributes.append('theta_angle_mean')
            units.append('(radians)')
        if 'normalized_peak_gamma' in self.keys:
            attributes.append('normalized_peak_gamma')
            units.append('(normalized intensity)')
        if 'normalized_avg_gamma' in self.keys:
            attributes.append('normalized_avg_gamma')
            units.append('(normalized intensity)')
        if 'avg_central_separation' in self.keys:
            attributes.append('avg_central_separation')
            units.append('nm')
        return attributes, units

    def collect_data(self):
        data = []
        for file in self.files.splitlines(keepends=False):
            instance = core.SuchSoftware.load(file)
            image_data = instance.graph.calc_condensed_property_data(filter_=self.filter_, recalc=self.recalc, keys=self.keys)
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
                    data[data_item['advanced_category_index']][h].append(data_item[attribute])
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
                    data[category_index][h].append(data_item[attribute])
        matrix_data = []
        for category_data in data:
            matrix_data.append(np.array(category_data))
        return matrix_data

    def concatenate_categories(self):
        concatenated_data = self.matrix_data[0]
        for i, category in enumerate(self.category_titles):
            if not i == 0:
                concatenated_data = np.concatenate((concatenated_data, self.matrix_data[i]), axis=1)
        print(concatenated_data.shape)
        return concatenated_data

    def norm_data(self):
        for attr_index, attr_key in enumerate(self.attribute_keys):
            mean = self.uncategorized_normal_dist.means[attr_index]
            for data_item in self.normalized_concatenated_matrix_data[attr_index, :]:
                data_item -= mean

    def calc_prediction(self, args):
        prediction = []
        for c, category in enumerate(self.category_titles):
            prediction.append(self.composite_model[c].prediction(args))
        prediction = utils.normalize_list(prediction, 1)
        return prediction

    def calc_alpha_prediction(self, args):
        prediction = []
        for c, category in enumerate(self.category_titles):
            prediction.append(self.alpha_model[c].prediction(args))
        prediction = utils.normalize_list(prediction, 1)
        return prediction

    def calc_avg_central_separation_prediction(self, arg):
        prediction = []
        for c, category in enumerate(self.category_titles):
            prediction.append(self.avg_central_separation_model[c].prediction(arg))
        prediction = utils.normalize_list(prediction, 1)
        return prediction

    def single_plot(self, attr):
        if type(attr) == int:
            attr_index = attr
            attr_key = self.attribute_keys[attr_index]
        else:
            attr_key = attr
            attr_index = self.attribute_keys.index(attr_key)

        attr_min_val = self.concatenated_matrix_data[attr_index, :].min()
        attr_max_val = self.concatenated_matrix_data[attr_index, :].max()

        line = np.linspace(attr_min_val, attr_max_val, 1000)

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(1, 1, figure=fig)
        ax_attr = fig.add_subplot(gs[0, 0])

        for c, category in enumerate(self.category_titles):
            mean = self.composite_model[c].means[attr_index]
            var = self.composite_model[c].variances[attr_index]
            ax_attr.plot(
                line,
                utils.normal_dist(line, mean, var),
                self.colours[c],
                label='{} ($\mu = ${:.2f}, $\sigma^2 = ${:.2f})'.format(category, mean, var)
            )

        ax_attr.set_title('{} fitted density'.format(attr_key))
        ax_attr.set_xlabel('{} {}'.format(attr_key, self.attribute_units[attr_index]))
        ax_attr.legend()

        plt.show()

    def dual_plot(self, attribute_1, attribute_2):

        if type(attribute_1) == int:
            attr_1_index = attribute_1
            attr_1_key = self.attribute_keys[attr_1_index]
        else:
            attr_1_key = attribute_1
            attr_1_index = self.attribute_keys.index(attr_1_key)
        if type(attribute_2) == int:
            attr_2_index = attribute_2
            attr_2_key = self.attribute_keys[attr_2_index]
        else:
            attr_2_key = attribute_2
            attr_2_index = self.attribute_keys.index(attr_2_key)

        attr_1_min_val, attr_2_min_val = self.concatenated_matrix_data[attr_1_index, :].min(), self.concatenated_matrix_data[attr_2_index, :].min()
        attr_1_max_val, attr_2_max_val = self.concatenated_matrix_data[attr_1_index, :].max(), self.concatenated_matrix_data[attr_2_index, :].max()

        line_1 = np.linspace(attr_1_min_val, attr_1_max_val, 1000)
        line_2 = np.linspace(attr_2_min_val, attr_2_max_val, 1000)

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig)
        ax_attr_1 = fig.add_subplot(gs[0, 0])
        ax_attr_2 = fig.add_subplot(gs[1, 0])
        ax_scatter = fig.add_subplot(gs[:, 1])

        for c, category in enumerate(self.category_titles):
            mean = self.composite_model[c].means[attr_1_index]
            var = self.composite_model[c].variances[attr_1_index]
            ax_attr_1.plot(
                line_1,
                utils.normal_dist(line_1, mean, var),
                self.colours[c],
                label='{} ($\mu$ = {:.2f}, $\sigma^2$ = {:.2f})'.format(category, mean, var)
            )

        ax_attr_1.set_title('{} fitted density'.format(attr_1_key))
        ax_attr_1.set_xlabel('{} {}'.format(attr_1_key, self.attribute_units[attr_1_index]))
        ax_attr_1.legend()

        for c, category in enumerate(self.category_titles):
            mean = self.composite_model[c].means[attr_2_index]
            var = self.composite_model[c].variances[attr_2_index]
            ax_attr_2.plot(
                line_2,
                utils.normal_dist(line_2, mean, var),
                c=self.colours[c],
                label='{} ($\mu$ = {:.2f}, $\sigma^2$ = {:.2f})'.format(category, mean, var)
            )

        ax_attr_2.set_title('{} fitted density'.format(attr_2_key))
        ax_attr_2.set_xlabel('{} {}'.format(attr_2_key, self.attribute_units[attr_2_index]))
        ax_attr_2.legend()

        for c, category in enumerate(self.category_titles):
            ax_scatter.scatter(
                self.matrix_data[c][attr_1_index, :],
                self.matrix_data[c][attr_2_index, :],
                c=self.colours[c],
                label='{}'.format(category),
                s=8
            )

        ax_scatter.set_title('Scatter-plot of {} vs {}'.format(attr_1_key, attr_2_key))
        ax_scatter.set_xlabel('{} {}'.format(attr_1_key, self.attribute_units[attr_1_index]))
        ax_scatter.set_ylabel('{} {}'.format(attr_2_key, self.attribute_units[attr_2_index]))
        ax_scatter.legend()

        plt.show()

    def model_plot(self):

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(3, 3, figure=fig)
        ax = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[1, 2]),
            fig.add_subplot(gs[2, 0]),
            fig.add_subplot(gs[2, 1]),
            fig.add_subplot(gs[2, 2])
        ]
        ax = ax[0:len(self.attribute_keys)]

        for attr_index, attr_key in enumerate(self.attribute_keys):

            min_val = self.concatenated_matrix_data[attr_index, :].min()
            max_val = self.concatenated_matrix_data[attr_index, :].max()
            line = np.linspace(min_val, max_val, 1000)

            for c, category in enumerate(self.category_titles):
                mean = self.composite_model[c].means[attr_index]
                var = self.composite_model[c].variances[attr_index]
                ax[attr_index].plot(
                    line,
                    utils.normal_dist(line, mean, var),
                    c=self.colours[c],
                    label='{} ($\mu = ${:.2f}, $\sigma = ${:.2f})'.format(category, mean, var)
                )

            ax[attr_index].set_title('{} fitted density'.format(attr_key))
            ax[attr_index].set_xlabel('{} {}'.format(attr_key, self.attribute_units[attr_index]))
            ax[attr_index].legend()

        fig.suptitle('Composite model component normal distributions')

        plt.show()

    def plot_pca(self):
        pass

    def save(self):
        with open(self.save_filename, 'wb') as f:
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




