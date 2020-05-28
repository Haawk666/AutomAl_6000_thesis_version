"""This module contains the different statistical models used in the column characterization algorithm. It also has
methods that can gather model parameters from sets of overlay files. See automal.org for more details."""

# Internal imports
import core
import utils
# External imports
import csv
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pickle
import copy
import logging
# Instantiate logger:
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MultivariateNormalDist:

    def __init__(self, data, category_title, attribute_keys):

        # k is the dimensionality of the data, n is the number of data-points
        (self.k, self.n) = data.shape
        # Title of category is mandatory
        self.category_title = category_title
        self.attribute_keys = attribute_keys

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
        self.covar_matrix_eigenvalues = None
        self.covar_matrix_eigenvectors = None

        if self.n > 1:
            self.calc_params(data)
        else:
            self.means = [0] * self.k
            self.variances = [1] * self.k

    def calc_params(self, data):
        """Two pass algorithm"""
        self.means = []
        self.variances = []
        for dimension in range(0, self.k):
            mean = utils.mean_val(data[dimension, :])
            var = utils.variance(data[dimension, :])
            self.means.append(mean)
            self.variances.append(var)

        for dimension_1 in range(0, self.k):
            for dimension_2 in range(0, self.k):
                covar = 0
                for data_index in range(0, self.n):
                    factor_1 = data[dimension_1, data_index] - self.means[dimension_1]
                    factor_2 = data[dimension_2, data_index] - self.means[dimension_2]
                    covar += factor_1 * factor_2
                covar /= self.n - 1
                if dimension_1 == dimension_2 and covar < 0.0000000001:
                    covar = 0.0001
                self.covar_matrix[dimension_1, dimension_2] = covar

        self.covar_matrix_determinant = np.linalg.det(self.covar_matrix)
        if not self.covar_matrix_determinant == 0:
            self.inverse_covar_matrix = np.linalg.inv(self.covar_matrix)
        else:
            self.inverse_covar_matrix = copy.deepcopy(self.covar_matrix)
        self.covar_matrix_eigenvalues, self.covar_matrix_eigenvectors = np.linalg.eig(self.covar_matrix)
        idx = np.argsort(self.covar_matrix_eigenvalues)[::-1]
        self.covar_matrix_eigenvalues = self.covar_matrix_eigenvalues[idx]
        self.covar_matrix_eigenvectors = self.covar_matrix_eigenvectors[:, idx]

    def get_partial_model(self, attr_indices):
        new_means = []
        for attr_index in attr_indices:
            new_means.append(self.means[attr_index])
        new_covar_matrix = np.array(self.covar_matrix[np.ix_(attr_indices, attr_indices)])
        new_inverse_covar_matrix = np.linalg.inv(new_covar_matrix)
        new_covar_matrix_determinant = np.linalg.det(new_covar_matrix)
        return new_means, new_covar_matrix_determinant, new_inverse_covar_matrix

    def prediction(self, dict_):
        args = []
        kwargs = []
        attr_indices = []
        for key in self.attribute_keys:
            if key in dict_:
                args.append(dict_[key])
                kwargs.append(key)
                attr_indices.append(self.attribute_keys.index(key))
        if len(kwargs) == len(self.attribute_keys):
            prob = utils.multivariate_normal_dist(args, self.means, self.covar_matrix_determinant, self.inverse_covar_matrix)
        else:
            new_means, new_covar_matrix_determinant, new_inverse_covar_matrix = self.get_partial_model(attr_indices)
            prob = utils.multivariate_normal_dist(args, new_means, new_covar_matrix_determinant, new_inverse_covar_matrix)
        return prob


class VertexDataManager:
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



    note
    ----------
    This dataManager is designed to handle vertex-based data. To analyse Arc-centered data, use the 'ArcDataManager' class


    :param files: String of full filenames separated by newline character. Data will be gathered from each of the files.
    :param filter_: (Optional, default: None) Filter inclusion of data by the tokens in the above table.
    :param attr_keys: (Optional, default: None) The keys of the data attributes to include
    :param save_filename: (Optional, default: 'model') Pickle the model data with this filename
    :param categorization: (Optional, default: 'advanced') Categorization keyword

    :type files: string
    :type filter_: [bool]
    :type attr_keys: [string]
    :type save_filename: string
    :type categorization: string

    :returns DataManager object:
    :rtype :code:`<statistics.DataManager>`

    """

    default_dict = {
        'Si_1': {
            'symmetry': 3,
            'atomic_species': 'Si',
            'atomic_radii': 117.50,
            'color': (255, 0, 0),
            'species_color': (255, 0, 0),
            'description': 'Q-prime Si'
        },
        'Si_2': {
            'symmetry': 3,
            'atomic_species': 'Si',
            'atomic_radii': 117.50,
            'color': (235, 20, 20),
            'species_color': (255, 0, 0),
            'description': 'Beta-pprime Si'
        },
        'Cu_1': {
            'symmetry': 3,
            'atomic_species': 'Cu',
            'atomic_radii': 127.81,
            'color': (255, 255, 0),
            'species_color': (255, 255, 0),
            'description': ''
        },
        'Al_1': {
            'symmetry': 4,
            'atomic_species': 'Al',
            'atomic_radii': 143.00,
            'color': (0, 255, 0),
            'species_color': (0, 255, 0),
            'description': ''
        },
        'Al_2': {
            'symmetry': 4,
            'atomic_species': 'Al',
            'atomic_radii': 143.00,
            'color': (20, 235, 20),
            'species_color': (0, 255, 0),
            'description': ''
        },
        'Mg_1': {
            'symmetry': 5,
            'atomic_species': 'Mg',
            'atomic_radii': 160.00,
            'color': (138, 43, 226),
            'species_color': (138, 43, 226),
            'description': ''
        },
        'Mg_2': {
            'symmetry': 5,
            'atomic_species': 'Mg',
            'atomic_radii': 160.00,
            'color': (118, 63, 206),
            'species_color': (138, 43, 226),
            'description': ''
        },
        'Un_1': {
            'symmetry': 3,
            'atomic_species': 'Un',
            'atomic_radii': 100.00,
            'color': (0, 0, 255),
            'species_color': (0, 0, 255),
            'description': ''
        }
    }

    def __init__(self, files, filter_=None, attr_keys=None, category_key='advanced_species', save_filename='model', recalc=False, species_dict=None):

        if filter_ is None:
            self.filter_ = {
                'exclude_edge_columns': True,
                'exclude_matrix_columns': False,
                'exclude_particle_columns': False,
                'exclude_hidden_columns': False,
                'exclude_flag_1_columns': False,
                'exclude_flag_2_columns': False,
                'exclude_flag_3_columns': False,
                'exclude_flag_4_columns': False
            }
        else:
            self.filter_ = filter_

        if attr_keys is None:
            self.keys = ['alpha_max', 'alpha_min', 'theta_max', 'theta_min',
                         'theta_angle_mean', 'normalized_peak_gamma', 'normalized_avg_gamma']
        else:
            self.keys = attr_keys
        self.attribute_keys, self.attribute_units = self.determine_attribute_keys()
        self.k = len(self.attribute_keys)
        self.pc_keys = []
        for i in range(0, self.k):
            self.pc_keys.append('PC {}'.format(i))

        if species_dict is None:
            self.species_dict = self.default_dict
        else:
            self.species_dict = species_dict

        self.files = files
        self.save_filename = save_filename
        self.recalc = recalc

        self.category_key = category_key

        self.original_dict_data, self.category_list = self.collect_data()
        self.n = len(self.original_dict_data)

        self.num_data_categories = len(self.category_list)

        self.matrix_data = self.vectorize_data()
        self.composite_model = []
        for c, category_data in enumerate(self.matrix_data):
            self.composite_model.append(MultivariateNormalDist(category_data, self.category_list[c], self.attribute_keys))

        self.concatenated_matrix_data = self.concatenate_categories()
        self.uncategorized_normal_dist = MultivariateNormalDist(self.concatenated_matrix_data, 'All categories', self.attribute_keys)

        self.normalized_concatenated_matrix_data = np.array(self.concatenated_matrix_data)
        self.normalized_matrix_data = []
        for category_data in self.matrix_data:
            self.normalized_matrix_data.append(np.array(category_data))
        self.norm_data()
        self.normalized_uncategorized_normal_dist = MultivariateNormalDist(self.normalized_concatenated_matrix_data, 'All categories', self.attribute_keys)
        self.pca_feature_vector = self.normalized_uncategorized_normal_dist.covar_matrix_eigenvectors.T
        self.composed_uncategorized_data = np.matmul(self.pca_feature_vector, self.normalized_concatenated_matrix_data)
        self.composed_data = []
        for normalized_category_data in self.normalized_matrix_data:
            self.composed_data.append(np.matmul(self.pca_feature_vector, normalized_category_data))
        self.composed_uncategorized_normal_dist = MultivariateNormalDist(self.composed_uncategorized_data, 'Column', self.pc_keys)
        self.composed_normal_dist = []
        for c, composed_category_data in enumerate(self.composed_data):
            self.composed_normal_dist.append(MultivariateNormalDist(composed_category_data, self.category_list[c], self.pc_keys))

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
        if 'theta_angle_variance' in self.keys:
            attributes.append('theta_angle_variance')
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
        if 'redshift' in self.keys:
            attributes.append('redshift')
            units.append('nm')
        if 'avg_redshift' in self.keys:
            attributes.append('avg_redshift')
            units.append('nm')
        return attributes, units

    def collect_data(self):
        data = []
        categories = set()
        for file in self.files.splitlines(keepends=False):
            instance = core.Project.load(file)
            all_keys = copy.deepcopy(self.attribute_keys)
            all_keys.append(self.category_key)
            image_data = instance.graph.calc_condensed_property_data(filter_=self.filter_, recalc=self.recalc, keys=all_keys)
            data += image_data
        for dataline in data:
            categories.add(dataline[self.category_key])
        categories = list(categories)
        categories.sort()
        return data, categories

    def vectorize_data(self):
        data = []
        for category in range(0, self.num_data_categories):
            data.append([])
            for attribute in range(0, self.k):
                data[category].append([])
        for data_item in self.original_dict_data:
            for h, attribute in enumerate(self.attribute_keys):
                data[self.category_list.index(data_item[self.category_key])][h].append(data_item[attribute])
        matrix_data = []
        for category_data in data:
            matrix_data.append(np.array(category_data))
        return matrix_data

    def concatenate_categories(self):
        concatenated_data = self.matrix_data[0]
        for i, category in enumerate(self.category_list):
            if not i == 0:
                concatenated_data = np.concatenate((concatenated_data, self.matrix_data[i]), axis=1)
        return concatenated_data

    def norm_data(self):
        # Normalize uncategorized data
        for attr_index, attr_key in enumerate(self.attribute_keys):
            mean = self.uncategorized_normal_dist.means[attr_index]
            for data_item in self.normalized_concatenated_matrix_data[attr_index, :]:
                data_item -= mean
        # Normalize categorized data
        for c, category in enumerate(self.category_list):
            for attr_index, attr_key, in enumerate(self.attribute_keys):
                mean = self.composite_model[c].means[attr_index]
                for data_item in self.normalized_matrix_data[c][attr_index, :]:
                    data_item -= mean

    def calc_prediction(self, dict_):
        prediction = {}
        for c, category in enumerate(self.category_list):
            prediction[category] = self.composite_model[c].prediction(dict_)
        prediction = utils.normalize_dict(prediction, 1)
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

        for c, category in enumerate(self.category_list):
            mean = self.composite_model[c].means[attr_index]
            var = self.composite_model[c].variances[attr_index]
            ax_attr.plot(
                line,
                utils.normal_dist(line, mean, var),
                c=np.array(utils.norm_rgb_tuple(self.species_dict[category]['color'])),
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

        for c, category in enumerate(self.category_list):
            mean = self.composite_model[c].means[attr_1_index]
            var = self.composite_model[c].variances[attr_1_index]
            ax_attr_1.plot(
                line_1,
                utils.normal_dist(line_1, mean, var),
                c=np.array(utils.norm_rgb_tuple(self.species_dict[category]['color'])),
                label='{} ($\mu$ = {:.2f}, $\sigma^2$ = {:.2f})'.format(category, mean, var)
            )

        ax_attr_1.set_title('{} fitted density'.format(attr_1_key))
        ax_attr_1.set_xlabel('{} {}'.format(attr_1_key, self.attribute_units[attr_1_index]))
        ax_attr_1.legend()

        for c, category in enumerate(self.category_list):
            mean = self.composite_model[c].means[attr_2_index]
            var = self.composite_model[c].variances[attr_2_index]
            ax_attr_2.plot(
                line_2,
                utils.normal_dist(line_2, mean, var),
                c=np.array(utils.norm_rgb_tuple(self.species_dict[category]['color'])),
                label='{} ($\mu$ = {:.2f}, $\sigma^2$ = {:.2f})'.format(category, mean, var)
            )

        ax_attr_2.set_title('{} fitted density'.format(attr_2_key))
        ax_attr_2.set_xlabel('{} {}'.format(attr_2_key, self.attribute_units[attr_2_index]))
        ax_attr_2.legend()

        for c, category in enumerate(self.category_list):
            ax_scatter.scatter(
                self.matrix_data[c][attr_1_index, :],
                self.matrix_data[c][attr_2_index, :],
                c=np.array(utils.norm_rgb_tuple(self.species_dict[category]['color'])),
                label='{}'.format(category),
                s=8
            )

        ax_scatter.set_title('Scatter-plot of {} vs {}'.format(attr_1_key, attr_2_key))
        ax_scatter.set_xlabel('{} {}'.format(attr_1_key, self.attribute_units[attr_1_index]))
        ax_scatter.set_ylabel('{} {}'.format(attr_2_key, self.attribute_units[attr_2_index]))
        ax_scatter.legend()

        plt.show()

    def plot_all(self):
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

            for c, category in enumerate(self.category_list):
                mean = self.composite_model[c].means[attr_index]
                var = self.composite_model[c].variances[attr_index]
                ax[attr_index].plot(
                    line,
                    utils.normal_dist(line, mean, var),
                    c=np.array(utils.norm_rgb_tuple(self.species_dict[category]['color'])),
                    label='{} ($\mu = ${:.2f}, $\sigma = ${:.2f})'.format(category, mean, var)
                )

            ax[attr_index].set_title('{} fitted density'.format(attr_key))
            ax[attr_index].set_xlabel('{} {}'.format(attr_key, self.attribute_units[attr_index]))
            ax[attr_index].legend()

        fig.suptitle('Composite model component normal distributions')

        plt.show()

    def z_plot(self):
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

            for c, category in enumerate(self.category_list):
                mean = self.composite_model[c].means[attr_index]
                var = self.composite_model[c].variances[attr_index]
                ax[attr_index].scatter(
                    self.matrix_data[c][attr_index, :],
                    utils.z_score(self.matrix_data[c][attr_index, :], mean, var),
                    c=np.array(utils.norm_rgb_tuple(self.species_dict[category]['color'])),
                    label='{}'.format(category),
                    s=1
                )

            ax[attr_index].set_title('{} z-scores'.format(attr_key))
            ax[attr_index].set_xlabel('{}'.format(attr_key))
            ax[attr_index].legend()

        fig.suptitle('Composite model attribute z-scores')

        plt.show()

    def plot_pca(self, show_category=True):
        attr_1_key = 'PC 1'
        attr_1_index = 0
        attr_2_key = 'PC 2'
        attr_2_index = 1

        attr_1_min_val, attr_2_min_val = self.composed_uncategorized_data[attr_1_index, :].min(), self.composed_uncategorized_data[attr_2_index, :].min()
        attr_1_max_val, attr_2_max_val = self.composed_uncategorized_data[attr_1_index, :].max(), self.composed_uncategorized_data[attr_2_index, :].max()

        line_1 = np.linspace(attr_1_min_val, attr_1_max_val, 1000)
        line_2 = np.linspace(attr_2_min_val, attr_2_max_val, 1000)

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig)
        ax_attr_1 = fig.add_subplot(gs[0, 0])
        ax_attr_2 = fig.add_subplot(gs[1, 0])
        ax_scatter = fig.add_subplot(gs[:, 1])

        if show_category:

            for c, category in enumerate(self.category_list):
                mean = self.composed_normal_dist[c].means[attr_1_index]
                var = self.composed_normal_dist[c].variances[attr_1_index]
                ax_attr_1.plot(
                    line_1,
                    utils.normal_dist(line_1, mean, var),
                    c=np.array(list(utils.norm_rgb_tuple(self.species_dict[category]['color']))),
                    label='{} ($\mu$ = {:.2f}, $\sigma^2$ = {:.2f})'.format(category, mean, var)
                )

            ax_attr_1.set_title('{} fitted density'.format(attr_1_key))
            ax_attr_1.set_xlabel('{}'.format(attr_1_key))
            ax_attr_1.legend()

            for c, category in enumerate(self.category_list):
                mean = self.composed_normal_dist[c].means[attr_2_index]
                var = self.composed_normal_dist[c].variances[attr_2_index]
                ax_attr_2.plot(
                    line_2,
                    utils.normal_dist(line_2, mean, var),
                    c=np.array(list(utils.norm_rgb_tuple(self.species_dict[category]['color']))),
                    label='{} ($\mu$ = {:.2f}, $\sigma^2$ = {:.2f})'.format(category, mean, var)
                )

            ax_attr_2.set_title('{} fitted density'.format(attr_2_key))
            ax_attr_2.set_xlabel('{}'.format(attr_2_key))
            ax_attr_2.legend()

            for c, category in enumerate(self.category_list):
                ax_scatter.scatter(
                    self.composed_data[c][attr_1_index, :],
                    self.composed_data[c][attr_2_index, :],
                    c=np.array(list(utils.norm_rgb_tuple(self.species_dict[category]['color']))),
                    label='{}'.format(category),
                    s=8
                )

            ax_scatter.set_title('Scatter-plot of {} vs {}'.format(attr_1_key, attr_2_key))
            ax_scatter.set_xlabel('{}'.format(attr_1_key))
            ax_scatter.set_ylabel('{}'.format(attr_2_key))
            ax_scatter.legend()

        else:

            mean = self.composed_uncategorized_normal_dist.means[attr_1_index]
            var = self.composed_uncategorized_normal_dist.variances[attr_1_index]
            ax_attr_1.plot(
                line_1,
                utils.normal_dist(line_1, mean, var),
                c='k',
                label='{} ($\mu$ = {:.2f}, $\sigma^2$ = {:.2f})'.format('Column', mean, var)
            )

            ax_attr_1.set_title('{} fitted density'.format(attr_1_key))
            ax_attr_1.set_xlabel('{} {}'.format(attr_1_key, self.attribute_units[attr_1_index]))
            ax_attr_1.legend()

            mean = self.composed_uncategorized_normal_dist.means[attr_2_index]
            var = self.composed_uncategorized_normal_dist.variances[attr_2_index]
            ax_attr_2.plot(
                line_2,
                utils.normal_dist(line_2, mean, var),
                c='k',
                label='{} ($\mu$ = {:.2f}, $\sigma^2$ = {:.2f})'.format('Column', mean, var)
            )

            ax_attr_2.set_title('{} fitted density'.format(attr_2_key))
            ax_attr_2.set_xlabel('{}'.format(attr_2_key))
            ax_attr_2.legend()

            ax_scatter.scatter(
                self.composed_uncategorized_data[attr_1_index, :],
                self.composed_uncategorized_data[attr_2_index, :],
                c='k',
                label='{}'.format('Column'),
                s=8
            )

            ax_scatter.set_title('Scatter-plot of {} vs {}'.format(attr_1_key, attr_2_key))
            ax_scatter.set_xlabel('{}'.format(attr_1_key))
            ax_scatter.set_ylabel('{}'.format(attr_2_key))
            ax_scatter.legend()

        plt.show()

    def plot_all_pc(self):

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

        for attr_index, attr_key in enumerate(self.pc_keys):

            min_val = self.composed_uncategorized_data[attr_index, :].min()
            max_val = self.composed_uncategorized_data[attr_index, :].max()
            line = np.linspace(min_val, max_val, 1000)

            for c, category in enumerate(self.category_list):
                mean = self.composed_normal_dist[c].means[attr_index]
                var = self.composed_normal_dist[c].variances[attr_index]
                ax[attr_index].plot(
                    line,
                    utils.normal_dist(line, mean, var),
                    c=np.array(utils.norm_rgb_tuple(self.species_dict[category]['color'])),
                    label='{} ($\mu = ${:.2f}, $\sigma = ${:.2f})'.format(category, mean, var)
                )

            ax[attr_index].set_title('{} fitted density'.format(attr_key))
            ax[attr_index].set_xlabel('{} {}'.format(attr_key, self.attribute_units[attr_index]))
            ax[attr_index].legend()

        fig.suptitle('Principle components')

        plt.show()

    def export_csv(self, filename, kwargs):
        with open(filename, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, kwargs, delimiter=',', extrasaction='ignore')
            writer.writeheader()
            for dict_ in self.original_dict_data:
                writer.writerow(dict_)

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


class ArcDataManager:

    def __inti__(self):

        pass


class ImageDataManager:

    def __inti__(self):

        pass

