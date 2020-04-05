"""This module contains the different statistical models used in the column characterization algorithm. It also has
methods that can gather model parameters from sets of overlay files. The default model parameters used here are
gathered from the images in the validation data set. See automal.org for more details."""


import core
import utils
import default_models

import numpy as np


def alpha_model(alpha_angles, return_advanced_model=False):
    alpha_max = max(alpha_angles)
    alpha_min = min(alpha_angles)
    x = [alpha_max, alpha_min]
    params = default_models.alpha_model
    advanced_model = [
        utils.multivariate_normal_dist(x, params[0][0], params[0][2], params[0][3]),
        utils.multivariate_normal_dist(x, params[1][0], params[1][2], params[1][3]),
        utils.multivariate_normal_dist(x, params[2][0], params[2][2], params[2][3]),
        utils.multivariate_normal_dist(x, params[3][0], params[3][2], params[3][3]),
        utils.multivariate_normal_dist(x, params[4][0], params[4][2], params[4][3]),
        utils.multivariate_normal_dist(x, params[5][0], params[5][2], params[5][3]),
        utils.multivariate_normal_dist(x, params[6][0], params[6][2], params[6][3]),
        utils.multivariate_normal_dist(x, params[7][0], params[7][2], params[7][3])
    ]
    simple_model = [
        advanced_model[0] + advanced_model[1],
        advanced_model[2],
        0,
        advanced_model[3] + advanced_model[4],
        0,
        advanced_model[5] + advanced_model[6] + advanced_model[7],
        0
    ]
    if return_advanced_model:
        return utils.normalize_list(advanced_model, 1)
    else:
        return utils.normalize_list(simple_model, 1)


def theta_model(Theta_angles):
    pass


def normalized_gamma_model(normalized_peak_gamma, normalized_avg_gamma):
    pass


def composite_model(vertex):
    pass


def product_model(vertex):
    pass


def weighted_model(vertex):
    pass


def get_model_parameters(model, files=None):
    """Get the parameters for a particular model.

    ======================= ==============================
    model                   Return parameters of
    ======================= ==============================
    0                       Alpha-model
    1                       Theta-model
    2                       Normalized gamma-model
    3                       Composite model
    4                       Product model
    5                       Weighted model
    ======================= ==============================

    """

    if model == 0:
        if files is None:
            pass
        else:
            pass

    elif model == 1:
        pass

    elif model == 2:
        pass

    elif model == 3:
        pass

    elif model == 4:
        pass

    elif model == 5:
        pass

    elif model == 6:
        pass

    else:
        return None


def calculate_parameters_from_files(files, model, filter=None, recalc_properties=False, savefile=None):
    """Calculate the parameters for a particular model from a list of provided files. To use this method effectively,
    keep in mind that all columns in the provided files should be correctly labeled, including flag-states to separate
    between the different categories! (If an Si column is encountered, if flag_5 is True, it is counted as an Si_1
    column, while it will be counted as an Si_2 if flag_5 is False!)

    ======================= ==============================
    model                   Return parameters of
    ======================= ==============================
    0                       Alpha-model
    1                       Theta-model
    2                       Normalized gamma-model
    3                       Composite model
    4                       Product model
    5                       Weighted model
    ======================= ==============================

    """
    if filter is None:
        filter = [False, True, True, True, True, True, True]
    data = [[[]], [[]], [[]], [[]], [[]], [[]], [[]], [[]]]
    for i, file in enumerate(files):
        instance = core.SuchSoftware.load(file)
        image_data = instance.graph.calc_condensed_property_data(filter, recalc_properties, evaluate_category=True)
        for advanced_species in range(0, 8):
            for property_ in range(0, 7):
                data[advanced_species][property_] += image_data[advanced_species][property_]

    params = [[], [], [], [], [], [], [], []]

    if model == 0:
        for advanced_species_index, species_data in enumerate(data):
            covar_matrix = [[0, 0], [0, 0]]
            mean_vector = []
            for property_ in range(0, 2):
                mean_vector.append(utils.mean_val(data[advanced_species_index][property_]))
                for secondary_property in range(0, 2):
                    if property_ == secondary_property:
                        covar_matrix[property_][secondary_property] = utils.variance(data[advanced_species_index][property_])
                    else:
                        covar_matrix[property_][secondary_property] = utils.covariance(data[advanced_species_index][property_], data[advanced_species_index][secondary_property])
            params[advanced_species_index].append(mean_vector)
            params[advanced_species_index].append(covar_matrix)
            params[advanced_species_index].append(np.linalg.det(np.array(covar_matrix)))
            params[advanced_species_index].append(np.linalg.inv(np.array(covar_matrix)).tolist())
        if savefile is None:
            default_models.alpha_model = params
        else:
            default_models.alpha_model = params

    elif model == 1:
        for advanced_species_index, species_data in enumerate(data):
            covar_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            mean_vector = []
            for property_ in range(2, 5):
                mean_vector.append(utils.mean_val(data[advanced_species_index][property_]))
                for secondary_property in range(2, 5):
                    if property_ == secondary_property:
                        covar_matrix[property_][secondary_property] = utils.variance(
                            data[advanced_species_index][property_])
                    else:
                        covar_matrix[property_][secondary_property] = utils.covariance(
                            data[advanced_species_index][property_], data[advanced_species_index][secondary_property])
            params[advanced_species_index].append(mean_vector)
            params[advanced_species_index].append(covar_matrix)
            params[advanced_species_index].append(np.linalg.det(np.array(covar_matrix)))
            params[advanced_species_index].append(np.linalg.inv(np.array(covar_matrix)).tolist())
        if savefile is None:
            default_models.theta_model = params
        else:
            default_models.theta_model = params

    elif model == 2:
        for advanced_species_index, species_data in enumerate(data):
            covar_matrix = [[0, 0], [0, 0]]
            mean_vector = []
            for property_ in range(5, 7):
                mean_vector.append(utils.mean_val(data[advanced_species_index][property_]))
                for secondary_property in range(5, 7):
                    if property_ == secondary_property:
                        covar_matrix[property_][secondary_property] = utils.variance(
                            data[advanced_species_index][property_])
                    else:
                        covar_matrix[property_][secondary_property] = utils.covariance(
                            data[advanced_species_index][property_], data[advanced_species_index][secondary_property])
            params[advanced_species_index].append(mean_vector)
            params[advanced_species_index].append(covar_matrix)
            params[advanced_species_index].append(np.linalg.det(np.array(covar_matrix)))
            params[advanced_species_index].append(np.linalg.inv(np.array(covar_matrix)).tolist())
        if savefile is None:
            default_models.gamma = params
        else:
            print(params)

    elif model == 3:
        for advanced_species_index, species_data in enumerate(data):
            covar_matrix = [[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]]
            mean_vector = []
            for property_ in range(0, 7):
                mean_vector.append(utils.mean_val(data[advanced_species_index][property_]))
                for secondary_property in range(0, 7):
                    if property_ == secondary_property:
                        covar_matrix[property_][secondary_property] = utils.variance(
                            data[advanced_species_index][property_])
                    else:
                        covar_matrix[property_][secondary_property] = utils.covariance(
                            data[advanced_species_index][property_], data[advanced_species_index][secondary_property])
            params[advanced_species_index].append(mean_vector)
            params[advanced_species_index].append(covar_matrix)
            params[advanced_species_index].append(np.linalg.det(np.array(covar_matrix)))
            params[advanced_species_index].append(np.linalg.inv(np.array(covar_matrix)).tolist())
        if savefile is None:
            default_models.composite_model = params
        else:
            print(params)

    elif model == 4:
        pass

    elif model == 5:
        pass

    elif model == 6:
        pass

    else:
        pass


