"""This module contains the different statistical models used in the column characterization algorithm. It also has
methods that can gather model parameters from sets of overlay files. The default model parameters used here are
gathered from the images in the validation data set. See automal.org for more details."""


import core


def alpha_model(alpha_angles):
    pass


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


def calculate_parameters_from_files(files, model, recalc_properties=False):
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

    if model == 0:

        data_alpha_max = [[], [], [], [], [], [], [], []]
        data_alpha_min = [[], [], [], [], [], [], [], []]

        for file in files:
            instance = core.SuchSoftware.load(file)
            if recalc_properties:
                instance.graph.refresh_graph()
            for vertex in instance.graph.vertices:
                if not vertex.void and not vertex.is_edge_column:
                    if vertex.species_index == 0:
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

