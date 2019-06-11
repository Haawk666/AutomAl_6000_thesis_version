# Utilities script
# -------------------------------------------
# This script contains a collection of static functions that are abstracted away from core.py

import numpy as np
import scipy.optimize as opt
from copy import deepcopy


def gen_gauss(amp, x_0, y_0, x_size, y_size, a, b, c, offset):
    mat = np.zeros((y_size, x_size))
    for x in range(0, x_size - 1):
        for y in range(0, y_size - 1):
            xy = np.array([[x], [y]])
            mat[y, x] = gauss_func(xy, amp, x_0, y_0, a, b, c, offset)
    return mat


def gauss_func(xy, amp, x_0, y_0, a, b, c, offset):
    x, y = xy
    inner = a * (x - x_0)**2
    inner = inner + 2 * b * (x - x_0) * (y - y_0)
    inner = inner + c * (y - y_0)**2
    return amp * np.exp(-inner) + offset


def y_line(slope, b, x):

    return slope * x + b


def normal_dist(x, mean, std):
    dist = (x - mean)**2
    dist = dist / (2 * std**2)
    dist = -dist
    dist = np.exp(dist)
    dist = (1 / np.sqrt(2 * np.pi * std**2)) * dist
    return dist


def find_angle(a_1, a_2, b_1, b_2):
    # Returns the smallest angle between two vectors (a_1, b_1) and (a_2, b_2)
    alpha = a_1 * a_2 + b_1 * b_2
    alpha = alpha / np.sqrt((a_1**2 + b_1**2) * (a_2**2 + b_2**2))
    alpha = np.arccos(alpha)

    return alpha


def find_angle_from_points(p1, p2, pivot):

    vec_1 = (p1[0] - pivot[0], p1[1] - pivot[1])
    vec_2 = (p2[0] - pivot[0], p2[1] - pivot[1])
    alpha = find_angle(vec_1[0], vec_2[0], vec_1[1], vec_2[1])

    if vector_cross_product_magnitude(vec_1[0], vec_2[0], vec_1[1], vec_2[1]) < 0:
        alpha = 2 * np.pi - alpha

    return alpha


def vector_magnitude(vec):

    length = vec[0] ** 2 + vec[1] ** 2
    length = np.sqrt(length)

    return length


def vector_cross_product_magnitude(a_1, a_2, b_1, b_2):

    return a_1 * b_2 - b_1 * a_2


def dual_sort(list_1, list_2):

    temp_list_1 = np.ndarray([list_1.shape[0]], dtype=type(list_1))
    temp_list_2 = np.ndarray([list_1.shape[0]], dtype=type(list_2))

    for x in range(0, list_1.shape[0]):

        index = list_1.argmin()
        temp_list_1[x] = deepcopy(list_1[index])
        temp_list_2[x] = deepcopy(list_2[index])
        list_1[index] = list_1.max() + 1

    return temp_list_1, temp_list_2


def sort_neighbours(indices, distances, n):
    temp_max = distances.max()
    temp_indices = np.ndarray([n], dtype=np.int)
    temp_distances = np.ndarray([n], dtype=np.float64)

    for j in range(0, n):
        k = distances.argmin()
        temp_distances[j] = distances[k]
        temp_indices[j] = indices[k]
        distances[k] = temp_max + j + 1

    return temp_indices, temp_distances


def gaussian_fit(mat, x_0, y_0, r):
    # outdated and not in use
    size_of_section = (2 * r + 1)**2
    size_of_circular_section = 529
    temp_mat = np.zeros((2 * r + 1, 2 * r + 1), dtype=type(mat))
    xy = np.zeros((2, size_of_circular_section), dtype=np.int)
    zobs = np.zeros(size_of_circular_section, dtype=np.float64)
    sigma = zobs
    counter = 0
    for x_i in range(x_0 - r, x_0 + r + 1):
        for y_i in range(y_0 - r, y_0 + r + 1):
            distance = np.sqrt((x_0 - x_i)**2 + (y_0 - y_i)**2)
            # create "portrait" of column weighted as distance from centre to prevent "bleed" from neighbouring atoms
            temp_mat[y_i - y_0 + r, x_i - x_0 + r] = mat[y_i, x_i]
            # Prepare data-set for curve fitting that is circularly sampled around the proposed centre
            if distance <= r:
                xy[:, counter] = (x_i - x_0 + r, y_i - y_0 + r)
                zobs[counter] = temp_mat[y_i - y_0 + r, x_i - x_0 + r]
                sigma[counter] = (distance / r) * temp_mat[y_i - y_0 + r, x_i - x_0 + r] + 0.001
                counter = counter + 1
    print(counter)
    guess = [temp_mat.max(), r, r, 0.0001, 0.0001, 0.0001, 0]
    (params, uncert) = opt.curve_fit(gauss_func, xy, zobs, p0=guess, sigma=sigma, absolute_sigma=1, maxfev=6400)
    print(params)
    mat = gen_gauss(params[0], np.floor(params[1]), np.floor(params[2]), 2 * r + 1, 2 * r + 1, params[3], params[4], params[5], params[6])
    # Translate to coordinates of the real image:
    x_fit = int(np.floor(params[1]) + x_0 - r)
    y_fit = int(np.floor(params[2]) + y_0 - r)
    return temp_mat, mat, params, uncert, x_fit, y_fit


def cm_fit(mat, x_0, y_0, r):
    # "Centre of mass" fit. Calculates the centre of mass of the pixels in the circle centred at (x_0, y_0) with radius
    # r.
    portrait_mat = mat[x_0 - r:x_0 + r, y_0 - r:y_0 + r]
    counter = 0
    total_mass = 0
    weighted_x_sum = 0
    weighted_y_sum = 0
    for x_i in range(x_0 - r, x_0 + r):
        for y_i in range(y_0 - r, y_0 + r):
            distance = np.sqrt((x_0 - x_i)**2 + (y_0 - y_i)**2)
            if distance <= r:
                total_mass = total_mass + mat[y_i, x_i]
                weighted_x_sum = weighted_x_sum + mat[y_i, x_i]*x_i
                weighted_y_sum = weighted_y_sum + mat[y_i, x_i]*y_i
                counter = counter + 1
    x_fit = weighted_x_sum / total_mass
    y_fit = weighted_y_sum / total_mass
    return portrait_mat, x_fit, y_fit














