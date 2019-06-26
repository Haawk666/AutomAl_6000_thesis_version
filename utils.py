# Utilities script
# -------------------------------------------
"""General utility/convenience functions"""

import numpy as np
import scipy.optimize as opt
from copy import deepcopy


def circularize_next_index(i, i_max):
    """Make an index cyclic

    parameters
    ----------
    i : int
        Integer index to be converted to a cyclic index
    i_max : int
        The maximum valid value of i.

    returns
    ----------
    i: int
        returns cyclic index

    note:
    ----------
    Python will automatically give you list[-1] = list[len(list) - 1], so checking for i == -1 might be redundant."""

    if i == i_max + 1:
        return 0
    elif i == -1:
        return i_max
    else:
        return i


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


def normalize_list(in_list, norm_sum=1):

    factor = norm_sum / sum(in_list)
    norm_list = [a * factor for a in in_list]
    return norm_list


def normalize_array(in_array, norm_sum=1):

    factor = in_array.sum() / norm_sum
    norm_array = in_array / factor
    return norm_array


def find_angle(a_1, a_2, b_1, b_2):
    # Returns the smallest angle between two vectors (a_1, b_1) and (a_2, b_2)
    alpha = a_1 * a_2 + b_1 * b_2
    alpha = alpha / np.sqrt((a_1**2 + b_1**2) * (a_2**2 + b_2**2))
    alpha = np.arccos(alpha)

    return alpha


def mean_val(data):

    return sum(data) / len(data)


def variance(data):

    mean = mean_val(data)

    sum_ = 0
    for item in data:
        sum_ += (item - mean)**2

    sum_ = sum_ / (len(data))

    return sum_


def deviation(data):

    return np.sqrt(variance(data))


def find_angle_from_points(p1, p2, pivot):

    # Always returns the anti-clockwise angle from p1 to p2 around the pivot (radians)

    vec_1 = (p1[0] - pivot[0], p1[1] - pivot[1])
    vec_2 = (p2[0] - pivot[0], p2[1] - pivot[1])
    alpha = find_angle(vec_1[0], vec_2[0], vec_1[1], vec_2[1])

    if vector_cross_product_magnitude(vec_1[0], vec_2[0], vec_1[1], vec_2[1]) > 0:
        alpha = 2 * np.pi - alpha

    return alpha


def vector_magnitude(vec):

    length = vec[0] ** 2 + vec[1] ** 2
    length = np.sqrt(length)

    return length


def vector_cross_product_magnitude(a_1, a_2, b_1, b_2):

    return a_1 * b_2 - b_1 * a_2


def side(a, b, c):
    """ Returns a position of the point c relative to the line going through a and b
        Points a, b are expected to be different
    """
    d = (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])
    return 1 if d > 0 else (-1 if d < 0 else 0)


def is_point_in_closed_segment(a, b, c):
    """ Returns True if c is inside closed segment, False otherwise.
        a, b, c are expected to be collinear
    """
    if a[0] < b[0]:
        return a[0] <= c[0] and c[0] <= b[0]
    if b[0] < a[0]:
        return b[0] <= c[0] and c[0] <= a[0]

    if a[1] < b[1]:
        return a[1] <= c[1] and c[1] <= b[1]
    if b[1] < a[1]:
        return b[1] <= c[1] and c[1] <= a[1]

    return a[0] == c[0] and a[1] == c[1]


def closed_segment_intersect(a, b, c, d):
    """ Verifies if closed segments a, b, c, d do intersect.
    """
    if a == b or c == d or a == c or a == d or b == c or b == d:
        return False

    s1 = side(a, b, c)
    s2 = side(a, b, d)

    # All points are collinear
    if s1 == 0 and s2 == 0:
        return \
            is_point_in_closed_segment(a, b, c) or is_point_in_closed_segment(a, b, d) or \
            is_point_in_closed_segment(c, d, a) or is_point_in_closed_segment(c, d, b)

    # No touching and on the same side
    if s1 and s1 == s2:
        return False

    s1 = side(c, d, a)
    s2 = side(c, d, b)

    # No touching and on the same side
    if s1 and s1 == s2:
        return False

    return True


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
    return x_fit, y_fit














