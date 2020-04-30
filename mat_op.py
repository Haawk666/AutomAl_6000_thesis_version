# Matrix operation and manipulation script
# -------------------------------------------
# This script contains typical functions that manipulate matrices with inputs and outputs that are logically
# connected to the main program. All inputs "mat" (short for matrix) are assumed to be 2d matrices with float64
# elements normalized to be in the range [0, 1]. Hello

import numpy as np
from PIL import Image
import os.path


def gen_fft(mat):
    """Generate the fast Fourier transform of a matrix."""
    fourier = np.fft.fft2(mat)
    fourier = np.fft.fftshift(fourier)
    fourier = abs(fourier)
    fourier = np.log10(fourier)
    lowest = np.nanmin(fourier[np.isfinite(fourier)])
    highest = np.nanmax(fourier[np.isfinite(fourier)])
    original_range = highest - lowest
    fourier = (fourier - lowest) / original_range * 255
    return fourier


def normalize_static(mat):
    """Normalize matrix such that its elements are \in [0, 1]"""
    # General matrix norm to interval (0, 1). Assumed to be float types
    if mat.min() == mat.max() or mat.max() == 0.0:
        pass
    else:
        mat = mat - mat.min()
        mat = (mat / mat.max())
    return mat


def gen_framed_mat(mat, r):
    """Return input matrix padded with a frame of zeros and width r.

    :param mat: Matrix to be padded.
    :type mat: np.array
    :param r: Width of frame
    :type r: int
    :returns Framed matrix:
    :rtype np.array:

    """
    hor_frame = np.zeros((r, mat.shape[1]), dtype=type(mat))
    ver_frame = np.zeros((mat.shape[0] + 2 * r, r), dtype=type(mat))
    framed_mat = np.concatenate((hor_frame, mat, hor_frame), axis=0)
    framed_mat = np.concatenate((ver_frame, framed_mat, ver_frame), axis=1)
    return framed_mat


def gen_de_framed_mat(temp_mat, r):
    """Return input matrix de-framed by width r.

    :param temp_mat: Matrix to be de-padded.
    :type temp_mat: np.array
    :param r: Width of frame
    :type r: int
    :returns De-framed matrix:
    :rtype np.array:

    """
    (height, width) = temp_mat.shape
    return temp_mat[r:height - r, r:width - r]


def delete_pixels(mat, x_0, y_0, r):
    # Deletes (or sets elements to 0) in the (discrete)-circular area defined by it's centre (x_0, y_0) and radius r.
    for x_i in range(x_0 - r, x_0 + r):
        for y_i in range(y_0 - r, y_0 + r):
            if np.sqrt((x_0 - x_i)**2 + (y_0 - y_i)**2) <= r:
                mat[y_i, x_i] = 0.0
    return mat


def draw_circle(mat, x_0, y_0, r):
    # Draws a simple pixel-wide ring in the input matrix centered at (x_0, y_0) with radius r. The input matrix is
    # assumed to be sparse, that is, mostly contains 0's.
    for x_i in range(x_0 - r, x_0 + r):
        for y_i in range(y_0 - r, y_0 + r):
            distance = np.sqrt((x_0 - x_i) ** 2 + (y_0 - y_i) ** 2)
            if np.floor(distance) == r:
                mat[y_i, x_i] = 1
    return mat


def average(mat, x_0, y_0, r):
    # Calculate the average value of the elements in the circle centered at (x_0, y_0) with radius r.
    element_sum = 0.0
    peak_element = 0.0
    counter = 0
    for x_i in range(x_0 - r, x_0 + r):
        for y_i in range(y_0 - r, y_0 + r):
            distance = np.sqrt((x_0 - x_i) ** 2 + (y_0 - y_i) ** 2)
            if distance <= r:
                element_sum = element_sum + mat[y_i, x_i]
                counter = counter + 1
                if mat[y_i, x_i] > peak_element:
                    peak_element = mat[y_i, x_i]
    element_average = element_sum / counter
    return element_average, peak_element


def find_index_from_coor(mat, x, y):
    return mat[y, x, 1]


def find_coor_from_index(mat, i):

    (M, N) = mat.shape()
    for x in range(0, N):
        for y in range(0, M):
            if mat[y, x, 1] == i:
                return x, y
    print('Error?')


def im_out_static(im_mat, filename_full):

    im_mat = (im_mat - np.min(im_mat)) / (np.max(im_mat) - np.min(im_mat))
    im_mat = np.uint8(np.round(im_mat * 255))
    im_dsp = Image.fromarray(im_mat)
    im_dsp.save(os.path.join(filename_full))


