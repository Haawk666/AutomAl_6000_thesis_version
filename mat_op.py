# Matrix operation and manipulation script
# -------------------------------------------
# This script contains typical functions that manipulate matrices with inputs and outputs that are logically
# connected to the main program. All inputs "mat" (short for matrix) are assumed to be 2d matrices with float64
# elements normalized to be in the range [0, 1].

import numpy as np
from PIL import Image
import os.path


def gen_fft(mat):
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
    # General matrix norm to interval (0, 1). Assumed to be float types
    if mat.min() == mat.max() or mat.max() == 0.0:
        pass
    else:
        mat = mat - mat.min()
        mat = (mat / mat.max())
    return mat


def gen_framed_mat(mat, r):
    # Generates a new matrix containing the input matrix, but framed with a frame of width r, with elements set to 0
    # in the frame. Used in functions where this is simpler than to consider boarder inputs
    (n, m) = mat.shape
    temp_mat = np.zeros((int(n + 2 * r), int(m + 2 * r)), dtype=mat.dtype)
    for i in range(0, n - 1 + (2 * r)):
        for j in range(0, m + (2 * r) - 1):
            if i in range(r, n + r - 1) and j in range(r, m + r - 1):
                temp_mat[i, j] = mat[i - r, j - r]
    return temp_mat


def gen_de_framed_mat(temp_mat, r):
    # Takes a matrix that is assumed to contain a frame with junk or superfluous data with width r. Returns the central
    # matrix. Used to de-frame a matrix.
    (n, m) = temp_mat.shape
    n = n - (2 * r)
    m = m - (2 * r)
    mat = np.zeros((n, m), dtype=temp_mat.dtype)
    for i in range(0, n - 1):
        for j in range(0, m - 1):
            mat[i, j] = temp_mat[i + r, j + r]
    return mat


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
