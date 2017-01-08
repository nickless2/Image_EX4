import numpy as np
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from scipy.signal import convolve as sig_convolve
from scipy.signal import convolve2d as sig_convolve2d
from scipy.ndimage.filters import convolve


def read_image(filename, representation):

    im = imread(filename)
    # check if we want to convert RGB pic to greyscale
    if representation == 1 and im.shape.__len__() == 3:
        im = rgb2gray(im)
        im = im.astype(np.float32)

    else:
        im = im.astype(np.float32)
        im /= 255

    return im


def blur_spatial(im, kernel_size):

    kernel = calc_kernel(kernel_size)

    return sig_convolve2d(im, kernel, mode='same')


def calc_kernel(kernel_size):

    dim1_kernel = np.array([1, 1]).astype(np.float32)
    dim1_result_kernel = dim1_kernel

    for i in range(kernel_size - 2):
        dim1_result_kernel = sig_convolve(dim1_result_kernel, dim1_kernel)

    norm_final_kernel = dim1_result_kernel / np.sum(dim1_result_kernel)

    return np.array([norm_final_kernel])


def reduce(im, filter_vec):

    # blur
    row_convolution = convolve(im, filter_vec)
    final_convolution = convolve(row_convolution, filter_vec.transpose())

    # sub-sample every second pixel
    final_convolution = final_convolution[::2, ::2]

    return final_convolution


def expand(im, filter_vec):

    # zero padding
    extended_im = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    extended_im[::2, ::2] = im

    # blur
    extended_im = convolve(extended_im, 2 * filter_vec)
    extended_im = convolve(extended_im, 2 * filter_vec.transpose())

    return extended_im


def build_gaussian_pyramid(im, max_levels, filter_size):

    # minimum image resolution
    MIN_IM_SIZE = 16

    filter_vec = calc_kernel(filter_size)

    # insert "im" as first item in regular python list
    pyr = [im]

    # perform reduce for "max_levels"
    for i in range(1, max_levels):
        reduced_im = reduce(pyr[i-1], filter_vec)
        if reduced_im.shape[0] < MIN_IM_SIZE or reduced_im.shape[1] < \
                MIN_IM_SIZE:
            break
        pyr.append(reduced_im)

    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):

    gauss_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    laplac_pyr = []

    for i in range(len(gauss_pyr) - 1):
        laplac_pyr.append(gauss_pyr[i] - expand(gauss_pyr[i+1], filter_vec))

    laplac_pyr.append(gauss_pyr[-1])

    return laplac_pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):

    im = lpyr[-1]

    for i in range(len(lpyr) - 2, -1, -1):
        im = expand(im, filter_vec)
        im = im + lpyr[i] * coeff[i]

    return im


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):

    # create respective pyramids
    im1_pyr, im1_filter = build_laplacian_pyramid(im1, max_levels,
                                                  filter_size_im)
    im2_pyr, im2_filter = build_laplacian_pyramid(im2, max_levels,
                                                  filter_size_im)
    mask_pyr, mask_filter = build_gaussian_pyramid(mask.astype(np.float32),
                                                   max_levels, filter_size_mask)

    # construct laplacian pyramid from blending
    out_im_pyr = []
    for i in range(len(mask_pyr)):
        out_im_pyr.append(mask_pyr[i] * im1_pyr[i] + (1 - mask_pyr[i]) *
                          im2_pyr[i])

    # construct image from laplacian pyramid
    out_im = laplacian_to_image(out_im_pyr, im1_filter, np.ones(len(out_im_pyr)))
    out_im = np.clip(out_im, 0, 1)

    return out_im