import sol4_utils as utils
import sol4_add
import numpy as np
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from scipy.signal import convolve as convolve
from scipy.signal import convolve2d as convolve2d
from scipy.ndimage.filters import convolve


def harris_corner_detector(im):
    
    K = 0.04

    # create convolution arrays
    dx_vec = np.array([[1, 0, -1]], )
    dy_vec = dx_vec.reshape(dx_vec.size, 1)

    # get dx and dy
    dx_img = convolve2d(im, dx_vec, mode='same')
    dy_img = convolve2d(im, dy_vec, mode='same')

    # multiply matrices and blur
    dx_sqr = utils.blur_spatial(dx_img.dot(dx_img), 3)
    dy_sqr = utils.blur_spatial(dy_img.dot(dy_img), 3)
    dx_dy = utils.blur_spatial(dx_img.dot(dy_img), 3)
    dy_dx = utils.blur_spatial(dy_img.dot(dx_img), 3)

    # calc det, trace and R
    det = dx_sqr * dy_sqr - dx_dy * dy_dx
    trace = dx_sqr + dy_sqr

    R_arr = det - K * (trace ** 2)

    return sol4_add.non_maximum_suppression(R_arr)
