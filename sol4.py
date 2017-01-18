import sol4_utils as utils
import sol4_add as sol4a
import numpy as np
from scipy.signal import convolve2d as convolve2d
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings("ignore")


def harris_corner_detector(im):
    K = 0.04
    BLUR_FILTER_SIZE = 3

    # create convolution arrays
    dx_vec = np.array([[1, 0, -1]], )
    dy_vec = dx_vec.reshape(dx_vec.size, 1)

    # get dx and dy
    dx_img = convolve2d(im, dx_vec, mode='same')
    dy_img = convolve2d(im, dy_vec, mode='same')

    # multiply matrices and blur
    dx_sqr = utils.blur_spatial(dx_img * dx_img, BLUR_FILTER_SIZE)
    dy_sqr = utils.blur_spatial(dy_img * dy_img, BLUR_FILTER_SIZE)
    dx_dy = utils.blur_spatial(dx_img * dy_img, BLUR_FILTER_SIZE)

    # calc det, trace and R
    det = dx_sqr * dy_sqr - dx_dy * dx_dy
    trace = dx_sqr + dy_sqr
    R_img = det - K * (trace * trace)
    R_img = sol4a.non_maximum_suppression(R_img).transpose()

    return np.argwhere(R_img == True)


def sample_descriptor(im, pos, desc_rad):

    desc = np.zeros((2 * desc_rad + 1, 2 * desc_rad + 1, pos.shape[0]))
    x, y = np.meshgrid(np.arange(-desc_rad, desc_rad + 1),
                       np.arange(-desc_rad, desc_rad + 1))

    for row in range(pos.shape[0]):

        patch = map_coordinates(im, [y + pos[row][1], x + pos[row][0]], order=1,
                                prefilter=False)
        mean = np.mean(patch)
        norm = np.linalg.norm(patch - mean)

        if norm != 0:
            desc[:, :, row] = ((patch - mean) / norm).transpose()

    return desc


def find_features(pyr):
    pos = sol4a.spread_out_corners(pyr[0], 7, 7, 3)
    desc = sample_descriptor(pyr[2], pos/4, 3)
    return pos, desc


def match_features(desc1, desc2, min_score):
    # calculate score
    score = np.tensordot(desc1, desc2, axes=[[0, 1], [0, 1]])
    match_row = np.zeros(score.shape)
    for row in range(score.shape[0]):
        # get second highest number in row and mark accordingly
        second_max = np.sort(score[row, :])[-2]
        match_row[row, score[row, :] >= second_max] = 1

    for col in range(score.shape[1]):
        # get second highest number in col and mark accordingly
        second_max = np.sort(score[:, col])[-2]
        match_col = np.zeros((1, score.shape[0]))
        match_col[0, score[:, col] >= second_max] = 1
        match_row[:, col] = match_row[:, col] * match_col

    match_row[score <= min_score] = 0
    match_row = np.argwhere(match_row == 1).transpose()

    return match_row[0], match_row[1]


def apply_homography(pos1, H12):
    # insert "1" 's as 3rd coordinate
    three_coord_pos1 = (np.insert(pos1, 2, 1, axis=1)).transpose()
    pos2 = H12.dot(three_coord_pos1)
    pos2 /= pos2[2]

    return pos2[:2].transpose()


def ransac_homography(pos1, pos2, num_iters, inlier_tol):

    inliers_max_len = 0

    for i in range(num_iters):
        # choose 4 random indexes
        indexes = np.random.choice(pos1.shape[0], 4)
        H12 = sol4a.least_squares_homography(pos1[indexes], pos2[indexes])

        if H12 is None:
            continue

        new_pos = apply_homography(pos1, H12)
        norm = np.linalg.norm(new_pos - pos2, axis=1)
        err = norm ** 2
        inliers = np.argwhere(err < inlier_tol)
        inliers_len = inliers.shape[0]

        # choose longest inlier
        if inliers_max_len < inliers_len:
            inliers_max_len = inliers_len
            best_inliers = inliers

    best_inliers = best_inliers.flatten()
    homography = sol4a.least_squares_homography(pos1[best_inliers],
                                                pos2[best_inliers])

    return homography, best_inliers


def display_matches(im1, im2, pos1, pos2, inliers):

    pos2[:, 0] += im1.shape[1]
    points = np.append(pos1, pos2, axis=0)
    plt.scatter(points[:, 0], points[:, 1], c='r', zorder=2)
    plt.imshow(np.hstack((im1, im2)), cmap=plt.cm.gray)

    for i in range(pos1.shape[0]):
        if i in inliers:
            plt.plot((pos1[i, 0], pos2[i, 0]),
                     (pos1[i, 1], pos2[i, 1]), 'y-', zorder=3)
        else:
            pass
            # plt.plot((pos1[i, 0], pos2[i, 0]), (pos1[i, 1], pos2[i, 1]),
            #          'b-', zorder=1)

    plt.show()


def accumulate_homographies(H_successive, m):

    H2m = [np.eye(3)]
    H_len = len(H_successive)

    # for i < m
    for i in range(m-1, -1, -1):
        new_H = H2m[0].dot(H_successive[i])
        new_H = new_H / new_H[2, 2]
        H2m.insert(0, new_H)

    # for i > m
    for i in range(m, len(H_successive)):
        inverse_H = np.linalg.inv(H_successive[i])
        new_H = H2m[-1].dot(inverse_H)
        new_H = new_H / new_H[2, 2]
        H2m.append(new_H)

    return H2m


def render_panorama(ims, Hs):
    panorama, pano_x, pano_y, strip_point = background_builder(ims, Hs)
    for i in range(len(ims) - 1):
        if i == 0:
            left_overlap = 0
            right_overlap = 10
        elif i > 1 and i < len(ims):
            left_overlap = 10
            right_overlap = 10
        else:
            left_overlap = 10
            right_overlap = 0

        start = int(strip_point[i] - left_overlap)
        end = int(strip_point[i + 1] + right_overlap)
        strip_x = pano_x[:, start:end]
        strip_y = pano_y[:, start:end]

        # to check if replace this with meshgrid function
        strip_x_2d = strip_x.flatten()
        strip_x_2d = strip_x.reshape((len(strip_x_2d), 1))
        strip_y_2d = strip_y.flatten()
        strip_y_2d = strip_y.reshape((len(strip_y_2d), 1))
        # strip_x_2d, strip_y_2d = np.meshgrid(strip_x, strip_y)

        res = apply_homography(np.column_stack((strip_x_2d, strip_y_2d)),
                               np.linalg.inv(Hs[i]))

        worp_x = np.reshape(res[:, 0], (strip_x.shape[0], strip_x.shape[1]))
        worp_y = np.reshape(res[:, 1], (strip_y.shape[0], strip_y.shape[1]))

        im_new = map_coordinates(ims[i], [worp_y, worp_x], order=1,
                                            prefilter=False)

        im_new[np.isnan(im_new)] = 0

        # panorama_temp = np.zeros((panorama.shape[0], panorama.shape[1]))
        # print(im_new.shape)
        # print(panorama_temp[:, start:end])
        # panorama_temp[:, start:end]


        mask_temp = np.ones((panorama.shape[0], panorama.shape[1]))
        mask_temp[0:im_new.shape[0], int(strip_point[i]):end] = 0
        rows_pad = power_bit_length(mask_temp.shape[0])
        cols_pad = power_bit_length(mask_temp.shape[1])
        panorama_temp = np.zeros((rows_pad, cols_pad))

        mask_new = np.zeros((rows_pad, cols_pad))
        print(mask_new.shape)
        mask_new[0:mask_temp.shape[0], 0:mask_temp.shape[1]] = mask_temp
        panorama_new = np.zeros((rows_pad, cols_pad))

        panorama_new[0:panorama.shape[0], 0:panorama.shape[1]] = panorama
        panorama_temp[0:im_new.shape[0],
        0:im_new.shape[1]] = im_new  # print(panorama_new)
        print(panorama_new.shape)
        print(mask_new.shape)
        print(panorama_temp.shape)
        panorama = utils.pyramid_blending(panorama_new, panorama_temp,
                                               mask_new, 5, 5, 5)

        print("all ok")
        return panorama



def background_builder(ims, Hs):
    """
    makes black background for the concatenated image
    """
    ims_len = len(ims)
    new_center = [0] * ims_len
    new_corners = [0] * ims_len

    for i in range(ims_len):
        rows = ims[i].shape[0]
        cols = ims[i].shape[1]

        corners = np.zeros((4, 2))
        corners[0] = np.array([0, 0])
        corners[1] = np.array([rows-1, 0])
        corners[2] = np.array([0, cols-1])
        corners[3] = np.array([rows-1, cols-1])

        new_corners[i] = apply_homography(corners, Hs[i])
        middle = np.asarray([[cols/2, rows/2]])
        new_center[i] = apply_homography(middle, Hs[i])

    new_corners = np.asarray(new_corners)
    new_corners = new_corners.reshape(new_corners.shape[0] *
                                      new_corners.shape[1], 2)
    new_center = np.asarray(new_center)
    new_center = new_center.reshape(new_center.shape[0] *
                                    new_center.shape[1], 2)

    rows_min = np.floor(min(new_corners[:, 1])).astype(int)
    rows_max = np.ceil(max(new_corners[:, 1])).astype(int)
    cols_min = np.floor(min(new_corners[:, 0])).astype(int)
    cols_max = np.ceil(max(new_corners[:, 0])).astype(int)

    panorama_x, panorama_y = np.meshgrid(range(rows_min, rows_max),
                                         range(cols_min, cols_max))

    background = np.zeros(
        ((rows_max - rows_min - 1), (cols_max - cols_min + 1)))

    strip = np.ones(ims_len)
    for i in range(ims_len - 1):
        strip[i] = (np.floor(new_center[i, 0] +
                             new_center[i+1, 0]) / 2).astype(int) + 1

    strip = np.insert(strip, 0, rows_min)
    strip = np.append(strip, rows_max)
    strip = strip - strip[0] + 1

    return background, panorama_x, panorama_y, strip


# finds the closest power of 2
def power_bit_length(x):
    return 2**(x-1).bit_length()


def main():
    corner_1 = [1, 2]
    x = np.array([1, 2,3])
    print(corner_1[1])
    print(x[1])

    # image = utils.read_image("external/oxford1.jpg", 1)
    #
    # image2 = utils.read_image("external/oxford2.jpg", 1)
    #
    # pyr = utils.build_gaussian_pyramid(image, 3, 3)[0]
    # pyr2 = utils.build_gaussian_pyramid(image2, 3, 3)[0]
    #
    # corners1, desctiptors1 = find_features(pyr)
    # corners2, desctiptors2 = find_features(pyr2)
    # match1, match2 = match_features(desctiptors1, desctiptors2, 0.5)
    #
    # H12, inliers = ransac_homography(corners1[match1, :], corners2[match2, :],
    #                                  1000, 6)
    #
    # # H22, inliers = ransac_homography(corners1[match3, :], corners2[match4, :],
    # #                                  1000, 6)
    #
    # display_matches(image, image2, corners1[match1, :], corners2[match2, :],
    #                 inliers)


if __name__ == "__main__":
    main()
