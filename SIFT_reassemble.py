from configparser import Interpolation
from gettext import find
from webbrowser import get
import numpy as np
import os
import re
import cv2
import math
from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from functools import cmp_to_key
from matplotlib import pyplot as plt

# Parameters for SIFT. Can be ajusted from default values if results are not satisfactory

sigma = 1.6
sigma_diff = sqrt((sigma)**2 - (2*0.5)**2)
float_tolerance = 1e-7 # For floating point estimation
n_scales_octave = 3
n_img_octave = n_scales_octave + 3
k = 2**(1./n_scales_octave)     # k is the space / size of the interval between subsequent images / scales in an octave.
contrast_threshold = 0.04
threshold = math.floor(0.5 * contrast_threshold / n_scales_octave * 255)


# To generate the scale space, we first need to know how many times we can downsample the image before it is too small for our calculations of local extrema. (This requires a neighborhood of at least 9 pixels. 3x3)
def find_num_octaves(image):
    smallest_dim = min(image.shape)
    num_octaves = math.floor(math.log(smallest_dim, 2))
    num_octaves = num_octaves if smallest_dim/2**num_octaves >= 3 else num_octaves-1 # Check if image too small

    return num_octaves


# Create scale space images and blur them
# I have chosen to use 3 scales per octave
# Meaning I need 6 images per octave since the first and last scales also require images to compute DoG
def get_gaussian_kernels(sig):
    kernels = np.zeros(n_img_octave)
    kernels[0] = sig

    for i in range(1, n_img_octave):
        sig_total = k*sig
        kernels[i] = sqrt(sig_total**2 - sig**2)
        sig = sig_total

    return kernels


def generate_octaves(img, g_kernels, num_octaves):
    blurred_images = []

    for i in range(num_octaves):
        octave_images = []
        octave_images.append(img)

        for kernel in g_kernels[1:]:
            blurred_img = cv2.GaussianBlur(img, (0, 0), sigmaX=kernel, sigmaY=kernel)
            octave_images.append(blurred_img)
        blurred_images.append(octave_images)


        # The next octave will have the third to last image in the current octave as it's base.
        # This is because of the way we calculate the scaling factor k used to apply blur to the images. (k = 2**(1/scales_per_octave))
        next_base = octave_images[-3]
        img = cv2.resize(next_base, (int(next_base.shape[1]/2), int(next_base.shape[0]/2)), interpolation=cv2.INTER_NEAREST) 

    return np.array(blurred_images)


# Calculate the difference of gaussian images
def difference_of_gaussians(gaussian_images):
    dog_images = [] # dog for difference of gaussian, not actually images of dogs :)

    for octave in gaussian_images:
        dog_imgs_octave = []
        for i in range(1,len(octave)):
            dog_img = cv2.subtract(octave[i], octave[i-1])
            dog_imgs_octave.append(dog_img)

        dog_images.append(dog_imgs_octave)
    
    return np.array(dog_images)


# First we'll start by checking if a pixel is a local extremum
def is_local_extremum(tile1, tile2, tile3):
    px_abs_val = abs(tile2[1,1])

    # Check if pixel value is sufficiently "distinct" to try to combat noise
    if px_abs_val > threshold:
        # Then check if the pixel value is greater than all other surrounding pixel values.
        # Only checking for the absolute value, since we just care if it is an extremum, not what kind.
        return all(px_abs_val >= abs(tile1)) and all(px_abs_val >= abs(tile2)) and all(px_abs_val >= abs(tile3))
    
    return False


""" 
If a pixel is a local extremum, we have to find it's location. We have to do this because alot of the extrema might actually not be situated exactly on top of a pixel, they're more likely to be located in sub-pixel locations, meaning we will have to approximate their actual location. This is done via quadratic fit optimization.
"""
def localize_extremum_via_quadratic_fit(i, j, img_i, octave_i, dog_imgs_octave, border_w, eigenvalue_ratio=10, n_attempts=5):

    img_shape = dog_imgs_octave[0].shape
    #print(dog_imgs_octave.shape, ", ", img_shape)
    #print(img_i)

    for attempt in range(n_attempts):
        first_img, second_img, third_img = dog_imgs_octave[img_i-1:img_i+2]

        # Neighborhood we are examining for each pixel
        cube = stack([first_img[i-1:i+2, j-1:j+2],
                      second_img[i-1:i+2, j-1:j+2],
                      third_img[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
        
        # Find gradient and hessian to find corners in the image, these are what we really want :)
        # Both the gradient and the hessian are approximated using finite difference approximations, making them less expensive to compute. 
        # It is also due to the fact that we are not dealing with a continuous function here. We are dealing with pixels. This makes h in the formula for finite differences become 1. 
        # Since we are using second-order finite differences, the formula for the gradient simplifies to:
        #   (f(x+1) - f(x-1)) / 2
        dx = 0.5 * (cube[1, 1, 2] - cube[1, 1, 0])
        dy = 0.5 * (cube[1, 2, 1] - cube[1, 0, 1])
        ds = 0.5 * (cube[2, 1, 1] - cube[0, 1, 1])
        gradient = np.array([dx, dy, ds])

        px_val = cube[1, 1, 1]

        # Second-order finite-diff approx for hessian
        dxx = cube[1, 1, 2] - 2 * px_val + cube[1, 1, 0]
        dyy = cube[1, 2, 1] - 2 * px_val + cube[1, 0, 1]
        dss = cube[2, 1, 1] - 2 * px_val + cube[0, 1, 1]
        dxy = 0.25 * (cube[1, 2, 2] - cube[1, 2, 0] - cube[1, 0, 2] + cube[1, 0, 0])
        dxs = 0.25 * (cube[2, 1, 2] - cube[2, 1, 0] - cube[0, 1, 2] + cube[0, 1, 0])
        dys = 0.25 * (cube[2, 2, 1] - cube[2, 0, 1] - cube[0, 2, 1] + cube[0, 0, 1])
        hessian = array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])

        position_update = -lstsq(hessian, gradient, rcond=None)[0]
        # Check if sufficient convergence has been met / satisfied. If under 0.5 change, means we have converged to changes of under a pixel and we can stop
        if all(abs(position_update) < 0.5):
            break

        # If not, continue iterating by updating x (j) and y (i) by the least squares solution
        j += int(round(position_update[0]))
        i += int(round(position_update[1]))
        img_i += int(round(position_update[2]))

        # Check that we are not out of bounds with new extrema location
        if i < border_w or i >= img_shape[0] - border_w or j < border_w or j >= img_shape[1] - border_w or img_i < 1 or img_i > n_scales_octave:
            return None
    
    if attempt == n_attempts - 1: # We did not converge to below a pixel in set amount of time, discard this extremum
        return None

    updated_value = cube[1, 1, 1] + 0.5 * dot(gradient, position_update)
    if abs(updated_value) * n_scales_octave >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = trace(xy_hessian)
        xy_hessian_det = det(xy_hessian)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            keypoint = cv2.KeyPoint()
            keypoint.pt = ((j + position_update[0]) * (2 ** octave_i), (i + position_update[1]) * (2 ** octave_i))
            keypoint.octave = octave_i + img_i * (2 ** 8) + int(round((position_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((img_i + position_update[2]) / float32(n_scales_octave))) * (2 ** (octave_i + 1))  # octave_index + 1 because the input image was doubled
            keypoint.response = abs(updated_value)
            return keypoint, img_i
    return None


def compute_keypoints_with_orientations(keypoint, octave_i, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    """
    A little explanation of the different variables:
    -   radius_factor: The base-radius of the square that we do computations on
    -   num_bins:      The amount of parts to divide degrees into. We have 10 degrees per bin -> num_bins=36
    -   scale_factor:  The amount we scaled the image for each octave
    -   weight_factor: Determines how much a pixel's contribution should drop off per unit of distance from the keypoint.

    """
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale = scale_factor * keypoint.size / float32(2 ** (octave_i + 1))
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = zeros(num_bins)
    smooth_histogram = zeros(num_bins)

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / float32(2 ** octave_i))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / float32(2 ** octave_i))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    gradient_magnitude = sqrt(dx * dx + dy * dy)
                    gradient_orientation = rad2deg(arctan2(dy, dx))

                    weight = exp(weight_factor * (i ** 2 + j ** 2)) # Give more weight to pixels close to the keypoint being examined
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    orientation_max = max(smooth_histogram)
    orientation_peaks = where(logical_and(smooth_histogram > roll(smooth_histogram, 1), smooth_histogram > roll(smooth_histogram, -1)))[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            # Quadratic peak interpolation
            # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < float_tolerance:
                orientation = 0
            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations


def find_keypoints(gaussian_images, dog_images, border_w=5):
    keypoints = []

    for octave_i, dog_imgs_octave in enumerate(dog_images):
        for img_i in range(1, len(dog_imgs_octave)-1):
            first_img, second_img, third_img = dog_imgs_octave[img_i-1:img_i+2]

            # This nested for-loop checks every 3x3x3 box of pixels to detect local extrema.
            # If the pixel in question is not an extrema, we just move on to the next
            for i in range(border_w, first_img.shape[0] - border_w):
                for j in range(border_w, first_img.shape[1] - border_w):
                    if is_local_extremum(first_img[i-1:i+2, j-1:j+2], second_img[i-1:i+2, j-1:j+2], third_img[i-1:i+2, j-1:j+2]):
                        # If we have found a local extrema, we will have to localize it's exact (almost) position.
                        # This is due to the fact that extrema are most likely not at an exact whole pixel value, we are mostly dealing with sub-pixel locations in this step.
                        # This localization is done via a quadratic fit
                        loc_res = localize_extremum_via_quadratic_fit(i, j, img_i, octave_i, dog_imgs_octave, border_w)
                        if loc_res is not None:
                            # The keypoint at this location is valid and has passed all the checks
                            # Proceed by finding it's orientation
                            keypoint, loc_img_i = loc_res
                            keypoints_w_orientations = compute_keypoints_with_orientations(keypoint, octave_i, gaussian_images[octave_i][loc_img_i])
                            for k in keypoints_w_orientations:
                                keypoints.append(k)

    return keypoints

"""
This following section is heavily fetched from https://github.com/rmislam/PythonSIFT/blob/master/pysift.py due to time constraints.
Yes, I started later than I would like
"""
def convert_keypoints_to_input_image_size(keypoints):
    """Convert keypoint point, size, and octave to input image size
    """
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints

def compare_keypoints(keypoint1, keypoint2):
    """Return True if keypoint1 is less than keypoint2
    """
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id

def remove_duplicate_keypoints(keypoints):
    """
    Sort keypoints and remove duplicate keypoints
    """
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compare_keypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
           last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
           last_unique_keypoint.size != next_keypoint.size or \
           last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints


def unpack_octave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / float32(1 << octave) if octave >= 0 else float32(1 << -octave)
    return octave, layer, scale

def generate_descriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    """
    Generate descriptors for each keypoint
    """
    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpack_octave(keypoint)
        gaussian_image = gaussian_images[octave + 1, layer]
        num_rows, num_cols = gaussian_image.shape
        point = round(scale * array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        cos_angle = cos(deg2rad(angle))
        sin_angle = sin(deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = zeros((window_width + 2, window_width + 2, num_bins))   # first two dimensions are increased by 2 to account for border effects

        # Descriptor window size (described by half_width) follows OpenCV convention
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(round(hist_width * sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
        half_width = int(min(half_width, sqrt(num_rows ** 2 + num_cols ** 2)))     # ensure half_width lies within image

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = sqrt(dx * dx + dy * dy)
                        gradient_orientation = rad2deg(arctan2(dy, dx)) % 360
                        weight = exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            # Smoothing via trilinear interpolation
            # Notations follows https://en.wikipedia.org/wiki/Trilinear_interpolation
            # Note that we are really doing the inverse of trilinear interpolation here (we take the center value of the cube and distribute it among its eight neighbors)
            row_bin_floor, col_bin_floor, orientation_bin_floor = floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        # Threshold and normalize descriptor_vector
        threshold = norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(norm(descriptor_vector), float_tolerance)
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        descriptor_vector = round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return array(descriptors, dtype='float32')


def get_keypoints_and_descriptors(image):
    """
        Takes a grayscale image as a float32 array and generates keypoints along with descriptors
    """

    img = cv2.resize(image, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)

    # Find amount of octaves possible for the input image.
    # The smallest image dimension must at least be 3x3 pixels.
    num_octaves = find_num_octaves(img)
    
    # Find gaussian kernels used to blur the images in each octave
    # The blur simulates seeing the same picture at different scales / distances by removing detail
    kernels = get_gaussian_kernels(sigma)

    # Generate n octaves with m pictures each.
    blurred_images = generate_octaves(img, kernels, num_octaves)

    # Calculate the differences between each pixel in each of the scales in the octaves.
    # This difference of gaussian function can closely approximate the laplacian of the gaussian, but much more cost-effective!
    diff_gauss_images = difference_of_gaussians(blurred_images)

    # Use these images that almost act as edge maps to find points of interest (keypoints)
    keypoints = find_keypoints(blurred_images, diff_gauss_images)
    keypoints = remove_duplicate_keypoints(keypoints) # These might contain duplicates, so we want to remove redundant ones
    
    # Adjust for the fact that keypoints can be found at different scales, meaning that we will have to move them according to the difference in scale from the original input image
    keypoints = convert_keypoints_to_input_image_size(keypoints)

    descriptors = generate_descriptors(keypoints, blurred_images)

    return keypoints, descriptors


import random
if __name__ == "__main__":
    path_to_main = "kitty_test.png"
    split_img_folder = "splits/kitty_test/"

    image_main = cv2.imread(path_to_main, 0) #grayscale to normalize the color values
    sub_img_fnames = os.listdir(split_img_folder) 
    random.shuffle(sub_img_fnames) # Just randomizing to show that the algorithm actually puts them in the right place
    sub_images = [cv2.imread(str(split_img_folder + fname), 0) for fname in sub_img_fnames]

    # Testing with rotations as well
    for i in range(int(len(sub_images)/2)):
        sub_images[i] = cv2.rotate(sub_images[i], cv2.ROTATE_90_CLOCKWISE)

    image_float32 = image_main.astype("float32")
    sub_images_float32 = [img.astype("float32") for img in sub_images]


    kp_main, des_main = get_keypoints_and_descriptors(image_float32)


    # Match descriptors and find subimage
    # FLANN-based matcher fetched from:
    #   https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    MIN_MATCH_COUNT = 10

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    f = plt.figure(figsize=(25,25))
    for i, sub_img in enumerate(sub_images_float32):
        kp_sub, des_sub = get_keypoints_and_descriptors(sub_img)
        matches = flann.knnMatch(des_sub,des_main,k=2)

        # ratio test as per Lowe's paper
        good_matches = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good_matches.append(m)

        if len(good_matches) > MIN_MATCH_COUNT:
            # Estimate homography between template and scene
            src_pts = np.float32([ kp_sub[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([ kp_main[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

            # Draw detected template in scene image
            h, w = sub_images[i].shape
            pts = np.float32([[0, 0],
                        [0, h - 1],
                        [w - 1, h - 1],
                        [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            image_main = cv2.polylines(image_main, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            h1, w1 = sub_images[i].shape
            h2, w2 = image_main.shape
            print("img1: ", h1, w1)
            print("img2: ", h2, w2)

            nWidth = w1 + w2
            nHeight = max(h1, h2)
            hdif = int((h2 - h1) / 2)
            newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

            for j in range(3):
                newimg[hdif:hdif + h1, :w1, j] = sub_images[i]
                newimg[:h2, w1:w1 + w2, j] = image_main

            for m in good_matches:
                pt1 = (int(kp_sub[m.queryIdx].pt[0]), int(kp_sub[m.queryIdx].pt[1] + hdif))
                pt2 = (int(kp_main[m.trainIdx].pt[0] + w1), int(kp_main[m.trainIdx].pt[1]))
                cv2.line(newimg, pt1, pt2, (255, 0, 0))
            f.add_subplot(1, len(sub_images), i+1)
            plt.imshow(newimg)
    
    plt.show()


