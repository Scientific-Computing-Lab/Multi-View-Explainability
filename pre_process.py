import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pd
import re
import torchvision.transforms.functional as TF

from PIL import Image
from skimage.exposure import match_histograms
from config import data_dir, full_groups_dir, preprocess_dir
from model2 import verbose


def get_max_width_height(images):
    max_height = 0
    max_width = 0
    for img in images:
        height, width = img.shape
        if height > max_height:
            max_height = height
        if width > max_width:
            max_width = width
    return max_height, max_width


def get_images(source_dir, matched_histograms=False):
    if matched_histograms:
        ref = cv2.imread(f'{full_groups_dir}/T483-2-5-0.png')
        return [cv2.cvtColor(match_histograms(cv2.imread(os.path.join(source_dir, filename)), ref, multichannel=True), cv2.COLOR_BGR2GRAY)
                if int(filename.split('-')[-1][0]) < 2
                else cv2.cvtColor(cv2.imread(os.path.join(source_dir, filename)), cv2.COLOR_BGR2GRAY)
                for filename in os.listdir(source_dir)], os.listdir(source_dir)
    return [cv2.cvtColor((cv2.imread(os.path.join(source_dir, filename))), cv2.COLOR_BGR2GRAY) for filename in os.listdir(source_dir)], os.listdir(source_dir)


def noise(image):
    image = np.array(image)
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.round(np.random.normal(mean, sigma, (row, col, ch)) * 10)
    gauss = gauss.reshape(row, col, ch)
    noisy = abs(image + gauss).astype(int)
    return noisy


def convert_to_png(src, dst):
    for fname in os.listdir(src):
        img = cv2.imread(os.path.join(src, fname))
        name = f'{fname.split(".")[0]}.png'
        cv2.imwrite(os.path.join(dst, name), img)


def convert_to_bins(img, bins):
    img = np.array(img)
    jump = int(255/bins)
    for min_val in range(0, 255-2*jump, jump):
        max_val = min_val + jump
        img[(img >= min_val) & (img <= max_val)] = min_val
    min_val += jump
    img[(img >= min_val)] = min_val
    return img


def find_white_bar(img):
    return int(np.where(img[:, 100:500] == max(img[:, 0]))[0].mean())


def padding(img, max_height, max_width):
    white_bar = find_white_bar(img)
    new_img = np.zeros([max_height, max_width])
    img_width = img.shape[1]
    start = int(new_img.shape[0]/2)-white_bar
    end = start + img.shape[0]
    if end < max_height:
        new_img[start:end, :img_width] = img[:, :]
    else:
        new_img[start:max_height, :img_width] = img[:max_height-start, :]
    return new_img


def min_bounding_circle(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center_coordinates = (int(x), int(y))
    radius = int(radius)
    return center_coordinates, radius


def find_contours(img):
    kernel = np.ones((5, 5), np.uint8)
    img_dilate = cv2.dilate(img, kernel, iterations=0)
    edged = cv2.Canny(img_dilate, 30, 200)
    _, contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def find_min_circle(img, verbose=0):
    contours, hierarchy = find_contours(img)
    if verbose > 0:
        cv2.drawContours(img, contours[0:], -1, (0, 255, 0), 3)
    center_coordinates, radius = min_bounding_circle(np.vstack(contours[0:]))
    return center_coordinates, radius


def bounding_square_crop(img):
    (x, y), radius = find_min_circle(img)
    return img[y-radius:y+radius, x-radius+1:x+radius]


def mask_circle(img, center_coordinates, radius=355, delta=0):
    mask = np.zeros(img.shape, dtype="uint8")
    cv2.circle(img, center_coordinates, radius+delta, (0, 255, 0), 2)
    cv2.circle(mask, center_coordinates, radius+delta, (255, 255, 255), -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    return masked


def center_circle(img):
    center_coordinates, radius = find_min_circle(img)
    img = np.roll(img, int(img.shape[1]/2) - center_coordinates[0], axis=1)
    img = np.roll(img, int(img.shape[0]/2) - center_coordinates[1], axis=0)
    return img, center_coordinates, radius


def detect_circle(img, radius, center_coordinates, verbose=0):
    img = mask_circle(img, center_coordinates=center_coordinates, radius=radius)
    center_coordinates, radius = find_min_circle(img, verbose)
    img = mask_circle(img, center_coordinates, radius)
    return img


def circle_permutation(img):
    min_white_pixels = 10000000
    radii = list(range(295, 320, 4))  # (295, 320, 4)
    for offset_x in range(-60, 60, 3):  # (-60, 60, 3)
        for offset_y in range(-100, 100, 3):  # (-42, 60, 3)  (-100, 60, 3)  (-100, 100, 3)
            for radius in radii:
                center_coordinates = (int(img.shape[0]/2) + offset_x, int(img.shape[1]/2) + offset_y)
                circle = mask_circle(img.copy(), center_coordinates=center_coordinates, radius=radius)
                white_pixels = len(np.where((circle >= 190))[0]) / len(np.where((circle < 190) & (circle != 0))[0])
                if white_pixels < min_white_pixels:
                    min_white_pixels = white_pixels
                    best_circle = circle
                    best_radius = radius
                    best_offset_x = offset_x
                    best_offset_y = offset_y
                    best_center_coordinates = center_coordinates
    if verbose > 1:
        print(f'best x offset: {best_offset_x} \nbest y offset: {best_offset_y} \nbest R: {best_radius} \n')
    return best_circle


def preprocess(images, filenames, save_dir, top_bottom=True, profile=True):
    max_height, max_width = get_max_width_height(images)
    for idx, img in enumerate(images):
        if int(filenames[idx].split('-')[-1][0]) <= 1:
            if top_bottom:
                if verbose > 2:
                    print(filenames[idx])
                img = bounding_square_crop(circle_permutation(img))
                group_name = filenames[idx].split('.')[0]
                cv2.imwrite(os.path.join(save_dir, f'{group_name}.png'), img)
        elif profile:
            img = convert_to_bins(img, bins=10)
            img = padding(img, max_height, max_width)
            cv2.imwrite(os.path.join(save_dir, f'{filenames[idx]}'), img)


source_dir = os.path.join(data_dir, 'circles_preprocess')  # source images directory
save_dir = os.path.join(data_dir, 'circles_preprocess_new')  # directory for saving the pre-processed images
# convert_to_png(src='data/preprocess_profiles', dst='data/preprocess_profiles_new')
images, filenames = get_images(source_dir=source_dir, matched_histograms=False)
preprocess(images, filenames, save_dir=save_dir, top_bottom=True, profile=False)
