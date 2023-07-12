import skimage.filters
import skimage.io
import skimage.color
import skimage.segmentation
import numpy as np
import skimage.feature
import skimage.morphology

import matplotlib.pyplot as plt
from skimage import data, feature, filters
from skimage.color import label2rgb
from skimage.measure import label
from skimage.segmentation import clear_border, watershed
from skimage.morphology import closing, square

from skimage import io, feature, color
from skimage.transform import probabilistic_hough_line

def Thresholding():
    image = skimage.io.imread('path/to/image.jpg')
    gray_image = skimage.color.rgb2gray(image)
    threshold_value = skimage.filters.threshold_otsu(gray_image)
    binary_image = gray_image > threshold_value


def edge_based():
    image = skimage.io.imread('path/to/image.jpg')
    edges = skimage.feature.canny(image, sigma=3)


def region_based():
    image = skimage.io.imread('path/to/image.jpg')
    elevation_map = skimage.filters.sobel(image)
    markers = skimage.segmentation.watershed(elevation_map, markers=250)


def active_contours():

    image = skimage.io.imread('path/to/image.jpg')
    gray_image = skimage.color.rgb2gray(image)

    # Initialize a circular contour
    s = np.linspace(0, 2 * np.pi, 400)
    init = 100 * np.array([np.cos(s), np.sin(s)]).T + 100

    snake = skimage.segmentation.active_contour(gray_image, init, alpha=0.015, beta=10, gamma=0.001)


def random_walkers():
    image = skimage.io.imread('path/to/image.jpg')
    markers = np.zeros_like(image)
    markers[20, 20] = 1  # Marker for object
    markers[40, 40] = 2  # Marker for background

    segmentation = skimage.segmentation.random_walker(image, markers)


def erosion():
    image = skimage.io.imread('path/to/image.jpg')
    erosion_result = skimage.morphology.erosion(image, skimage.morphology.disk(5))


def corner_detection(path):
    image = io.imread("../../import_Almere/segmentation.jpg")
    # Load and convert the image to grayscale
    gray_image = color.rgb2gray(image)
    # Perform Harris corner detection
    corners = feature.corner_harris(gray_image)
    # Find corner coordinates
    corner_coordinates = feature.corner_peaks(corners, min_distance=30)
    # Visualize the results
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)
    ax.plot(corner_coordinates[:, 1], corner_coordinates[:, 0], 'ro', markersize=4, markeredgecolor='none')
    ax.axis('off')
    plt.show()


if __name__ == '__main__':

    path = "../../import_Almere/segmentation.jpg"
    corner_detection(path)