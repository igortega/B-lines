import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from polar import max_slope, polar
import os
import cv2
from dataset import load_data
from scoring import get_score_array
from sklearn.model_selection import train_test_split


def polar_format(n_rows=10, n_cols=10):
    # Get frame dimensions
    xlen, ylen = 905, 632

    # Polar image array
    polar_img = np.zeros((n_rows, n_cols))

    # Cartesian coordinates array
    x = np.array(range(xlen))
    y = np.array(range(ylen))
    Y, X = np.meshgrid(y, x)

    # Polar coordinates array
    rows, cols = np.zeros((xlen, ylen)), np.zeros((xlen, ylen))

    # Calculate image boundaries
    dx, dy = max_slope(10)
    ymid = int(ylen/2)
    m0 = dy/dx
    b0 = ymid - dy
    x0 = int(b0/m0)
    r0 = xlen + x0

    # Radius and slope arrays
    R = np.sqrt((X+x0)**2 + (Y-ymid)**2)
    S = (Y-ymid)/(X+x0)

    # Calculate region out of boundaries
    out = (S > m0) | (S < -m0) | (R > r0)

    # Sample radii and slopes
    r_vec = np.linspace(x0, r0, n_rows+1)
    m_vec = np.linspace(-m0, m0, n_cols+1)

    # Segment image by radius and slope
    i = 0
    for radius in r_vec:
        rows[R >= radius] = i
        i += 1

    j = 0
    for slope in m_vec:
        cols[S > slope] = j
        j += 1

    # Discard region out of boundaries
    rows[out] = np.NaN
    cols[out] = np.NaN

    sections = rows + cols
    plt.imshow(sections)
    plt.savefig('figures/polar_original.png')

    for i in range(n_rows):
        for j in range(n_cols):
            segment = sections[(rows == i) & (cols == j)]
            polar_img[i, j] = segment.mean()

    plt.imshow(polar_img)
    plt.show()
    plt.savefig('figures/polar_new.png')

    return sections, polar_img


def polar_example():
    img = cv2.imread('polar/video_9/centroid_0.jpg')
    plt.imshow(img)
    plt.savefig('figures/polar.png')


def feature_histograms(score_array, labels):
    """ Plot histograms of all features by class """
    feature_list = ['Column mean brightness',
                    'Column center of mass',
                    'Bottom quarter mean brightness',
                    'Maximum bottom value',
                    'Length times mean brightness']

    fig, axs = plt.subplots(nrows=1, ncols=len(feature_list), figsize=(16, 3))
    colors = ('tab:blue', 'tab:orange')
    tags = ('No B-lines', 'B-lines')
    for f in range(len(feature_list)):
        axs[f].set_title(feature_list[f])
        for lab in (0, 1):
            axs[f].hist(score_array[:, f][labels == lab], bins=15, alpha=0.5, label=tags[lab])
            mean_value = score_array[:, f][labels == lab].mean()
            axs[f].axvline(x=mean_value, ymin=0, ymax=1, color=colors[lab], alpha=0.5)
    axs[f].legend()
    return fig


if __name__ == "__main__":
    sector_images, polar_images, labels = load_data()
    score_array = get_score_array(polar_images)
    train_scores, X_test, train_labels, y_test = train_test_split(score_array, labels, test_size=0.2)
    fig = feature_histograms(train_scores, train_labels)

