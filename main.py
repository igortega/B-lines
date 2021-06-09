"""

"""

import numpy as np
import pandas as pd
import os
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


def load_data(labels_path='frame_labels.csv'):
    """ Get arrays of normalized sector images, polar images and corresponding labels """
    labels_df = pd.read_csv(labels_path, sep=';')

    sector_dataset_dir = 'dataset'
    polar_dataset_dir = 'polar_dataset'

    sector_images = []
    polar_images = []
    for frame_name in labels_df['frame']:
        sector_image_path = os.path.join(sector_dataset_dir, frame_name + '.png')
        polar_image_path = os.path.join(polar_dataset_dir, frame_name + '.npy')

        sector_image = cv2.imread(sector_image_path)[:, :, 0]/255
        sector_image = cv2.resize(sector_image, (200, 200))

        polar_image = np.load(polar_image_path)
        polar_image[np.isnan(polar_image) == False] /= 255

        sector_images.append(sector_image)
        polar_images.append(polar_image)

    labels = np.array(labels_df['label'])
    return sector_images, polar_images, labels


def mean_below_pleura(polar_image):
    """ Returns highest column's brightness mean below maximum (pleural line) """
    peak_ycoordinates = np.nanargmax(polar_image, axis=0)
    mean_list = []
    for k in range(len(peak_ycoordinates)):
        mean = np.nanmean(polar_image[peak_ycoordinates[k]:, k])
        mean_list.append(mean)

    score = max(mean_list)
    return score


def center_of_mass(polar_image):
    """ Returns maximum y coordinate of column's center of mass """
    polar_image[np.isnan(polar_image)] = 0
    coords = list(range(len(polar_image)))
    center_list = []
    for k in range(len(polar_image)):
        column = polar_image[:, k]
        column_center = np.sum(column * coords) / np.sum(column)
        center_list.append(column_center)

    score = max(center_list) / len(polar_image)
    return score


def bottom_quarter(polar_image):
    """ Returns highest column's brightness mean over bottom quarter """
    image_len = len(polar_image)
    quarter = round(3 * image_len / 4)
    quarter_mean = np.nanmean(polar_image[quarter:, :], axis=0)
    score = np.max(quarter_mean)
    return score


def bottom_max(polar_image):
    """ Returns maximum intensity at most bottom point of a column """
    bottom_values = []
    for k in range(len(polar_image)):
        column = polar_image[:, k]
        try:
            first_nan = list(np.isnan(column)).index(True)
            bottom_value = column[first_nan - 1]
        except ValueError:
            bottom_value = column[-1]

        bottom_values.append(bottom_value)
    score = np.max(bottom_values)
    return score


def length_times_mean(polar_image):
    """ Returns maximum product of column's length below peak (pleura) times its average brightness """
    peak_ycoordinates = np.nanargmax(polar_image, axis=0)
    score_list = []
    for k in range(len(polar_image)):
        column = polar_image[peak_ycoordinates[k]:, k]
        column_length = np.sum(column[np.isnan(column) == False]) / 50
        column_mean = np.nanmean(column)
        score_list.append(column_mean * column_length)

    score = max(score_list)
    return score


def get_score_array(polar_images_list,
                    score_functions=[mean_below_pleura, center_of_mass, bottom_quarter, bottom_max, length_times_mean]):
    """ Returns array shaped (number of samples, number of features) containing scores of input data """

    n_samples = len(polar_images_list)
    score_array = np.zeros((n_samples, len(score_functions)))
    for k in range(n_samples):
        for j in range(len(score_functions)):
            score_array[k, j] = score_functions[j](polar_images_list[k])

    return score_array


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

    # FEATURE-BASED CLASSIFICATION
    score_array = get_score_array(polar_images)

    train_scores, test_scores, train_labels, test_labels = train_test_split(score_array, labels, test_size=0.2, random_state=42)

    feature_histograms(train_scores, train_labels)

    feature_reg = LogisticRegression(class_weight='balanced', random_state=42)
    feature_reg.fit(train_scores, train_labels)

    feature_test_predict = feature_reg.predict(test_scores)

    print('Feature-based classification: ')
    feature_correct_predictions = sum(feature_test_predict == test_labels)
    print(feature_correct_predictions, "correct predictions out of", len(test_labels))
    print(feature_correct_predictions / len(test_labels) * 100, "percent accuracy")

    plot_confusion_matrix(feature_reg, test_scores, test_labels,
                          display_labels=np.array(['No B-lines', 'B-lines']),
                          colorbar=False)

    # IMAGE-BASED CLASSIFICATION
    n_samples = len(labels)
    sector_images_data = np.array(sector_images).reshape((n_samples, -1))

    train_images, test_images, train_labels, test_labels = train_test_split(sector_images_data, labels, test_size=0.2, random_state=42)

    image_reg = LogisticRegression(class_weight='balanced', random_state=42)
    image_reg.fit(train_images, train_labels)

    image_test_predict = image_reg.predict(test_images)

    print('Image-based classification: ')
    image_correct_predictions = sum(image_test_predict == test_labels)
    print(image_correct_predictions, "correct predictions out of", len(test_labels))
    print(image_correct_predictions / len(test_labels) * 100, "percent accuracy")

    plot_confusion_matrix(image_reg, test_images, test_labels,
                          display_labels=np.array(['No B-lines', 'B-lines']),
                          colorbar=False)