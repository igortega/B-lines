"""

B-lines scoring functions

"""

import numpy as np
import matplotlib.pyplot as plt


def mean_below_pleura(polar_image):
    """ Returns highest column's brightness mean below maximum (pleural line)

    Parameters
    ----------
    polar_image

    Returns
    -------

    """
    peak_ycoordinates = np.nanargmax(polar_image, axis=0)
    mean_list = []
    for k in range(len(peak_ycoordinates)):
        mean = np.nanmean(polar_image[peak_ycoordinates[k]:, k])
        mean_list.append(mean)

    score = max(mean_list)
    return score


def center_of_mass(polar_image):
    """ Returns maximum y coordinate of column's center of mass

    Parameters
    ----------
    polar_image

    Returns
    -------

    """
    polar_image[np.isnan(polar_image)] = 0
    coords = list(range(len(polar_image)))
    center_list = []
    for k in range(len(polar_image)):
        column = polar_image[:, k]
        column_center = np.sum(column*coords)/np.sum(column)
        center_list.append(column_center)

    score = max(center_list)/len(polar_image)
    return score


def bottom_quarter(polar_image):
    """ Returns highest column's brightness mean over bottom quarter

    Parameters
    ----------
    polar_image

    Returns
    -------

    """
    image_len = len(polar_image)
    quarter = round(3*image_len/4)
    quarter_mean = np.nanmean(polar_image[quarter:, :], axis=0)
    score = np.max(quarter_mean)
    return score


def bottom_max(polar_image):
    """ Returns maximum intensity at most bottom point of a column

    Parameters
    ----------
    polar_image

    Returns
    -------

    """
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
    """ Returns maximum product of column's length below peak (pleura) times its average brightness

    Parameters
    ----------
    polar_image

    Returns
    -------

    """
    peak_ycoordinates = np.nanargmax(polar_image, axis=0)
    score_list = []
    for k in range(len(polar_image)):
        column = polar_image[peak_ycoordinates[k]:, k]
        column_length = np.sum(column[np.isnan(column) == False])/50
        column_mean = np.nanmean(column)
        score_list.append(column_mean*column_length)

    score = max(score_list)
    return score


def get_score_array(polar_images_list,
                    score_functions=[mean_below_pleura, center_of_mass, bottom_quarter, bottom_max, length_times_mean]):
    """ Returns array shaped (number of samples, number of features) containing scores of input data

    Parameters
    ----------
    polar_images_list
    score_functions

    Returns
    -------

    """
    n_samples = len(polar_images_list)
    score_array = np.zeros((n_samples, len(score_functions)))
    for k in range(n_samples):
        for j in range(len(score_functions)):
            score_array[k, j] = score_functions[j](polar_images_list[k])

    return score_array


