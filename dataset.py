"""
Data set functions
"""

import numpy as np
import pandas as pd
import os
import cv2
import random
import matplotlib.pyplot as plt


def load_data(labels_path='frame_labels.csv'):
    """ Get arrays of normalized sector images, polar images and corresponding labels

    Parameters
    ----------
    labels_path

    Returns
    -------

    """
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

