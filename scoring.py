"""

B-lines scoring functions

"""


import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def highest_mean(polar_img_path):
    """ 
    SCORE 1
    Input: path and depth of polar image to be scored
    Output: score for highest  column mean brightness
    
    """

    polar_img = cv2.imread(polar_img_path)
    polar_img = polar_img[:,:,0]
    
    
    maximum = np.average(polar_img, axis=0).max()
    maximum /= polar_img.max() # Normalization
    
    return maximum



def last_quarter(polar_img_path):
    """ 
    SCORE 2
    Input: path and depth of polar image to be scored
    Output: maximum sum of lowest quarter brightness
    
    """
    
    polar_img = cv2.imread(polar_img_path)
    polar_img = polar_img[:,:,0]
    
    quarter_sum = np.average(polar_img[40:,:], axis=0).max()
    quarter_sum /= polar_img.max() # Normalization
    
    return quarter_sum



def above_half_max(polar_img_path):
    """ 
    SCORE 3
    Input: path and depth of polar image to be scored
    Output: length of column above its half maximum
    
    """
    polar_img = cv2.imread(polar_img_path)
    polar_img = polar_img[:,:,0]
    
    half_max = polar_img.max(axis=0)/2
    
    length_list = []
    for k in range(len(polar_img)):
        line = polar_img[:,k]
        above = line > half_max[k]
        length_list.append(np.sum(above))
        
    max_score = max(length_list)/50 # Normalization
    
    return max_score



def video_score(polar_centroids_dir):
    """ 
    Selects and returns maximum scores in every feature for a given video
    
    """
        
    path_list = os.listdir(polar_centroids_dir)
    
    score_array = np.zeros((3,3))
    
    # Calculate scores for each centroid
    for i in range(3):
        path = os.path.join(polar_centroids_dir, path_list[i])      
        score_array[i,0] = highest_mean(path)
        score_array[i,1] = last_quarter(path)
        score_array[i,2] = above_half_max(path)
    
    # Select maximum score for each feature
    score_maximum = score_array.max(axis=0)
    
    return score_maximum


def score_all(video_paths):
    video_paths = np.array(video_paths)
    score_array = np.zeros((len(video_paths), 3))
    
    for k in range(len(video_paths)):
        path = os.path.join('key_frames', video_paths[k])
        score_array[k, :] = video_score(path)
    
    np.savetxt('scores.txt', score_array, fmt='%.2f')
    return score_array

