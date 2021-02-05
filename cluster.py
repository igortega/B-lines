"""

Functions for frame clustering and video segmentation

"""

from sklearn.cluster import KMeans
from frame_extraction import order
import numpy as np
import pandas as pd
import os
import cv2



def cluster(frames_dir, n_clusters=3):
    """ 
    Calculates 'n_clusters' of frames contained in 'frames_dir' by KMeans
    Saves centroids to 'centroids_dir'
    Returns list of labels for each frame
    
    """
    
    # Get frame paths
    centroids_dir = os.path.join('centroids', frames_dir)
    frames_dir = os.path.join('frames', frames_dir)
    paths = os.listdir(frames_dir)
    paths.sort(key=order)
    
    
    # Read frames and load data
    print('Loading data...')
    data = []
    for name in paths:
        frame_path = os.path.join(frames_dir, name)
        img = cv2.imread(frame_path)
        img = img[:,:,0]
        
        # Reshape and normalize image
        sample = img.reshape(-1)/255
        
        data.append(sample)
        
    data = np.array(data)
    
    # Calculate clusters
    print('Calculating centroids...')
    kmeans = KMeans(n_clusters)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Reshape centroids
    height, width = img.shape
    centroids = centroids.reshape(n_clusters, height, width)
    
    
    # Create directory
    if os.path.exists(centroids_dir) == False:
        os.makedirs(centroids_dir)
    
    # Save images
    for i in range(n_clusters):
        name = 'centroid_%d.jpg' % i
        centroid_path = os.path.join(centroids_dir, name)
        cv2.imwrite(centroid_path, 255*centroids[i,:,:])
    
    print('Centroids saved to directory:', centroids_dir)
    
    return labels
     

    

def cluster_all():
    """ 
    Calculates clusters for every video
    Saves label data to .csv
    
    """
    video_list = pd.read_csv('labels.csv', sep=';')
    video_list = video_list['Id']
    
    for video in video_list:
        labels = cluster(video)
        labels_path = os.path.join('centroids', video+'.txt')
        np.savetxt(labels_path, labels, fmt='%d')
    
      

    