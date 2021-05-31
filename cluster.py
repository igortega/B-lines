"""

Functions for frame clustering and video segmentation

"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from frame_extraction import order
import numpy as np
import pandas as pd
import os
import cv2


def key_frames(frames_dir, n_clusters=3):
    """Looks for key frames of each cluster in a set of frames

    Parameters
    ----------
    frames_dir
    n_clusters

    Returns
    -------
    frame_ids
        list of ids of key frames (as many as n_clusters)
    """

    frame_filenames = os.listdir(frames_dir)
    frame_filenames.sort(key=order)

    # Read frames and load data
    # print('Loading data...')
    data = []
    for name in frame_filenames:
        frame_path = os.path.join(frames_dir, name)
        img = cv2.imread(frame_path)
        img = img[:, :, 0]  # get only one channel (black and white)
        img = cv2.resize(img, (200, 200))
        sample = img.reshape(-1) / 255  # suitable shape for clustering and normalize to (0,1)
        data.append(sample)

    data = np.array(data)

    # Calculate clusters
    # print('Calculating clusters...')
    kmeans = KMeans(n_clusters)
    kmeans.fit(data)

    labels = kmeans.labels_
    silhouettes = silhouette_samples(data, labels)

    # Locate key frames
    frame_ids = []
    for n in range(n_clusters):
        # Search for frame with maximum silhouette
        maximum = silhouettes[np.where(labels == n)].max()
        index, = np.where(silhouettes == maximum)
        frame_ids.append(index[0])

    return frame_ids


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
    if not os.path.exists(centroids_dir):
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


def key_frames_all(n_clusters=3):
    """Get key frames ids for every video and save to .csv

    Returns
    -------

    """
    main_frames_dir = 'frames'

    data = pd.read_csv('labels.csv', sep=';')

    key_frames_df = data.copy().drop(['B-lines', 'Range'], axis=1)
    for k in range(len(key_frames_df)):
        print('Video', k, 'out of', len(key_frames_df))
        frames_dir = os.path.join(main_frames_dir, key_frames_df.loc[k, 'Id'])
        key_frames_ids = key_frames(frames_dir, n_clusters)
        for j in range(n_clusters):
            key_frames_df.loc[k, 'Cluster%d' % j] = int(key_frames_ids[j])

    key_frames_df.to_csv('key_frames.csv', sep=';', index=None)
    return key_frames_df
