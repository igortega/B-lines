"""

Reshape to polar coordinates format

"""

from frame_extraction import order
import os
import cv2
import numpy as np
import pandas as pd



def max_slope(depth):
    """
    Returns 'dx' and 'dy' for slope of image boundaries given 'depth'

    """
    if depth == 14:
        dy = 280
        dx = 410
    if depth == 13:
        dy = 255
        dx = 370
    if depth == 10:
        dy = 236
        dx = 360
    if depth == 9:
        dy = 246
        dx = 370
    if depth == 8:
        dy = 230
        dx = 360
    if depth == 7:
        dy = 222
        dx = 332
        
    return dx, dy


    

def polar(img, depth, n_rows=50, n_cols=50):
    
    " Description "
    
    if len(img.shape) == 3:
        img = img[:,:,0]
    # print("Reshaping image...")
    
    # Get frame dimensions
    xlen, ylen = img.shape
    
    # Polar image array
    polar_img = np.zeros((n_rows, n_cols))
    
    # Cartesian coordinates array
    x = np.array(range(xlen))
    y = np.array(range(ylen))   
    Y, X = np.meshgrid(y,x)
    
    # Polar coordinates array
    rows, cols = np.zeros((xlen,ylen)), np.zeros((xlen,ylen))
    
    # Calculate image boundaries
    dx, dy = max_slope(depth)
    ymid = int(ylen/2)    
    m0 = dy/dx
    b0 = ymid - dy
    x0 = int(b0/m0)
    r0 = xlen + x0
    
    # Radius and slope arrays
    R = np.sqrt( (X+x0)**2 + (Y-ymid)**2 )
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
    
    # Assign values to polar image
    for i in range(n_rows):
        for j in range(n_cols):
            segment = img[(rows == i) & (cols == j)]
            polar_img[i,j] = segment.mean()
            
    # print("Image reshaped")
            
    return polar_img




def mask(n_rows=50, n_cols=50):
    """ 
    Creates and saves masks of polar images for every depth value.
    
    """
    xlen, ylen = 905, 632
    depth_values = [7, 8, 9, 10, 13, 14]
    
    if os.path.exists('mask') == False:
        os.makedirs('mask')
    
    for depth in depth_values:
        img = np.zeros((xlen,ylen))
        img = polar(img, depth, n_rows, n_cols)
        mask_img = np.isnan(img)
        dst_path = os.path.join('mask', 'depth%d.jpg' % depth)
        cv2.imwrite(dst_path, mask_img*255)



def polar_list(frames_dir, polar_dir, depth, n_rows=50, n_cols=50):
    """ 
    Reshapes all frames in 'frames_dir' into polar format
    Saves polar images to 'polar_dir' 
    Returns list of paths
    
    """
    
    # Create directory
    if os.path.exists(polar_dir) == False:
        os.makedirs(polar_dir)
        
    # Get ordered list of frame paths
    frame_paths = os.listdir(frames_dir)
    frame_paths.sort(key=order)
    
    print('Reshaping ', frames_dir, ' to polar...')
    for frame in frame_paths:
        src_path = os.path.join(frames_dir, frame)
        dst_path = os.path.join(polar_dir, frame)
        
        img = cv2.imread(src_path)
        polar_img = polar(img, depth, n_rows, n_cols)
        cv2.imwrite(dst_path, polar_img)




def polar_all():
    """ 
    (TAKES TOO LONG)
    Reshapes all frames from all videos to polar
    
    """
    labels = pd.read_csv('labels.csv', sep=';')
    
    if os.path.exists('polar') == False:
        os.makedirs('polar')
    
    for k in range(len(labels)):
        print('[', k, '/', len(labels), ']')
        src_dir = os.path.join('centroids', labels['Id'][k])
        dst_dir = os.path.join('polar', labels['Id'][k])
        polar_list(src_dir, dst_dir, depth=labels['Range'][k])