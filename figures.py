import matplotlib.pyplot as plt
import numpy as np
from polar import max_slope, polar
import os
import cv2

if os.path.exists('figures') == False:
    os.makedirs('figures')

def polar_format(n_rows=10, n_cols=10):
    # Get frame dimensions
    xlen, ylen = 905, 632

    # Polar image array
    polar_img = np.zeros((n_rows, n_cols))
    
    # Cartesian coordinates array
    x = np.array(range(xlen))
    y = np.array(range(ylen))   
    Y, X = np.meshgrid(y,x)
    
    # Polar coordinates array
    rows, cols = np.zeros((xlen,ylen)), np.zeros((xlen,ylen))
    
    # Calculate image boundaries
    dx, dy = max_slope(10)
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
    
    sections = rows + cols
    plt.imshow(sections)
    plt.savefig('figures/polar_original.png')
    
    for i in range(n_rows):
        for j in range(n_cols):
            segment = sections[(rows == i) & (cols == j)]
            polar_img[i,j] = segment.mean()
            
    plt.imshow(polar_img)
    plt.show()
    plt.savefig('figures/polar_new.png')
    
    return sections, polar_img

def polar_example():
    img = cv2.imread('polar/video_9/centroid_0.jpg')
    plt.imshow(img)
    plt.savefig('figures/polar.png')
    
polar_format()
polar_example()

