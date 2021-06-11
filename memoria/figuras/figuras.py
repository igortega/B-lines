# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 12:27:47 2021

@author: Ignacio
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

#%%

img1 = cv2.imread(r"C:\Principal\fisica-ucm\tfg\B-lines\memoria\figuras\frame1.png")[:,:,0]
img2 = cv2.imread(r"C:\Principal\fisica-ucm\tfg\B-lines\memoria\figuras\frame2.png")[:,:,0]
img3 = cv2.imread(r"C:\Principal\fisica-ucm\tfg\B-lines\memoria\figuras\frame3.png")[:,:,0]

img = np.concatenate((img1, img2, img3), axis=1) 
height, width = img.shape
l = np.round(width/6)

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(img, cmap='gray')
# ax.get_xaxis().set_visible(False)
ax.set_xticks([l, 3*l, 5*l])
ax.get_xaxis().set_ticklabels(['frame 1', 'frame 2', 'frame 3'])

# yticks = np.linspace(0, 13, height)
# cm = int(np.round(height/13))
# yticks_values = [yticks[k*cm] for k in range(13)]
# yt = [int(np.where(yticks==yticks_values[k])[0]) for k in range(13)]
# yt.append(height)

# yticks_locations = [yt[0], yt[2], yt[4], yt[7], yt[10], yt[13]]
yticks_locations = [0, 140, 280, 490, 700, 905]
yticks_labels = [0, 2, 4, 7, 10, 13]


ax.set_yticks(yticks_locations)
ax.get_yaxis().set_ticklabels(yticks_labels)
ax.set_ylabel('Depth [cm]')

cbar = fig.colorbar(im, ax=ax, shrink=0.6)
cbar.set_label('Brightness')

# fig.savefig('figuras/key_frames.png')

#%%

img = cv2.imread(r"C:\Principal\fisica-ucm\tfg\B-lines\memoria\figuras\alines.jpg")[:, :, 0]

fig, ax = plt.subplots(figsize=(5,10))
im = ax.imshow(img, cmap='gray')

yticks_locations = [0, 140, 280, 490, 700, 905]
yticks_labels = [0, 2, 4, 7, 10, 13]


ax.set_yticks(yticks_locations)
ax.get_yaxis().set_ticklabels(yticks_labels)
ax.set_ylabel('Depth [cm]')

ax.get_xaxis().set_visible(False)

fig.colorbar(im, ax=ax, shrink=0.55)

#%%
yticks_locations = [140, 490, 905]
yticks_labels = [2, 7, 13]

sector_example = np.load(r'C:\Principal\fisica-ucm\tfg\B-lines\memoria\figuras\sector_example.npy')
polar_example = np.load(r'C:\Principal\fisica-ucm\tfg\B-lines\memoria\figuras\polar_example.npy')

sector_image = cv2.imread(r'C:\Principal\fisica-ucm\tfg\B-lines\memoria\figuras\frame3.png')[:,:,0]
polar_image = cv2.imread(r'C:\Principal\fisica-ucm\tfg\B-lines\memoria\figuras\polar.png')[:,:,0]

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 7))

im1 = axs[0, 0].imshow(sector_example)
axs[0, 0].set_ylabel('Depth [px]')
axs[0, 0].set_title('Sector format')

im2 = axs[0, 1].imshow(polar_example)
axs[0, 1].set_title('Polar format')
axs[0, 1].set_ylabel('Radius [px]')
axs[0, 1].set_xlabel('Angle [px]')

im3 = axs[1, 0].imshow(sector_image, cmap='gray')
axs[1, 0].set_yticks(yticks_locations)
axs[1, 0].get_yaxis().set_ticklabels(yticks_labels)
axs[1, 0].set_ylabel('Depth [cm]')
axs[1, 0].get_xaxis().set_visible(False)

im4 = axs[1, 1].imshow(polar_image, cmap='gray')
yticks_locations = [8, 27, 49]
yticks_labels = [2, 7, 13]
axs[1, 1].set_yticks(yticks_locations)
axs[1, 1].get_yaxis().set_ticklabels(yticks_labels)
axs[1, 1].set_ylabel('Radius [cm]')
# axs[1, 1].get_xaxis().set_visible(False)
xticks_locations = [0, 25, 49]
xticks_labels = [-34, 0, 34]
axs[1, 1].set_xticks(xticks_locations)
axs[1, 1].get_xaxis().set_ticklabels(xticks_labels)
axs[1, 1].set_xlabel('Angle [º]')

#%%

yticks_locations = [0, 140, 280, 490, 700, 905]
yticks_labels = [0, 2, 4, 7, 10, 13]

img = cv2.imread(r'C:\Principal\fisica-ucm\tfg\B-lines\memoria\figuras\spleen.jpg')[:, :, 0]
fig, ax = plt.subplots(figsize=(5, 10))
im = ax.imshow(img, cmap='gray')

ax.set_yticks(yticks_locations)
ax.get_yaxis().set_ticklabels(yticks_labels)
ax.set_ylabel('Depth [cm]')

ax.get_xaxis().set_visible(False)

fig.colorbar(im, ax=ax, shrink=0.55)