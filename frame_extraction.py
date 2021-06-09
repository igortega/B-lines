"""
Extract and crop frames

"""

import os
import cv2
import pandas as pd


def order(string):
    """ 
    Sorts frame list by natural order
    
    """
    string = string.split('_')[-1]
    n = string.split('.')[0]
    return int(n)


def crop(img):
    """ 
    Crops region of interest and returns
    
    """
    if len(img.shape) == 3:
        img = img[:, :, 0]
        
    img[50:90, 30:70] = 0  # Remove logo
    img[:, 585:] = 0  # Remove scale
    img = img[25:930, :]  # Crop upper and lower margins
    
    return img


def extract_and_crop(video_path, frames_dir, step=1):
    """ 
    Opens video in 'video_path'. Extracts and crops each frame. Saves frames to 'frames_dir'
    
    """
    
    frame_path_list = []
    
    # Creates frames destination directory
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
        
    # Opens video
    cap = cv2.VideoCapture(video_path)
    
    print('Extracting frames from', video_path)
    
    # Loops through each frame
    while cap.isOpened():
        frameId = int(cap.get(1))
        ret, frame = cap.read()
            
        # Breaks loop after last frame
        if not ret:
            print("\n Frames saved in", frames_dir)
            break
        
        # Saves frames
        if frameId % step == 0:  # Skip frames at step
            print(frameId, end=' ')
            frame_path = os.path.join(frames_dir, 'frame_%d.png' % frameId)
            frame_path_list.append(frame_path)
            frame = crop(frame)
            cv2.imwrite(frame_path, frame)  
        
    cap.release()
    

def extract_all(video_path_list, frames_dir):
    """ 
    Extracts and crops all videos in 'video_path_list'.
    Saves to 'frames_dir'
    
    """
    
    for video_path in video_path_list:
        video_name = os.path.splitext(os.path.split(video_path)[-1])[0]
        print(video_name)
        dst_dir = os.path.join(frames_dir, video_name)
        print(dst_dir)
        
        extract_and_crop(video_path=video_path, frames_dir=dst_dir)



