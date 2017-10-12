# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:22:50 2017

@author: vikass
"""
# Import everything needed to edit/save/watch video clips
from moviepy.editor import *
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import cv2

vikas=[]
image = VideoFileClip('solidWhiteRight.mp4') 

count =0
for frames in image.iter_frames():
    gray = cv2.cvtColor(frames, cv2.COLOR_RGB2YUV)
    # Define a kernel size and apply Gaussian smoothing
    kernel_size =5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    low_threshold =100
    high_threshold = 200
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    rho = 9
    theta = np.pi/180
    threshold = 15
    min_line_length =50
    max_line_gap = 3
    line_image = np.copy(frames)*0 #creating a blank to draw lines on
    
    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
    
    # Iterate over the output "lines" and draw lines on the blank
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    
    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 
    vikas.append(cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) )
    
    count+=1
    print(count)
  
myclip = [ImageClip(m).set_duration(.1)
      for m in vikas]
concat_clip = concatenate_videoclips(myclip, method="compose")
concat_clip.write_videofile("test1.mp4", fps=30)