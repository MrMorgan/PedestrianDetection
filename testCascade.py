#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:49:14 2019
demo of project if interested in car detection
Check following tutorial
https://github.com/duyetdev/opencv-car-detection
https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
"""

import cv2
from matplotlib import pyplot as plt
 
ncars=0
car_cascade = cv2.CascadeClassifier('./cars.xml')
img = cv2.imread('./car3.jpg', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect cars
cars = car_cascade.detectMultiScale(gray, 1.1, 1)

# Draw border
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    ncars = ncars + 1

# Show image
plt.figure(figsize=(10,20))
plt.imshow(img)