#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:00:07 2020

@author: doublewaters
"""

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('1.jpg',0)
img1 = cv2.pyrDown(img)
img2 = cv2.pyrDown(img1)
img3 = cv2.pyrDown(img2)

plt.subplot(221),plt.imshow(img,'gray'),plt.title("Orginal")
plt.subplot(222),plt.imshow(img1,'gray'),plt.title("Lower_reso1")
plt.subplot(223),plt.imshow(img2,'gray'),plt.title("Lower_reso2")
plt.subplot(224),plt.imshow(img3,'gray'),plt.title("Lower_reso3")
