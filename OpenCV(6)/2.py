#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:58:54 2020

@author: doublewaters
"""

import cv2
import matplotlib.pyplot as plt
img = cv2.imread('1.jpg',0)

img1 = cv2.pyrDown(img)
img2 = cv2.pyrDown(img1)

img3 = cv2.pyrUp(img2)
img4 = cv2.pyrUp(img3)

plt.subplot(131),plt.imshow(img,'gray'),plt.title("Orginal")
plt.subplot(132),plt.imshow(img2,'gray'),plt.title("Lower_reso")
plt.subplot(133),plt.imshow(img4,'gray'),plt.title("Higher_reso")
