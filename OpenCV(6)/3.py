#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:56:05 2020

@author: doublewaters
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

lower_reso = cv2.pyrDown(img)
higer_reso = cv2.pyrUp(lower_reso)

lp_img = cv2.subtract(img, higer_reso)

plt.figure(figsize=(10,10),dpi= 80)
plt.subplot(121)
plt.imshow(img)
plt.xlabel("Orginal")
plt.subplot(122)
plt.imshow(lp_img)
plt.xlabel("lp_img")
plt.tight_layout()
plt.show()

