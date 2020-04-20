#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:15:34 2020

@author: doublewaters
"""

import cv2
import numpy as np,sys
from matplotlib import pyplot as plt

A = cv2.imread('apple.png')
B = cv2.imread('orange.png')

A = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)
B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)

# generate Gaussian pyramid for A
G = A.copy()
print(G)
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)
lpA = [gpA[5]]
for i in range(6,0,-1):
    print(i)
    GE = cv2.pyrUp(gpA[i])
    GE=cv2.resize(GE,gpA[i - 1].shape[-2::-1])
    L = cv2.subtract(gpA[i-1],GE)
    print(L.shape)
    lpA.append(L)
# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(6,0,-1):
    print(i)
    GE = cv2.pyrUp(gpB[i])
    GE = cv2.resize(GE, gpB[i - 1].shape[-2::-1])
    L = cv2.subtract(gpB[i-1],GE)
    print(L.shape)
    lpB.append(L)
# Now add left and right halves of images in each level
LS = []
lpAc=[]
for i in range(len(lpA)):
    b=cv2.resize(lpA[i],lpB[i].shape[-2::-1])
    print(b.shape)
    lpAc.append(b)
print(len(lpAc))
print(len(lpB))
j=0
for i in zip(lpAc,lpB):
    print(i)
    print('ss')
    la,lb=i
    print(la)
    print(lb)
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
    j=j+1
    print(j)
    LS.append(ls)
ls_ = LS[0]
for i in range(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_= cv2.resize(ls_, LS[i].shape[-2::-1])
    ls_ = cv2.add(ls_, LS[i])
# image with direct connecting each half
B= cv2.resize(B, A.shape[-2::-1])
real = np.hstack((A[:,:cols//2],B[:,cols//2:]))
# cv2.imwrite('Pyramid_blending2.jpg',ls_)
# cv2.imwrite('Direct_blending.jpg',real)

plt.subplot(221),plt.imshow(A),plt.title("Apple")
plt.subplot(222),plt.imshow(B),plt.title("Orange")
plt.subplot(223),plt.imshow(ls_),plt.title("Pyramid_blending")
plt.subplot(224),plt.imshow(real),plt.title("Direct_blending")
plt.show()
