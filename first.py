import cv2
import numpy as np
import pandas as pd
import sklearn
from PIL import Image
import matplotlib.pyplot as plt
imgcv2 = cv2.imread("freeai/data/34921.jpg")
imgpil = Image.open("freeai/data/34921.jpg")
img = plt.imread("freeai/data/34921.jpg")

# plt.imshow(img)
# plt.show()

print(imgcv2.shape)
print(imgcv2)
print(img)
r = img[:, :, 0]  # red
g = img[:, :, 1]  # green
b = img[:, :, 2]  # blue

print(r, g, b)
print(r.shape, g.shape, b.shape)
# plt.subplot(1, 3, 1)
# plt.imshow(r, cmap='gray')

# plt.subplot(1, 3, 2)
# plt.imshow(g, cmap='gray')

# plt.subplot(1, 3, 3)
# plt.imshow(b, cmap='gray')
# plt.show()

# gray = cv2.cvtColor(imgcv2, cv2.COLOR_BGR2GRAY)

# # 10*10 array

# arr = np.arange(0, 100, 1)
# print(arr)
# arr1 = arr.reshape((10, 10))
# print(arr)


# plt.imshow(arr1)
# plt.show()


# arr2 = np.random.randint(0, 150, (10, 10))
# print(arr2)
# plt.imshow(arr2, cmap='gray')
# plt.show()


img_parr = cv2.imread("freeai/data/test.jpg")
print(img_parr.shape)
gray = cv2.cvtColor(img_parr, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()

print(gray[0:10, 0:10])
slice = gray[0:10, 0:10]
plt.imshow(slice, cmap='gray')
plt.show()


# shrink the image

# img_pree = cv2.resize(gray, (500, 400), cv2.INTER_AREA)

# plt.imshow(img_pree)
# plt.show()
img_en = cv2.resize(gray, (2000, 1600), cv2.INTER_CUBIC)
plt.imshow(img_en)
plt.show()
