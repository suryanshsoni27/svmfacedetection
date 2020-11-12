import cv2
import numpy as np
import matplotlib.pyplot as plt


# step1 read image
img = cv2.imread("freeai/data/male_000281.jpg")
# step2 convet intp gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# apply haar cascade
haar = cv2.CascadeClassifier('freeai/data/haarcascade_frontalface_default.xml')

faces = haar.detectMultiScale(gray, 1.3, 5)
print(faces)


cv2.rectangle(img, (154, 94), (154+261, 94+261), (0, 255, 0), 10)
# plt.imshow(img)
# plt.show()

cv2.imshow('object detect', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# crop the face
sliceo = img[94:94+261, 154:154+261]
plt.imshow(sliceo)
plt.show()
