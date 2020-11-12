import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob


# classifier defined here
haar = cv2.CascadeClassifier(
    'freeai/model/haarcascade_frontalface_default.xml')
female = []
male = []

female = glob('freeai/data/Female/*.jpg')
male = glob('freeai/data/Male/*.jpg')


path = female[0]
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faces = haar.detectMultiScale(gray, 1.5, 5)


for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(img, cmap='gray')


crop_img = img[y:y+h, x:x+h]
plt.imshow(crop_img)


cv2.imwrite('f_01.png', crop_img)

# apply crop to all images


def extract_image(path, gender, i):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.5, 5)
    for x, y, w, h in faces:
        roi = img[y:y+h, x:x+w]
        if gender == 'male':
            cv2.imwrite(
                'freeai/data/cropMale/{}_{}.png'.format(gender, i), roi)
        else:
            cv2.imwrite(
                'freeai/data/cropFemale/{}_{}.png'.format(gender, i), roi)


extract_image(female[0], 'female', 1)
# for i, path in enumerate(female):
#     try:
#         extract_image(path, 'female', i)
#         print('INFO: {}/{} processed successfully'.format(i, len(female)))

#     except:
#         print('INFO: {}/{} not processed successfully'.format(i, len(female)))


# for i, path in enumerate(male):
#     try:
#         extract_image(path, 'male', i)
#         print('INFO: {}/{} processed successfully'.format(i, len(male)))

#     except:
#         print('INFO: {}/{} not processed successfully'.format(i, len(male)))
