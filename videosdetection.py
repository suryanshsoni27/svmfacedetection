import cv2
import numpy as np
import matplotlib.pyplot as plt

vid = cv2.VideoCapture('freeai/data/video.mp4')

haar = cv2.CascadeClassifier("freeai/data/haarcascade_frontalface_default.xml")


def face_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 10)
    return img


while(True):
    ret, frame = vid.read()
    if ret == False:
        break

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = face_detect(frame)
    cv2.imshow('object_detect', frame)
    #cv2.imshow('gray', gray)
    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
vid.release()
