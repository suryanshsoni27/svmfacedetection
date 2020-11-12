import numpy as np
import pandas as pd
import pickle
from PIL import Image
import cv2

df = pickle.load(open('freeai/data/dataframe_images_100_100.pickle', 'rb'))
print(df.head())
print(df.info())

print(df.isnull().sum())

# remove the missing values
df.dropna(axis=0, inplace=True)

# split data into two parts
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values


Xnorm = X/X.max()

print(Xnorm.shape)

y_norm = np.where(y == 'female', 1, 0)
print(y_norm)
np.savez('freeai/data/data_100_100', Xnorm, y_norm)

# save x and y in numpy zip
