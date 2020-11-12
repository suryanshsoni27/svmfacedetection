from sklearn.decomposition import PCA
import pickle
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image

data = np.load('freeai/data/data_100_100.npz')

data.files
X = data['arr_0']
y = data['arr_1']
print(X.shape)
print(y.shape)

# eigne image
X1 = X - X.mean(axis=0)

pca = PCA(n_components=None, whiten=True, svd_solver='auto')
x_pca = pca.fit_transform(X1)

eigen_ratio = pca.explained_variance_ratio_
eigen_ratio_cm = np.cumsum(eigen_ratio)

# using elbow method consider numebbr of components is betwen 25-30
pca_50 = PCA(n_components=50, whiten=True, svd_solver='auto')
x_pca_50 = pca_50.fit_transform(X1)
