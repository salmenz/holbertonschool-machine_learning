#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

data = np.load("data.npy")
labels = np.load("labels.npy")

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)
x = pca_data[:, 0]
y = pca_data[:, 1]
z = pca_data[:, 2]
fig = plt.figure(figsize=(11,11))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('PCA of Iris Dataset', fontsize=20)
ax.scatter(x, y, z, c=labels, cmap="plasma", edgecolor='face', s = 90)
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
plt.show()