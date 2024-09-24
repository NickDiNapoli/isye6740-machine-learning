import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

mat_file = scipy.io.loadmat("./data/cs.mat")
#print(type(mat_file))
#print(mat_file['img'])
image = mat_file['img']
print(image.shape)

image_flattened = image.flatten()
#print(image_flattened.shape)

lin = 1300
I = 25*np.eye(lin)
#print(I)

A = np.random.randn(lin, image.shape[0]*image.shape[1])
#n = np.random.rand()
#print(A)

y = np.dot(A, image_flattened) + n

#plt.figure()
#plt.imshow(image)