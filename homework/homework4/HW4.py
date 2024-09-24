import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("../homework4/data/marriage.csv", header=None)
# data = np.load("../homework4/data/marriage.csv")

#print(df.head())
data = np.asarray(df)
data = data[:, :-1]
m, n = data.shape
print(m, n)
labels = data[:, -1]
print(labels.shape)

# split training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(data, labels, test_size= 0.2, shuffle=False)
#print(Xtrain)

Xtrain_var = np.var(Xtrain, axis=0)
print(Xtrain.shape, Xtrain_var.shape)
# ensure variance is not less that 1e-3
Xtrain_var = np.where(Xtrain_var < 1e-3, 1e-3, Xtrain_var)

NB = GaussianNB()
NB.fit(Xtrain, ytrain)
#NB.partial_fit(Xtrain, ytrain, np.unique(ytrain))
print(NB.predict(Xtest))






# Project data into 2D space
pca = PCA(n_components=2)
data_red = pca.fit(data).transform(data)

print(data_red)