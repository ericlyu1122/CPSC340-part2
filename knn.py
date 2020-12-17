"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y
        # print(X.shape,y.shape)

    def predict(self, Xtest):
        X = self.X
        y = self.y
        t = Xtest.shape[0]

        k = self.k
        # Compute cosine_distance distances between X and Xtest
        dist2 = self.cosine_distance(X1=X, X2=Xtest)
        # size of yhat should be (test_size,2) while 2 means all (x,y) of x_0 to x_14
        yhat=[]

        if k ==1:
            for i in range(t):
                # sort the distances to other points
                inds = np.argsort(dist2[:,i])
                yhat.append(y[inds[0]])
        else:
            for i in range(t):
                # sort the distances to other points
                inds = np.argsort(dist2[:,i])
                tmp = y[inds[:k]]
                # ensure the size of each elememt is 30 
                for j in range(t):
                    if len(tmp[j,0])!=30:
                        np.insert(tmp[j,0],float,np.mean(tmp[j,0]) )
                
                    if len(tmp[j,1])!=30:
                        np.insert(tmp[j,1],float, np.mean(tmp[j,1]))
                xmean = np.mean(tmp[:,0])
                x_mean = np.mean(xmean)
                ymean = np.mean(tmp[:,1])
                y_mean = np.mean(ymean)
                yhat.append([x_mean,y_mean])

        return np.array(yhat)



    def cosine_distance(self,X1, X2):
        return 1 - cosine_similarity(X1, X2, dense_output=True)
