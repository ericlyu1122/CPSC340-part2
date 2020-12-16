import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils


class MultiFeaturesAutoRegressor:
    def __init__(self,K):
        self.K = K

    def fit(self,X):
        K=self.K
        print("K = ",K)

#        CL = X['country_id'].unique()

        #acc matrix for multi Countries
        y_auto = None
        x = None

#        print("features used : ", X.columns.values[1:])
#        print("countries used : ", CL)
        for cl in range(0,1):
            # Per country data and matrics
            country_dat = X
#            print(country_dat.shape)
            row, coln=country_dat.shape
            x_part=np.ones((row-K,1))
            for i in range(K-1):
                x_part=np.column_stack((x_part,country_dat[i+1:row-K+i+1,:]))

            # Add to Acc
            if y_auto is None:
                y_auto = country_dat[K:]
            else:
                y_auto = np.concatenate((y_auto,country_dat[K:]),axis=0)


            if x is None:
                x = x_part
            else:
                x = np.concatenate((x,x_part),axis=0)

        x = x.astype('float64')
        y_auto = y_auto.astype('float64')
        # solve for self w
        self.w = solve(x.T@x, x.T@y_auto)

    def predict(self, X, day):
        ''' YOUR CODE HERE '''
        K=self.K
        # test Data
        y_temp=X.values[:,1:]
        row, coln=y_temp.shape

        #starter for each prediction step
        add = np.ones((1,1))
        #result y_pred
        y_pred = np.zeros((day,coln))
        for i in range(day):
            #loop last K rows from y_temp into x
            temp=y_temp[row-K+i+1:,:]
            x=add
            for j in range(K-1):
               x=np.concatenate((x,[temp[j]]),axis=1)

            #result for each step
            result =x@self.w
            #push result to y_pred for return
            y_pred[i]=result
            #push result to y_temp for next predictions step
            y_temp = np.concatenate((y_temp,result),axis=0)
        return y_pred[:,0].T
