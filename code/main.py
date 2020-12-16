import os
import argparse
import time
import pickle
import time
# 3rd party libraries
import numpy as np
import pandas as pd
import datetime as datetime
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
from sklearn.tree import DecisionTreeClassifier
import utils
import linear_model
import sklearn.metrics
import pandas as pd
import matplotlib.pylab as plt
import os
from knn import KNN
from sklearn.metrics import mean_squared_error
import csv

def eucli_dist(x_coord, y_coord):
    return np.sqrt(x_coord ** 2 + y_coord ** 2)
def buildresult(yhat):
    out=[]
    iI=0
    while iI in range(20):
        for j in range(30):
            if iI==20:
                break
            
            arr=yhat[iI][0]
            out.append(arr[j])
            del arr
            arr=yhat[iI][1]
            out.append(arr[j])
            del arr
        iI=iI+1
    return np.array(out)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question


    if question == "2":
    
        trainx=[]
        trainy=[]
       
        ty=[]
        X_test = []
        mode = "train"
        for i in range(2307):
            filename = mode + "/X/X_" + str(i) + ".csv"
            with open(os.path.join("..", "data", filename), "rb") as f:
                df = pd.read_csv(f)
            
            agent_idx = -1
            for j in range(10):
                if df[' role' + str(j)].values[0]==' agent':
                    agent_idx = j
                    
            if agent_idx == -1:
                print("error")
            
            trainX = df[' x' + str(agent_idx)].values
            trainY = df[' y' + str(agent_idx)].values
            
            deltaX = trainX[1] - trainX[9]
            deltay = trainY[1] - trainY[9]
            trainx.append([deltaX, deltay])
            
            Yfile = mode + "/y/y_" + str(i) + ".csv"
            with open(os.path.join("..", "data", Yfile), "rb") as f2:
                df_T = pd.read_csv(f2)
            
            trainYx = df_T[' x'].values
            trainYy = df_T[' y'].values
            
            trainy.append([trainYx, trainYy])
        
        X = np.array(trainx)
        y = np.array(trainy, dtype=object)
        mode = "test"
        for i in range(20):
            filename = mode + "/X/X_" + str(i) + ".csv"
            with open(os.path.join("..", "data", filename), "rb") as f:
                df = pd.read_csv(f)
            
            agent_idx = -1
            for j in range(10):
                if df[' role' + str(j)].values[0]==' agent':
                    agent_idx = j
                    
            if agent_idx == -1:
                print("error")
            
            trainX = df[' x' + str(agent_idx)].values
            trainY = df[' y' + str(agent_idx)].values
            
            deltaX = trainX[1] - trainX[9]
            deltay = trainY[1] - trainY[9]
            X_test.append([deltaX, deltay])
            
        
        Xtest = np.array(X_test)
#        print(Xtest)
        model = KNN(1)
        model.fit(X,y)
        yhat = model.predict(Xtest)

        out=buildresult(yhat)
#        print(out)
        ######### acc ########
        trainx=[]
        trainy=[]
        ty=[]
        X_test = []
        mode = "train"
        for i in range(2307):
            filename = mode + "/X/X_" + str(i) + ".csv"
            with open(os.path.join("..", "data", filename), "rb") as f:
                df = pd.read_csv(f)
            
            agent_idx = -1
            for j in range(10):
                if df[' role' + str(j)].values[0]==' agent':
                    agent_idx = j
                    
            if agent_idx == -1:
                print("error")
            
            trainX = df[' x' + str(agent_idx)].values
            trainY = df[' y' + str(agent_idx)].values
            
            x_v1 = trainX[1] - trainX[0]
            x_v10 = trainX[10] - trainX[9]
            x_acc = np.linalg.norm(x_v1- x_v10)/0.1

            y_v1 = trainY[1] - trainY[0]
            y_v10 = trainY[10] - trainY[9]
            y_acc = np.linalg.norm(y_v1 - y_v10) / 0.1
            trainx.append([x_acc, y_acc])
            
            Yfile = mode + "/y/y_" + str(i) + ".csv"
            with open(os.path.join("..", "data", Yfile), "rb") as f2:
                df_T = pd.read_csv(f2)
            
            trainYx = df_T[' x'].values
            trainYy = df_T[' y'].values
            
            trainy.append([trainYx, trainYy])
        
        X = np.array(trainx)
        y = np.array(trainy, dtype=object)
        mode = "test"
        for i in range(20):
            filename = mode + "/X/X_" + str(i) + ".csv"
            with open(os.path.join("..", "data", filename), "rb") as f:
                df = pd.read_csv(f)
            
            agent_idx = -1
            for j in range(10):
                if df[' role' + str(j)].values[0]==' agent':
                    agent_idx = j
                    
            if agent_idx == -1:
                print("error")
            
            trainX = df[' x' + str(agent_idx)].values
            trainY = df[' y' + str(agent_idx)].values
            
            x_v1 = trainX[1] - trainX[0]
            x_v10 = trainX[10] - trainX[9]
            x_acc = np.linalg.norm(x_v1- x_v10)/0.1

            y_v1 = trainY[1] - trainY[0]
            y_v10 = trainY[10] - trainY[9]
            y_acc = np.linalg.norm(y_v1 - y_v10) / 0.1
           
            X_test.append([x_acc, y_acc])
            
        
        Xtest = np.array(X_test)
#        print(Xtest)
        model = KNN(1)
        model.fit(X,y)
        yhat = model.predict(Xtest)
#        print(yhat)
        out2=buildresult(yhat)
        ###########################################
        
        
        outf=np.concatenate((out.reshape((len(out),1)),out2.reshape((len(out2),1))), axis = 1)
#        outf=np.concatenate((outf,out3.reshape((len(out3),1))), axis = 1)
#        outf=np.concatenate((outf,out4.reshape((len(out4),1))), axis = 1)
        outf = np.mean(outf, axis =1 )
    
        df_out = pd.DataFrame(outf, columns=["location"])



        df_out.to_csv("../data/"+ "xaxis.csv")
            
            
        # Data set
#        data = pd.read_csv(os.path.join('..','data','phase2_training_data.csv'))
#        # put death on the first column for auto regression.
#
#        data['date'] = pd.to_datetime(data['date'],format='%m/%d/%Y')
#
#        ''' Choose start Date '''
#        startdate = datetime.datetime(2020,7,15)
#        print('startdate : ', startdate)
#        data['date'] -= startdate#start Date
#
#
#        data['date'] /= np.timedelta64(1, 'D')
#        data.replace(np.nan, -1, inplace=True)
#        data.replace(np.inf, -1, inplace=True)
#        data['date'].astype('int')
#        data=data[data['date']>=0]
#
#
#        ''' Features to choose from '''
#        X = data.loc[:,['country_id', 'deaths',
#                                      #12'cases',
#                                      'cases_14_100k',
#                                      'cases_100k'
#                                      ]]
#
#        ''' Choose from Countries for training '''
#        #X = X[(X['country_id']=='CA')|(X['country_id']=='SE')]
#        print(X.shape)
##        ''' Choose K '''
#        K = 35
#        mintest = 1000
#        ans= np.array([9504,9530,9541,9557,9585,9585,9585,9627,9654,9664,9699])
#        #for k in range(K):
#            # Fit weighted least-squares estimator
#        model = linear_model.MultiFeaturesAutoRegressor(K)
#        model.fit(X)
#        #    currtest = np.sqrt(sklearn.metrics.mean_squared_error(model.predict(X[X['country_id']=='CA'],11), ans))
#        #    print(k)
#        #    if currtest<=mintest:
#        #        mintest = currtest
#        #        print(mintest)
#        r = model.predict(X[X['country_id']=='CA'],5)
#        #print(np.sqrt(sklearn.metrics.mean_squared_error(r, ans)))
#        print(r)
#        for n in range(20):
#            filename = "test/X/X_" + str(n) + ".csv"
#            with open(os.path.join("..", "data", filename), "rb") as f:
#                df = pd.read_csv(f)
#            # filename = "val/y/y_" + str(n) + ".csv"
#            # with open(os.path.join("..", "data", filename), "rb") as f:
#            #     val = pd.read_csv(f).to_numpy()
#            # validation = val[:, 1]
#            # validation = np.column_stack((validation, val[:, 2]))
#            days = 10
#            K = 2
#            agent_position = 0
#            data_ca = np.zeros((11, 30))
#            for n in range(10):
#                if df.iloc[0, 6 * n + 2] == " agent":
#                    agent_position = n
#                data_ca[:, 2 * n] = df.iloc[:, 6 * n + 2 + 2].to_numpy()
#                data_ca[:, 2 * n + 1] = df.iloc[:, 6 * n + 2 + 3].to_numpy()
#                data_ca[:, 20 + n] = eucli_dist(df.iloc[:, 6 * n + 2 + 2].to_numpy(), df.iloc[:, 6 * n + 2 + 3].to_numpy())
#            D = data_ca
#
#            model = linear_model.MultiFeaturesAutoRegressor(K=K)
#            model.fit(D)
#            y_pred = model.predict(D,30)
#
#            for i in range(30):
#                for j in range(2):
#                    print(float(y_pred[:, 2*agent_position:2*agent_position+2][i][j]))
#    # print(mean_squared_error(validation, y_pred[:, 2*agent_position:2*agent_position+2], squared=False))
