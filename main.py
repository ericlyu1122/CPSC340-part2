import os
import argparse
import time
import pickle
import time
# 3rd party libraries
import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime
from sklearn.tree import DecisionTreeClassifier
import utils
import sklearn.metrics
import matplotlib.pylab as plt
import os
from knn import KNN
import csv

def eucli_dist(x_coord, y_coord):
    return np.sqrt(x_coord ** 2 + y_coord ** 2)
    
def featureMatrix(X_pos, y_pos):
    # the speed of agent in direction of x at -950ms
    X_v1 = X_pos[1] - X_pos[0]
    # the speed of agent in direction of x at -50ms
    X_v10 = X_pos[10] - X_pos[9]
    # the acceleration in X axis from -950ms to -50ms by calculating a = (v_final - v_initial)/t
    X_acc = np.linalg.norm(X_v10- X_v1)/0.9
    
    # the speed of agent in direction of y at -950ms
    y_v1 = y_pos[1] - y_pos[0]
    # the speed of agent in direction of y at -50ms
    y_v10 = y_pos[10] - y_pos[9]
    # the acceleration in y axis from -950ms to -50ms by calculating a = (v_final - v_initial)/t
    y_acc = np.linalg.norm(y_v10 - y_v1)/0.9

    # calculate the average speed in the direction of x and y from -1000s to 0s
    X_avg_speed = (X_pos[10] - X_pos[0])
    y_avg_speed = (y_pos[10] - y_pos[0])
    
    # forming the feature matrix in order of
    # [initial position at -1000ms,  final position at 0ms,  initial speed,  final speed,  avg speed in last 1000ms,  acceleration in last 1000ms]
    return [X_pos[0],y_pos[0],  X_pos[10],y_pos[10],  X_v1,y_v1,  X_v10,y_v10,  X_avg_speed, y_avg_speed,  X_acc, y_acc]
    
def buildresult(yhat):
    # helper for manipulate the output
    # size of output
    n = yhat.shape[0]
    out=[]
    for i in range(n):
        for j in range(30):
            # get X of X_j
            arr=yhat[i][0]
            out.append(arr[j])
            # get y of y_j
            arr=yhat[i][1]
            out.append(arr[j])
    return np.array(out)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question


    if question == "2":
        ################################ GET TRAIN DATA ################################
        trainx=[]
        trainy=[]
        mode = "train"
        for i in range(2307):
            filename = mode + "/X/X_" + str(i) + ".csv"
            with open(os.path.join("..", "data", filename), "rb") as f:
                df = pd.read_csv(f)
            # retrieve the index of agent
            agent = -1
            for j in range(10):
                if df[' role' + str(j)].values[0]==' agent':
                    agent = j
            # get the position of the agent in last 1000ms
            trainX = df[' x' + str(agent)].values
            trainY = df[' y' + str(agent)].values
            # append one sample(one row) to the feature matrix
            trainx.append(featureMatrix(trainX,trainY))
            
            # retrieve values of (x,y) in next 3 sec from train
            Yfile = mode + "/y/y_" + str(i) + ".csv"
            with open(os.path.join("..", "data", Yfile), "rb") as f2:
                df_T = pd.read_csv(f2)
            # y of the next 3 sec should be consistent with the output in order of (X_pos, Y_pos)
            trainYx = df_T[' x'].values
            trainYy = df_T[' y'].values
            
            trainy.append([trainYx, trainYy])
        
        X = np.array(trainx)
        y = np.array(trainy)
        ################################ GET VAL DATA ################################
        valx=[]
        valy=[]
        mode = "val"
        for i in range(523):
            filename = mode + "/X/X_" + str(i) + ".csv"
            with open(os.path.join("..", "data", filename), "rb") as f:
                df = pd.read_csv(f)
            # retrieve the index of agent
            agent = -1
            for j in range(10):
                if df[' role' + str(j)].values[0]==' agent':
                    agent = j
            # get the position of the agent in last 1000ms
            valX = df[' x' + str(agent)].values
            valY = df[' y' + str(agent)].values
            # append one sample(one row) to the feature matrix
            valx.append(featureMatrix(valX,valY))
            
            # retrieve values of (x,y) in next 3 sec from train
            Yfile = mode + "/y/y_" + str(i) + ".csv"
            with open(os.path.join("..", "data", Yfile), "rb") as f2:
                df_T = pd.read_csv(f2)
            # y of the next 3 sec should be consistent with the output in order of (X_pos, Y_pos)
            valYx = df_T[' x'].values
            valYy = df_T[' y'].values
            
            valy.append([valYx, valYy])
        
        val_X = np.array(valx)
        val_y = np.array(valy)
        
        ################################ GET TEST DATA ################################
        X_test = []
        mode = "test"
        for i in range(20):
            filename = mode + "/X/X_" + str(i) + ".csv"
            with open(os.path.join("..", "data", filename), "rb") as f:
                df = pd.read_csv(f)
            # get the agent index
            agent = -1
            for j in range(10):
                if df[' role' + str(j)].values[0]==' agent':
                    agent = j
            
            testX = df[' x' + str(agent)].values
            testY = df[' y' + str(agent)].values
            
            X_test.append(featureMatrix(testX,testY))
        Xtest = np.array(X_test)
        
        ################################ FIT KNN WITH K = 1 ################################
        # for final used, train set + val set (2307 + 523)
        X_all = np.concatenate((X, val_X), axis=0)
        y_all = np.concatenate((y, val_y), axis=0)
        
        # for train used (X,y) //// for val used (val_X,val_y) //// for final stage used (X_all, y_all)
        # Used the KNN with k = 1
        k = 1
        print("We used K = ", k)
        model = KNN(1)
        model.fit(X_all,y_all)
        yhat = model.predict(Xtest)
        # transfer the result from yhat to the submission format
        output=buildresult(yhat)
        df_out = pd.DataFrame(output, columns=["loc"])
        df_out.to_csv("../data/result.csv")
            
 
