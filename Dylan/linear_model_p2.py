import pandas as pd
import numpy as np
import os
import scipy.stats as si
import matplotlib
import matplotlib.pyplot as plt
import datetime
import random
import math
from numpy.linalg import solve

def main():
    # read and pre-processing dataframe 
    filename = "huge_dataframe.csv"
    with open(os.path.join("..", "data", filename), "rb") as f:
        df = pd.read_csv(f,na_filter=False)
    
    #Training Steps
    #Linear prediction Model
    df_X = prune_data_X(df)
    X_x, X_y = divide_data_X(df_X)
    y_x,y_y = prune_divide_data_y(df)
    w_x = solve(X_x.T@X_x, X_x.T@y_x)
    w_y = solve(X_y.T@X_y, X_y.T@y_y)

    #Time Series Model
    

    print(w_x,w_y)
    # y,y_time_series = divide_data_y(df_X)


def onehotencoding(ori_df): # Used get_dummies from pandas
    column_names = list(ori_df.columns)
    # column_names = list(hudf)
    cat_cols = [s for s in column_names if ('type' in s)]
    changed_df = pd.get_dummies(ori_df, prefix=cat_cols, columns=cat_cols)
    return changed_df

def prune_data_X(df):
    column_names = list(df.columns)
    # drop redundant columns
    df = df.iloc[:,1:1002]
    for i in range(1,11):
        df = df.drop(['time step_-'+str(i*100)],axis=1)
    df = df.drop(['time step_0'],axis=1)
    drop_cols = [s for s in column_names if (('id' in s) or ('role' in s))]
    df = df.drop(drop_cols,axis=1)

    # add dummy variables
    df = onehotencoding(df)
    return df

def prune_divide_data_y(df):
    column_names = list(df.columns)
    # drop redundant columns
    df = df.iloc[:,1002:]
    
    #combine messed columns
    df1 = df.iloc[:,:90].values
    for i in range(df1.shape[0]):
        for j in range(df1.shape[1]):
            if df1[i,j]=='':
                df1[i,j]= 0
    df2 = df.iloc[:,90:].values
    for i in range(df2.shape[0]):
        for j in range(df2.shape[1]):
            if df2[i,j]=='':
                df2[i,j]= 0

    df1=df1.astype(np.float)
    df2=df2.astype(np.float)
    combined_y = df1+df2
    y_x = np.zeros((2308,1))
    y_x[:,0] = combined_y[:,1]
    y_y = np.zeros((2308,1))
    y_y[:,0] = combined_y[:,2]
    return y_x,y_y
    

def divide_data_X(df):
    column_names = list(df.columns)
    X_x_cols = [s for s in column_names if (('x' in s) or ('e_distance' in s))]
    X_y_cols = [s for s in column_names if ((('y' in s) or ('e_distance' in s)) and ('type' not in s))]
    
    X_x = df[X_x_cols].values
    X_y = df[X_y_cols].values
    return X_x, X_y

if __name__ == "__main__":
    main()