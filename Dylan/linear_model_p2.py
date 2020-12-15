import pandas as pd
import numpy as np
import os
import scipy.stats as si
import matplotlib
import matplotlib.pyplot as plt
import datetime
import random
import math

def main():
    # read and pre-processing dataframe 
    filename = "huge_dataframe.csv"
    with open(os.path.join("..", "data", filename), "rb") as f:
        df = pd.read_csv(f,na_filter=False)
    
    df = preprocessing_data(df)
    X_x, X_y = divide_data(df)
    print(X_x, X_y)

def onehotencoding(ori_df): # Used get_dummies from pandas
    column_names = list(ori_df.columns)
    # column_names = list(hudf)
    cat_cols = [s for s in column_names if ('type' in s)]
    changed_df = pd.get_dummies(ori_df, prefix=cat_cols, columns=cat_cols)
    return changed_df

def preprocessing_data(df):
    column_names = list(df.columns)
    # drop redundant columns
    df = df.iloc[:,1:1002]
    for i in range(1,11):
        df = df.drop(['time step_-'+str(i*100)],axis=1)
    df = df.drop(['time step_0'],axis=1)
    id_cols = [s for s in column_names if (('id' in s) or ('role' in s))]
    df = df.drop(id_cols,axis=1)

    # add dummy variables
    df = onehotencoding(df)
    return df

def divide_data(df):
    column_names = list(df.columns)
    X_x_cols = [s for s in column_names if (('x' in s) or ('e_distance' in s))]
    X_y_cols = [s for s in column_names if ((('y' in s) or ('e_distance' in s)) and ('type' not in s))]
    
    X_x = df[X_x_cols]
    X_y = df[X_y_cols]

    return X_x, X_y

if __name__ == "__main__":
    main()