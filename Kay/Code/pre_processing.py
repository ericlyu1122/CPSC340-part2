# standard Python imports
#%%
import os
import pickle
import gzip
import argparse
import random
import pandas as pd
import numpy as np

#%%

# load one single csv file
def load_single_dataset(filename):
    df = pd.read_csv(filename)
    return df

# Add distance column to different vehicle, calculate distance value, and order in ascending order. agent itself distance is 0
def append_distances(no_d_df):
    yes_d_df = no_d_df
    # a) Add in "distance" column to after each vehicle
    for d in range(0, 10):
        yes_d_df.insert(loc=(7 + d * 9), column='distance_x' + str(d), value=0)
        yes_d_df.insert(loc=(8 + d * 9), column='distance_y' + str(d), value=0)
        yes_d_df.insert(loc=(9 + d * 9), column='e_distance' + str(d), value=0) #euclidian distance
    # b) Find which loc has the "agent"
    # Note the columns from df.columns are: Index(['time step', ' id0', ' role0', ' type0', ' x0', ' y0', ' present0',
    # 'distance0', ' id1', ' role1', ' type1', ' x1', ' y1', ' present1',
    # 'distance1', ' id2', ' role2', ' type2', ' x2', ' y2', ' present2',
    # 'distance2', ' id3', ' role3', ' type3', ' x3', ' y3', ' present3',
    # Note the space before id1, role1, type1, and x1, y1, present!
    row0_series = yes_d_df[[' role0', ' role1', ' role2', ' role3', ' role4', ' role5', ' role6', ' role7', ' role8', ' role9']].iloc[0]
    row0_series = row0_series.fillna('others')
    row0_series = row0_series.replace(0, 'others')
    agent_col_name = row0_series[row0_series.str.contains('agent')].index[0] #Use "contains" because the value was actually [space]agent
    ai = int(agent_col_name.replace(' role', ''))
    yes_d_df[" x" + str(ai)] = yes_d_df[" x" + str(ai)].fillna(0)
    yes_d_df[" y" + str(ai)] = yes_d_df[" y" + str(ai)].fillna(0)

    # c) Calculate distances
    for d in range(0, 10):
        yes_d_df[" x" + str(d)] = yes_d_df[" x" + str(d)].fillna(0)
        yes_d_df[" y" + str(d)] = yes_d_df[" y" + str(d)].fillna(0)
        if (d != ai):  # If same as agent_col_name all distances are 0
            yes_d_df["distance_x" + str(d)] = abs(yes_d_df[" x" + str(d)] - yes_d_df[" x" + str(ai)]) # note had to add a space in front: " x" instead of "x" because the csv has space in front.
            yes_d_df["distance_y" + str(d)] = abs(yes_d_df[" y" + str(d)] - yes_d_df[" y" + str(ai)]) # note had to add a space in front: " x" instead of "x" because the csv has space in front.
            yes_d_df["e_distance" + str(d)] = (yes_d_df["distance_y" + str(d)] * yes_d_df["distance_y" + str(d)] + yes_d_df["distance_x" + str(d)] * yes_d_df["distance_x" + str(d)]).apply(np.sqrt)

    return yes_d_df

# End goal: for each X csv, merge lines (11 total, 10 past points, 1 current points) into one.
# Goal of method, take in dataframe, merge rows between start and end into one row, start inclusive, end exclusive.
def merge_to_one_row(multi_row_df, start, end):
    new_df = multi_row_df[start:end]
    columns = new_df.columns
    columns = [s+"_"+str(new_df.iloc[0,0]) for s in columns]
    merged_columns = columns.copy()

    for d in range(start+1, end):
        new_columns = columns.copy()
        new_columns = [s.replace("_" + str(new_df.iloc[0, 0]), "_" + str(new_df.iloc[d, 0])) for s in new_columns]
        # print("merged_columns: ", merged_columns)
        merged_columns=merged_columns+new_columns

    merged_row = new_df.values.flatten()

    new_df_ret = pd.DataFrame(data=merged_row[np.newaxis, : ], columns=merged_columns)

    return new_df_ret


def onehotencoding(ori_df): # Used get_dummies from pandas
    column_names = list(ori_df.columns)
    # column_names = list(hudf)
    cat_cols = [s for s in column_names if (('role' in s) or ('type' in s))]
    print(cat_cols)
    changed_df = pd.get_dummies(ori_df, prefix=cat_cols, columns=cat_cols)
    return changed_df



# ------------------ Main Frame Procedure -----------------------------------------------------
#%%

# Note when I was making the big grid, I encountered the following data issues:
# 1) NAN data for x, and y for 9th vehicel, this was handled by converting NAN to 0, see code above under "append_distances"
# 2) 0 values for "roles", I have changed the 0 values to 'Other', see code above under "append_distances"
# 3) the following y items had missing records (usually 29th or 30th row missing), I copied the values from the previous time stamps.
# The items mentioned in 3) above are: y_162, y_337, y_640, y_779, y_891, y_1017, y_1161, y_1667, y_2170, y_2171, 2172.
# I have included the modified csvs same folder under "New_Y_Files_After_fix_missing_Records\train". 

dflist = []

for c in range(0, 2308):
#for c in range(2169, 2308):
    cur_fname = ".\\Q2\\train\\X\\X_"+str(c)+".csv"
    cur_fname_y = ".\\Q2\\train\\y\\y_"+str(c)+".csv"
    print(cur_fname)
    df = load_single_dataset(cur_fname)
    print("Currently working on: ", cur_fname, "loaded: ", df.head(1))

    df = append_distances(df)

    df = merge_to_one_row(df, 0, 11)

    dfy = load_single_dataset(cur_fname_y)

    dfy = merge_to_one_row(dfy, 0, 30)
    # endy = len(dfy)
    # dfy = merge_to_one_row(dfy, 0, endy) #Didn't use these two lines because the huge matrix will need values.

    long_df = pd.concat([df, dfy], axis=1)

    dflist.append(long_df)

hudf = pd.concat(dflist)

hudf.to_csv('huge_dataframe.csv')

hudf_bu = hudf.copy()

#Deal with category vars, used pandas get_dummies
hudf_caty_dummy = onehotencoding(hudf)




# Check for missing values

# Visualize numerical distribution

# Split to x only and y only dataframes

