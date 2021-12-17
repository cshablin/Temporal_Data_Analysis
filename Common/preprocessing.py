import os
import sys
import pandas as pd
import numpy as np

#load training dataset
df_train = pd.read_csv("data/WADI_14days_new.csv")

print(df_train.head())

print(df_train.columns)

empty_columns_to_remove = ["Row","2_LS_001_AL","2_LS_002_AL","2_P_001_STATUS","2_P_002_STATUS"]

print(len(df_train.columns))

"""

#dropping empty columns

"""

df_train = df_train.drop(empty_columns_to_remove, axis = 1)

print(len(df_train.columns))

"""

#removing columns with constant value

"""
constant_value_columns = []
for column in df_train.columns:
    if(df_train[column].nunique()==1):
        constant_value_columns.append(column)

df_train = df_train.loc[:, df_train.apply(pd.Series.nunique) != 1]

print(len(df_train.columns))


        


"""

#we are left with 95 columns - out of which two are identifiers - "Date", "Time"

"""

set_of_empty_vals = np.where(pd.isnull(df_train))
print(len(set_of_empty_vals[0]))

"""

# we have just 34 rows where we have empty values. I recommend dropping them. No need to impute for the values because 34 rows is negligible when compared to 784537 rows

"""
df_train = df_train.drop(set_of_empty_vals[0])


"""
################### Repeat process for test set #######################
"""
df_test = pd.read_csv("data/WADI_attackdataLABLE.csv")


print(df_test.head())

print(df_test.columns)

empty_columns_to_remove = ["Row ","2_LS_001_AL","2_LS_002_AL","2_P_001_STATUS","2_P_002_STATUS"]

print(len(df_test.columns))

"""

#dropping empty columns

"""

df_test = df_test.drop(empty_columns_to_remove, axis = 1)

print(len(df_test.columns))

"""

#removing columns with constant value (in a way that makes it compatible with the training set removed columns. NOTE : There are 3 
#columns that were removed in the train set that are not removed in the test set since they are not having constant values in the test set.
#Do you think we should do this for the test set first instead of leading with train? )

"""
df_test = df_test.drop(constant_value_columns, axis=1)

print(len(df_test.columns))


"""

#we are left with 95 columns - out of which two are identifiers - "Date", "Time"

"""

set_of_empty_vals = np.where(pd.isnull(df_test))
print(len(set_of_empty_vals[0]))

"""

# we have just 194 rows where we have empty values in the test set. we can safely remove them from our evaluation.

"""
df_test = df_test.drop(set_of_empty_vals[0])

df_train.to_csv("data/Cleaned_Trainset.csv")

df_test.to_csv("data/Cleaned_Testset.csv")