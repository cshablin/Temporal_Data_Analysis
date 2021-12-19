import os
import sys
import pandas as pd
import numpy as np

#load training dataset
df_train = pd.read_csv("../Data/WADI_14days_new.csv")

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

Train const columns:
['1_LS_001_AL', '1_LS_002_AL', '1_MV_002_STATUS', '1_MV_003_STATUS', '1_P_002_STATUS', '1_P_004_STATUS', '1_P_006_STATUS', '2_MCV_007_CO', '2_MV_001_STATUS', '2_MV_002_STATUS', '2_MV_004_STATUS', '2_MV_005_STATUS', '2_MV_009_STATUS', '2_P_004_STATUS', '2_PIC_003_SP', '2_SV_101_STATUS', '2_SV_201_STATUS', '2_SV_301_STATUS', '2_SV_401_STATUS', '2_SV_501_STATUS', '2_SV_601_STATUS', '3_LS_001_AL', '3_MV_001_STATUS', '3_MV_002_STATUS', '3_MV_003_STATUS', '3_P_001_STATUS', '3_P_002_STATUS', '3_P_003_STATUS', '3_P_004_STATUS', 'PLANT_START_STOP_LOG']

Test const columns: 
['1_LS_001_AL', '1_LS_002_AL', '1_P_002_STATUS', '1_P_004_STATUS', '2_MV_001_STATUS', '2_MV_002_STATUS', '2_MV_004_STATUS', '2_MV_005_STATUS', '2_MV_009_STATUS', '2_P_004_STATUS', '2_SV_101_STATUS', '2_SV_201_STATUS', '2_SV_301_STATUS', '2_SV_401_STATUS', '2_SV_501_STATUS', '2_SV_601_STATUS', '3_AIT_001_PV', '3_AIT_002_PV', '3_LS_001_AL', '3_MV_001_STATUS', '3_MV_002_STATUS', '3_MV_003_STATUS', '3_P_001_STATUS', '3_P_002_STATUS', '3_P_003_STATUS', '3_P_004_STATUS', 'PLANT_START_STOP_LOG']


In Train not in Test constant columns:
1_MV_002_STATUS, 1_MV_003_STATUS, 1_P_006_STATUS, 2_MCV_007_CO,  2_PIC_003_SP

"""
in_train_not_in_test_const_col = ['1_MV_002_STATUS', '1_MV_003_STATUS', '1_P_006_STATUS', '2_MCV_007_CO',  '2_PIC_003_SP']
constant_value_columns = []
for column in df_train.columns:
    if(df_train[column].nunique()==1):
        constant_value_columns.append(column)

constant_value_columns = list(set(constant_value_columns) - set(in_train_not_in_test_const_col))

# df_train = df_train.loc[:, df_train.apply(pd.Series.nunique) != 1]
df_train = df_train.drop(constant_value_columns, axis=1)

print(len(df_train.columns))


        


"""

#we are left with 100 columns - out of which two are identifiers - "Date", "Time"

"""

set_of_empty_vals = np.where(pd.isnull(df_train))
print(len(set_of_empty_vals[0]))

"""

# we have just 34 rows where we have empty values. I recommend dropping them. No need to impute for the values because 34 rows is negligible when compared to 784537 rows
for example rows 61703~61712 missing value in same column '2B_AIT_004_PV' which makes it hard to impute 
"""
df_train = df_train.drop(set_of_empty_vals[0])


"""
################### Repeat process for test set #######################
"""
df_test = pd.read_csv("../Data/WADI_attackdataLABLE.csv")


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

#we are left with 100 columns (+1 with label, the last one) - out of which two are identifiers - "Date", "Time"

"""

set_of_empty_vals = np.where(pd.isnull(df_test))
print(len(set_of_empty_vals[0]))

"""

# we have just 2 (172801, 172802) rows (all empty) in the test set. we can safely remove them from our evaluation.

"""
df_test = df_test.drop(set_of_empty_vals[0])

df_train.to_csv("../Data/Cleaned_Trainset.csv")

df_test.to_csv("../Data/Cleaned_Testset.csv")