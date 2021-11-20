from os import getcwd
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np

directory = getcwd()
filename = directory + "/heart.csv"
dataframe = pd.read_csv(filename)
print(dataframe.head)

# Dividing dataframe into two sub-dataframes, one for numerical value and one for categorical values
numeric = dataframe[["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak", "HeartDisease"]]
categorical = dataframe[["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]]

# Finding missing values
print(numeric.isnull().sum())
print(categorical.isnull().sum())

# Identifying outliers based on zscore
z = np.abs(stats.zscore(numeric))
print(z)

num_out = numeric[(z<3).all(axis = 1)]
outliers = np.where(z>3)

print("outliers")
print(outliers)

num_out.boxplot(column=['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak'], grid=False)
plt.show()

# Identifying outliers based on IQR
Q1 = np.percentile(numeric, 25, interpolation='midpoint')
Q3 = np.percentile(numeric, 75, interpolation='midpoint')
IQR = Q3 - Q1

outliers = numeric[((numeric < (Q1-1.5*IQR)) | (numeric > (Q3+1.5*IQR))).any(axis=1)]
num_out = numeric[~((numeric < (Q1-1.5*IQR)) | (numeric > (Q3+1.5*IQR))).any(axis=1)]

num_out.boxplot(column=['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak'], grid=False)
plt.show()

print("outliers")
print(outliers)


# Outliers of every numerical column - IQR
Q1_R = np.percentile(numeric['RestingBP'], 25, interpolation='midpoint')
Q3_R = np.percentile(numeric['RestingBP'], 75, interpolation='midpoint')
IQR_R = Q3_R - Q1_R

num_out = numeric[~((numeric['RestingBP'] < (Q1_R-1.5*IQR_R)) | (numeric['RestingBP'] > (Q3_R+1.5*IQR_R)))]

Q1_C = np.percentile(num_out['Cholesterol'], 25, interpolation='midpoint')
Q3_C = np.percentile(num_out['Cholesterol'], 75, interpolation='midpoint')
IQR_C = Q3_C - Q1_C

num_out = num_out[~((num_out['Cholesterol'] < (Q1_C-1.5*IQR_C)) | (num_out['Cholesterol'] > (Q3_C+1.5*IQR_C)))]

Q1_F = np.percentile(num_out['FastingBS'], 25, interpolation='midpoint')
Q3_F = np.percentile(num_out['FastingBS'], 75, interpolation='midpoint')
IQR_F = Q3_F - Q1_F

num_out = num_out[~((num_out['FastingBS'] < (Q1_F-1.5*IQR_F)) | (num_out['FastingBS'] > (Q3_F+1.5*IQR_F)))]

Q1_M = np.percentile(num_out['MaxHR'], 25, interpolation='midpoint')
Q3_M = np.percentile(num_out['MaxHR'], 75, interpolation='midpoint')
IQR_M = Q3_M - Q1_M

num_out = num_out[~((num_out['MaxHR'] < (Q1_M-1.5*IQR_M)) | (num_out['MaxHR'] > (Q3_M+1.5*IQR_M)))]

Q1_O = np.percentile(num_out['Oldpeak'], 25, interpolation='midpoint')
Q3_O = np.percentile(num_out['Oldpeak'], 75, interpolation='midpoint')
IQR_O = Q3_O - Q1_O

num_out = num_out[~((num_out['Oldpeak'] < (Q1_O-1.5*IQR_O)) | (num_out['Oldpeak'] > (Q3_O+1.5*IQR_O)))]

num_out.boxplot(column=['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak'], grid=False)
plt.show()


