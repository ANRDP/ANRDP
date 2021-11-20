from os import getcwd
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

directory = getcwd()
filename = directory + "/heart.csv"
dataframe = pd.read_csv(filename)
print(dataframe.head)


# Getting rid of outliers

Q1_R = np.percentile(dataframe['RestingBP'], 25, interpolation='midpoint')
Q3_R = np.percentile(dataframe['RestingBP'], 75, interpolation='midpoint')
IQR_R = Q3_R - Q1_R

clean = dataframe[~((dataframe['RestingBP'] < (Q1_R-1.5*IQR_R)) | (dataframe['RestingBP'] > (Q3_R+1.5*IQR_R)))]

Q1_C = np.percentile(clean['Cholesterol'], 25, interpolation='midpoint')
Q3_C = np.percentile(clean['Cholesterol'], 75, interpolation='midpoint')
IQR_C = Q3_C - Q1_C

clean = clean[~((clean['Cholesterol'] < (Q1_C-1.5*IQR_C)) | (clean['Cholesterol'] > (Q3_C+1.5*IQR_C)))]

Q1_F = np.percentile(clean['FastingBS'], 25, interpolation='midpoint')
Q3_F = np.percentile(clean['FastingBS'], 75, interpolation='midpoint')
IQR_F = Q3_F - Q1_F

clean = clean[~((clean['FastingBS'] < (Q1_F-1.5*IQR_F)) | (clean['FastingBS'] > (Q3_F+1.5*IQR_F)))]

Q1_M = np.percentile(clean['MaxHR'], 25, interpolation='midpoint')
Q3_M = np.percentile(clean['MaxHR'], 75, interpolation='midpoint')
IQR_M = Q3_M - Q1_M

clean = clean[~((clean['MaxHR'] < (Q1_M-1.5*IQR_M)) | (clean['MaxHR'] > (Q3_M+1.5*IQR_M)))]

Q1_O = np.percentile(clean['Oldpeak'], 25, interpolation='midpoint')
Q3_O = np.percentile(clean['Oldpeak'], 75, interpolation='midpoint')
IQR_O = Q3_O - Q1_O

clean = clean[~((clean['Oldpeak'] < (Q1_O-1.5*IQR_O)) | (clean['Oldpeak'] > (Q3_O+1.5*IQR_O)))]

# clean.boxplot(column=['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak'], grid=False)
# plt.show()

print(clean)