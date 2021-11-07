from os import getcwd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

directory = getcwd()
filename = directory + "/heart.csv"
dataframe = pd.read_csv(filename)

genderDict = {'M': 0, 'F': 1} #Categorical Variable Converted into Numerical
yesNoDict = {'Y': 1, 'N': 0} #Categorical Variable Converted into Numerical
dataframe.Sex = [genderDict[item] for item in dataframe.Sex]
dataframe.ExerciseAngina = [yesNoDict[item] for item in dataframe.ExerciseAngina]


correlation_mat = dataframe.corr(method = 'kendall')
correlation_mat2 = dataframe.corr(method = 'spearman')
correlation_mat3 = dataframe.corr()
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

fig.suptitle('Kendall\'s, Spearman\'s, Pearson\'s')
sns.heatmap(ax=axes[0],data=correlation_mat, annot = True)
sns.heatmap(ax=axes[1],data=correlation_mat2, annot = True)
sns.heatmap(ax=axes[2],data=correlation_mat3, annot = True)
plt.show()
