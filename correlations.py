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

#Cramer's V ChestPainType, RestingECG, ST_Slope
crosstab1 = np.array(pd.crosstab(dataframe["RestingECG"], dataframe["ChestPainType"], rownames= None, colnames=None))
crosstab2 = np.array(pd.crosstab(dataframe["RestingECG"], dataframe["ST_Slope"], rownames= None, colnames=None))
crosstab3 = np.array(pd.crosstab(dataframe["ChestPainType"], dataframe["ST_Slope"], rownames= None, colnames=None))

chi_1 = chi2_contingency(crosstab1)[0]
chi_2 = chi2_contingency(crosstab2)[0]
chi_3 = chi2_contingency(crosstab3)[0]

obs1 = np.sum(crosstab1)
obs2 = np.sum(crosstab2)
obs3 = np.sum(crosstab3)

minimum1 = min(crosstab1.shape)-1
minimum2 = min(crosstab2.shape)-1
minimum3 = min(crosstab3.shape)-1

result1 = chi_1/(obs1*minimum1)
result2 = chi_2/(obs2*minimum2)
result3 = chi_3/(obs3*minimum3)

print("RestingECG + ChestPainType", result1)
print("RestingECG + ST_Slope", result2)
print("ChestPainType + ST_Slope", result3)

correlation_mat = dataframe.corr(method = 'kendall')
correlation_mat2 = dataframe.corr(method = 'spearman')
correlation_mat3 = dataframe.corr()
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

fig.suptitle('Kendall\'s, Spearman\'s, Pearson\'s')
sns.heatmap(ax=axes[0],data=correlation_mat, annot = True)
sns.heatmap(ax=axes[1],data=correlation_mat2, annot = True)
sns.heatmap(ax=axes[2],data=correlation_mat3, annot = True)
plt.show()
