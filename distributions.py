from os import getcwd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

directory = getcwd()
filename = directory + "/heart.csv"
dataframe = pd.read_csv(filename)
print(dataframe)

# Summary statistics
print(dataframe.describe())


# DISTRIBUTIONS #
sns.set(style="darkgrid")

# Age
sns.histplot(data=dataframe["Age"], kde=True)
plt.title("Distribution of Age variable")
plt.show()

# Sex
print(dataframe["Sex"].value_counts())
sns.countplot(data=dataframe, x="Sex")
plt.title("Distribution of Sex variable")
plt.show()

# Chest Pain Type
print(dataframe["ChestPainType"].value_counts())
sns.countplot(data=dataframe, x="ChestPainType")
plt.title("Distribution ChestPainType variable")
plt.show()

# Resting Blood Pressure
sns.histplot(data=dataframe["RestingBP"], binwidth=20, kde=True)
plt.title("Distribution RestingBP variable")
plt.show()

# Cholesterol
sns.histplot(data=dataframe["Cholesterol"], kde=True)
plt.title("Distribution Cholesterol variable")
plt.show()

# Fasting Blood Sugar
print(dataframe["FastingBS"].value_counts())
sns.countplot(data=dataframe, x="FastingBS")
plt.title("Distribution FastingBS variable")
plt.show()

# Resting ElectroCardioGram
print(dataframe["RestingECG"].value_counts())
sns.countplot(data=dataframe, x="RestingECG")
plt.title("Distribution RestingECG variable")
plt.show()

# Max Heart Rate
sns.histplot(data=dataframe["MaxHR"], kde=True)
plt.title("Distribution MaxHR variable")
plt.show()

# ExerciseAngina
print(dataframe["ExerciseAngina"].value_counts())
sns.countplot(data=dataframe, x="ExerciseAngina")
plt.title("Distribution ExerciseAngina variable")
plt.show()

# Oldpeak
print(dataframe["Oldpeak"].describe())
sns.histplot(data=dataframe["Oldpeak"])
plt.title("Distribution Oldpeak variable")
plt.show()

# The slope of the peak exercise ST segment
print(dataframe["ST_Slope"].value_counts())
sns.countplot(data=dataframe, x="ST_Slope")
plt.title("Distribution ST_Slope variable")
plt.show()

# HeartDisease
print(dataframe["HeartDisease"].describe())
print(dataframe["HeartDisease"].value_counts())
sns.countplot(data=dataframe, x="HeartDisease")
plt.title("Distribution HeartDisease variable")
plt.show()



