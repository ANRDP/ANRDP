from os import getcwd
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from numpy import asarray
from sklearn import metrics, preprocessing, tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

directory = getcwd()
filename = directory + "/heart.csv"
dataframe = pd.read_csv(filename)
print(dataframe.head)

numeric = dataframe[["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak", "HeartDisease"]]
categorical = dataframe[["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]]

categorical = dataframe.select_dtypes(include=object).columns
dataframe = pd.get_dummies(dataframe, columns=categorical, drop_first=True)

print(dataframe.head)

features = dataframe.drop("HeartDisease", axis=1)
target = dataframe["HeartDisease"]

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier(max_depth=3)

clf = clf.fit(features_train, target_train)

target_predict = clf.predict(features_test)

print("Accuracy: ", metrics.accuracy_score(target_test, target_predict))

fig = plt.figure(figsize=(20,20))
_ = tree.plot_tree(clf,
                   feature_names=features.columns,
                   # class_names=iris.target_names,
                   filled=True)
fig.savefig("decistion_tree.png")

#Cross Validation
number=10
scores = cross_val_score(clf, features, target, cv=number)

sum=0
for i in scores:
    sum += i

average = sum/number


