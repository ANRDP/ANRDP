from os import getcwd
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from numpy import asarray
from sklearn import metrics, preprocessing, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import export_graphviz 
import pydotplus
from yellowbrick.classifier import ClassPredictionError


directory = getcwd()
filename = directory + "/heart.csv"
dataframe = pd.read_csv(filename)

numeric = dataframe[["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak", "HeartDisease"]]
categorical = dataframe[["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]]

categorical = dataframe.select_dtypes(include=object).columns
dataframe = pd.get_dummies(dataframe, columns=categorical, drop_first=True)


features = dataframe.drop("HeartDisease", axis=1)
target = dataframe["HeartDisease"]

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.1, random_state=1)
#X_train, X_test, y_train, y_test 

clf = RandomForestClassifier(n_estimators=100,max_depth=5)

clf = clf.fit(features_train, target_train)

target_predict = clf.predict(features_test) #y_pred

target_train_pred = clf.predict(features_train) #y_train_pred

rf_f1 = metrics.f1_score(target_test, target_predict)

print(f"F-SCORE: {rf_f1}")

visualizer = ClassPredictionError(clf)

# Evaluate the model on the test data
visualizer.fit(features_train, target_train)

visualizer.score(features_test, target_test)

visualizer.poof()


#Cross Validation
rf_xvalid_model = RandomForestClassifier(n_estimators=100,max_depth=5)

rf_xvalid_model_scores = cross_validate(rf_xvalid_model, features_train, target_train, scoring = ["accuracy", "precision", "recall", "f1"], cv = 10)
rf_xvalid_model_scores = pd.DataFrame(rf_xvalid_model_scores, index = range(1, 11))

print(f"After cross validation: \n {rf_xvalid_model_scores.mean()[2:]}")




