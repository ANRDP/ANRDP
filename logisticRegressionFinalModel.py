import os
from os import getcwd

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from numpy import asarray
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score

directory = getcwd()
filename = directory + "/heart.csv"
dataframe = pd.read_csv(filename)
print(dataframe.head)

numeric = dataframe[["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak", "HeartDisease"]]
categor = dataframe[["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]]

pd.crosstab(dataframe.Sex,dataframe.HeartDisease).plot(kind='bar')
plt.title('Sex vs Heart Disease')
plt.xlabel('Sex')
plt.ylabel('Heart Disease')
plt.savefig('sex_hd')

pd.crosstab(dataframe.ChestPainType,dataframe.HeartDisease).plot(kind='bar')
plt.title('Chest Pain Type vs Heart Disease')
plt.xlabel('Chest Pain Type')
plt.ylabel('Heart Disease')
plt.savefig('chestpain_hd')

pd.crosstab(dataframe.RestingECG,dataframe.HeartDisease).plot(kind='bar')
plt.title('Resting ECG vs Heart Disease')
plt.xlabel('Resting ECG')
plt.ylabel('Heart Disease')
plt.savefig('resting_hd')

pd.crosstab(dataframe.ExerciseAngina,dataframe.HeartDisease).plot(kind='bar')
plt.title('Exercise Angina vs Heart Disease')
plt.xlabel('Exercise Angina')
plt.ylabel('Heart Disease')
plt.savefig('exer_hd')

pd.crosstab(dataframe.ST_Slope,dataframe.HeartDisease).plot(kind='bar')
plt.title('ST Slope vs Heart Disease')
plt.xlabel('ST Slope')
plt.ylabel('Heart Disease')
plt.savefig('st_hd')

categorical = dataframe.select_dtypes(include=object).columns
dataframe = pd.get_dummies(dataframe, columns=categorical, drop_first=True)

features = dataframe.drop("HeartDisease", axis=1)
target = dataframe["HeartDisease"]

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2)
column_names = features_train.columns

LogReg = LogisticRegression(max_iter=100, solver='lbfgs',penalty = 'l2',C=1)

LogReg.fit(features_train, target_train)

#Cross Validation
number = 10
scores = cross_val_score(LogReg, features_train, target_train, cv=number, scoring="f1")
sum=0
for i in scores:
    sum += i
average = sum/number
print(scores)
print("Average f1 score: ", average)

target_pred = LogReg.predict(features_test)

confusion_m = metrics.confusion_matrix(target_test, target_pred)
print(confusion_m)
print("Accuracy: ", metrics.accuracy_score(target_test, target_pred))

precision = metrics.precision_score(target_test, target_pred)
recall = metrics.recall_score(target_test, target_pred)
print("Precision: ", precision)
print("Recall: ", recall)

f_score = 2*(precision*recall)/(precision+recall)
print("F-score: ", f_score)

class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(confusion_m), annot=True, cmap="RdYlBu" ,fmt='g')
ax.xaxis.set_label_position("bottom")
plt.title('Confusion matrix', y=1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# ROC curve - true positive rate against the false positive rate
target_pred_probab = LogReg.predict_proba(features_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(target_test, target_pred_probab)
auc = metrics.roc_auc_score(target_test, target_pred_probab)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()





