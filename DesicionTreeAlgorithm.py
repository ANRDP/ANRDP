from os import getcwd
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from numpy import asarray
from sklearn import metrics, preprocessing, tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import export_graphviz 
import pydotplus

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

clf = DecisionTreeClassifier(max_depth=5)

clf = clf.fit(features_train, target_train)

target_predict = clf.predict(features_test)

print("Accuracy: ", metrics.accuracy_score(target_test, target_predict))

# fig = plt.figure(figsize=(20,20))
# _ = tree.plot_tree(clf,
#                    feature_names=features.columns,
#                    filled=True)
# fig.savefig("decistion_tree.png")

dot_data = export_graphviz(clf, filled=True, rounded=True,
                                    feature_names=features.columns,
                                    out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png(directory+'/tree.png')

#Cross Validations
number=10
scores = cross_val_score(clf, features, target, cv=number, scoring="f1")

sum=0
for i in scores:
    sum += i

average = sum/number
print(f"F1 score after cross validation: {average}")

max_depth = []
fscore = []
for i in range(1,30):
    dtree = DecisionTreeClassifier(max_depth=i)
    dtree.fit(features_train, target_train)
    pred = dtree.predict(features_test)
    number=10
    scores = cross_val_score(clf, features, target, cv=number, scoring="f1")
    sum=0
    for j in scores:
        sum += j
    average = sum/number
    # acc_gini.append(metrics.f1_score(target_test, pred, average='binary'))
    fscore.append(average)
    max_depth.append(i)
    d = pd.DataFrame({'F1 score':pd.Series(fscore), 
    'max_depth':pd.Series(max_depth)})
# visualizing changes in parameters

plt.plot('max_depth','F1 score', data=d, label='F1 score')
plt.xlabel('max_depth')
plt.ylabel('F1 score')
plt.legend()
plt.show()

