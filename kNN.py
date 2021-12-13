from os import getcwd

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

directory = getcwd()
filename = directory + "/heart.csv"
dataframe = pd.read_csv(filename)
#print(dataframe)
#dataframe.info()


# Standardization
numerical = dataframe.select_dtypes(exclude=object).drop(columns='HeartDisease')
for col in numerical:
    dataframe[col] = StandardScaler().fit_transform(dataframe[[col]])
    print(dataframe[col])

'''
# Normalization of numerical data
numerical = dataframe.select_dtypes(exclude=object).columns
for col in numerical:
    dataframe[col] = MinMaxScaler().fit_transform(dataframe[[col]])
    print(dataframe[col])
'''

# One hot encoding on categorical data
categorical = dataframe.select_dtypes(include=object).columns
dataframe = pd.get_dummies(dataframe, columns=categorical, drop_first=True)

# HeartDisease - target variable
X = dataframe.drop(columns='HeartDisease')
y = dataframe['HeartDisease']

'''
# Splitting data into training and testing data
points = zeros(20)
maxes = zeros(20)
for i in range(1, 100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)    # random_state=42
    f1_scores = []

    for k in range(1, 20):
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)

    max_score = max(f1_scores)
    max_index = f1_scores.index(max_score)
'''

## standardized 0.3 15 ~90/91 | 0.2 9 ~91/92 | 0.4 7 ~89/90 | 0.1 15 ~93-95
## normalized (MIN MAX) 0.1 7 ~ 93 | 0.2 11 ~92 | 0.3 7 90 | 0.4 5 90

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)    # random_state=42

'''
# Creating KNN model
knn_model = KNeighborsClassifier(n_neighbors=15)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
'''


f1_scores = []
f1_plot = []
for k in range(1, 20):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)
    print(k-1, ": ", f1)

max_score = max(f1_scores)
max_index = f1_scores.index(max_score)
print("Max score: ", max_score, " k: ",  max_index)

nazwawykresu = plt.plot(range(1, 20), f1_scores)
plt.xlabel("K - number of neighbours")
plt.ylabel("Value")
plt.xticks(np.arange(0, 20, 1))
plt.show()



'''
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("F1 score: ", round(f1, 4))
print("Accuracy score: ", round(accuracy, 4))
'''