from os import getcwd
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_validate
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

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.1,
                                                                            random_state=1)

# AdaBoost
clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.6)
clf = clf.fit(features_train, target_train)
target_predict = clf.predict(features_test)
target_train_pred = clf.predict(features_train)
ab_f1 = metrics.f1_score(target_test, target_predict)

# F-score without cross validation
print(f"F-SCORE: {ab_f1}")

# Visualize Class Predicition Error for AdaBoost Classifier
visualizer = ClassPredictionError(clf)
# Fit training data
visualizer.fit(features_train, target_train)
# Evaluate the model on test data
visualizer.score(features_test, target_test)
visualizer.show()

# Cross Validation
ab_cv = AdaBoostClassifier(n_estimators=100, learning_rate=0.6)
ab_cv_scores = cross_validate(ab_cv, features_train, target_train, scoring=["accuracy", "precision", "recall", "f1"],
                              cv=10)
ab_cv_scores = pd.DataFrame(ab_cv_scores, index=range(1, 11))

print(f"Scores after cross validation:\n{ab_cv_scores.mean()[2:]}")
