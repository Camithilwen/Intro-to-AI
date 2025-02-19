import pandas as pd
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()

#Convert dataset to DataFrameap target values to specie
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

#Map target values to species names
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

#Display the first few rows and some data description
print(df.head())
print(df.describe())
print(df.isnull().sum())
print(df['species'].value_counts())

#Data preparation
x = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Standardize features (mean 0, variance 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Use K-Nearest-Neighbors algorithm for classification
knn= KNeighborsClassifier(n_neighbors=3)

#Train the model
knn.fit(X_train, y_train)

#Predict on test set
y_pred = knn.predict(X_test)

#Evaluate model performance and return reports
#Accuracy (total percentage of correct classification)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .2f}")

#Classifications (precision, recall, and F1 score per classification)
print("Classification Report:")
print(classification_report(y_test, y_pred))

#Confusion matrix (shows actual vs. prediction)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
