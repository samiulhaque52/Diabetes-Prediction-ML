import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

#dataset loading
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
df = pd.read_csv(url, names=column_names)

#Dataset loading
df.head()
df.info()
df.describe()

#missing zeros
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zero:
  print(f"{col} has {df[df[col] == 0].shape[0]} zeros")

#Data preprocessing & cleaning
for col in cols_with_zero:
  median = df[col].median()
  df[col] = df[col].replace(0, median)

for col in cols_with_zero:
  print(f"{col} has {df[df[col] == 0].shape[0]} zeros")

#Axis setup
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

#Test and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state =72)

print("Training set size: ", X_train.shape)
print("Test set size: ", X_test.shape)

# Model training and accuracy checking
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LogisticRegression(max_iter=400)
model_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = model_scaled.predict(X_test_scaled)

#Output checking
from sklearn.metrics import accuracy_score
print("Accuracy after scaling (Standard Scaler):", accuracy_score(y_test, y_pred_scaled))
