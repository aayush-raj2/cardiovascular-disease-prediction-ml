import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ---------------- LOAD DATA ----------------
df = pd.read_csv("cardio_train.csv", sep=';')


# ---------------- PREPROCESSING ----------------
df.drop("id", axis=1, inplace=True)

# convert age from days to years
df["age"] = df["age"] / 365


# ---------------- DATA ANALYSIS & VISUALIZATION ----------------

# target variable distribution
sns.countplot(x="cardio", data=df)
plt.title("Heart Disease Distribution")
plt.xlabel("Cardio (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# age vs heart disease
plt.figure(figsize=(8,5))
sns.boxplot(x="cardio", y="age", data=df)
plt.title("Age vs Heart Disease")
plt.xlabel("Cardio (0 = No, 1 = Yes)")
plt.ylabel("Age (years)")
plt.show()

# cholesterol vs heart disease
plt.figure(figsize=(8,5))
sns.countplot(x="cholesterol", hue="cardio", data=df)
plt.title("Cholesterol Level vs Heart Disease")
plt.xlabel("Cholesterol Level")
plt.ylabel("Count")
plt.show()

# blood pressure vs heart disease
plt.figure(figsize=(8,5))
sns.boxplot(x="cardio", y="ap_hi", data=df)
plt.title("Systolic Blood Pressure vs Heart Disease")
plt.xlabel("Cardio (0 = No, 1 = Yes)")
plt.ylabel("Systolic Blood Pressure")
plt.show()

# correlation matrix
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# ---------------- TRAIN TEST SPLIT ----------------
X = df.drop("cardio", axis=1)
y = df["cardio"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------------- MODEL TRAINING ----------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVM": SVC(kernel="rbf"),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=200)
}

print("\nModel Accuracies:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"{name}: {acc:.4f}")


# ---------------- FINAL MODEL ----------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

final_predictions = rf.predict(X_test)

print("\nFinal Model: Random Forest\n")
print(classification_report(y_test, final_predictions))
