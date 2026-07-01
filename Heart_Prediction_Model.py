"""
===========================================================
Heart Disease Prediction using Machine Learning
===========================================================

Goal:
Predict whether a person is at risk of heart disease
using Logistic Regression and Decision Tree Classifier.

Dataset:
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Author: Your Name
===========================================================
"""

# ==========================
# Import Libraries
# ==========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

# ==========================
# Load Dataset
# ==========================

def load_data():
    df = pd.read_csv("heart.csv")      # Change filename if necessary
    return df


# ==========================
# Exploratory Data Analysis
# ==========================

def perform_eda(df):

    print("\nFirst Five Rows")
    print(df.head())

    print("\nDataset Shape")
    print(df.shape)

    print("\nDataset Information")
    print(df.info())

    print("\nStatistical Summary")
    print(df.describe())

    print("\nMissing Values")
    print(df.isnull().sum())

    # Target Distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x="target", data=df)
    plt.title("Target Distribution (0 = No Disease, 1 = Disease)")
    plt.show()

    # Gender Distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x="sex", data=df)
    plt.title("Gender Distribution")
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


# ==========================
# Train Logistic Regression
# ==========================

def logistic_model(X_train, X_test, y_train, y_test):

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nLogistic Regression Accuracy: {:.2f}%".format(accuracy * 100))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.show()

    # ROC Curve
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression")
    plt.legend()
    plt.show()

    return model


# ==========================
# Train Decision Tree
# ==========================

def decision_tree_model(X_train, X_test, y_train, y_test, feature_names):

    tree = DecisionTreeClassifier(random_state=42)

    tree.fit(X_train, y_train)

    predictions = tree.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nDecision Tree Accuracy: {:.2f}%".format(accuracy * 100))

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greens")
    plt.title("Confusion Matrix - Decision Tree")
    plt.show()

    # Feature Importance
    importance = pd.Series(
        tree.feature_importances_,
        index=feature_names
    )

    importance.sort_values().plot(
        kind="barh",
        figsize=(8,6)
    )

    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.show()

    return tree


# ==========================
# Main Function
# ==========================

def main():

    # Load Dataset
    df = load_data()

    # EDA
    perform_eda(df)

    # Split Data
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\nTraining Logistic Regression...")
    logistic_model(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nTraining Decision Tree...")
    decision_tree_model(
        X_train,
        X_test,
        y_train,
        y_test,
        X.columns
    )


# ==========================
# Run Program
# ==========================

if __name__ == "__main__":
    main()