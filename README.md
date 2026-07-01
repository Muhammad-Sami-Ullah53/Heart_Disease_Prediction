# ❤️ Heart Disease Prediction using Machine Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-4C72B0?style=for-the-badge)

</p>

<p align="center">
  <b>An End-to-End Machine Learning Project for Predicting Heart Disease Risk</b><br>
  Built using Python, Scikit-learn, Pandas, Matplotlib, and Seaborn.
</p>

---

# 📖 Overview

Heart disease is one of the leading causes of death worldwide. Early prediction can help healthcare professionals identify high-risk patients and take preventive measures.

This project develops a Machine Learning model capable of predicting whether a patient is likely to have heart disease based on various medical attributes.

The project includes:

- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data Preprocessing
- 🤖 Machine Learning Model Training
- 📈 Model Evaluation
- 📉 ROC Curve
- 🔥 Correlation Heatmap
- 📌 Feature Importance Analysis

---

# 🎯 Project Objective

The primary objective is to classify patients into:

- ❤️ Heart Disease Present
- 💚 No Heart Disease

using patient health information such as:

- Age
- Gender
- Chest Pain Type
- Blood Pressure
- Cholesterol
- Blood Sugar
- ECG Results
- Heart Rate
- Exercise Induced Angina
- Old Peak
- and more...

---

# 🧠 Machine Learning Models

The following supervised learning algorithms were implemented:

| Model | Purpose |
|--------|----------|
| Logistic Regression | Binary Classification |
| Decision Tree Classifier | Classification & Feature Importance |

---

# 🛠 Tech Stack

- 🐍 Python
- 📊 Pandas
- 🔢 NumPy
- 📈 Matplotlib
- 🎨 Seaborn
- 🤖 Scikit-Learn

---

# 📂 Dataset

**Dataset Used**

Heart Disease Dataset from Kaggle

https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

The dataset contains multiple clinical features commonly used for heart disease diagnosis.

---

# 📋 Features Used

| Feature | Description |
|----------|-------------|
| age | Age of patient |
| sex | Gender |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Cholesterol level |
| fbs | Fasting blood sugar |
| restecg | Resting ECG |
| thalach | Maximum heart rate |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | Slope of peak exercise |
| ca | Number of major vessels |
| thal | Thalassemia |
| target | Heart Disease (Target Variable) |

---

# ⚙️ Project Workflow

```text
               Dataset
                   │
                   ▼
          Data Preprocessing
                   │
                   ▼
      Exploratory Data Analysis
                   │
                   ▼
        Train-Test Split (80/20)
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
 Logistic Regression    Decision Tree
         │                   │
         └─────────┬─────────┘
                   ▼
          Model Evaluation
                   │
                   ▼
 Accuracy • ROC Curve • Feature Importance
```

---

# 📊 Exploratory Data Analysis

The project performs several visualization techniques including:

✅ Target Distribution

✅ Gender Distribution

✅ Correlation Heatmap

These visualizations help understand the dataset before model training.

---

# 📈 Model Evaluation

The trained models are evaluated using:

- Accuracy Score
- Confusion Matrix
- ROC Curve
- Feature Importance

---

# 🌳 Feature Importance

The Decision Tree model provides feature importance analysis showing which medical features contribute most to predicting heart disease.

---

# 📁 Project Structure

```
Heart-Disease-Prediction/
│
├── heart_disease_prediction.py
├── heart.csv
├── README.md
├── requirements.txt
└── images/
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Heart-Disease-Prediction.git
```

Go to project folder

```bash
cd Heart-Disease-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

```bash
python heart_disease_prediction.py
```

---

# 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install all packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

# 📷 Output

The program generates:

- 📊 Dataset Summary
- 📈 Statistical Analysis
- 🔥 Correlation Heatmap
- ❤️ Target Distribution
- 📉 ROC Curve
- 📌 Feature Importance Chart
- 📊 Confusion Matrix
- ✅ Model Accuracy

---

# 📈 Future Improvements

- Hyperparameter Tuning
- Random Forest Classifier
- XGBoost Classifier
- Support Vector Machine
- Model Deployment with Flask
- Streamlit Web Application
- Real-time Patient Prediction
- Cross Validation
- Model Saving with Pickle

---

# 🎯 Learning Outcomes

Through this project, you will understand:

- Data Analysis using Pandas
- Data Visualization
- Binary Classification
- Logistic Regression
- Decision Tree Algorithm
- Feature Engineering Basics
- Model Evaluation Metrics
- Machine Learning Workflow

---

# 🤝 Contributing

Contributions are always welcome!

If you'd like to improve this project:

- Fork the repository
- Create a new branch
- Make your changes
- Commit your work
- Submit a Pull Request

---

# ⭐ Support

If you found this project helpful,

⭐ Star this repository

🍴 Fork it

💡 Share it with others

---

# 👨‍💻 Author

**Sami Ullah**

AI & Machine Learning Enthusiast

Passionate about building intelligent solutions using Data Science, Machine Learning, and Artificial Intelligence.

---

<p align="center">

### ❤️ If you like this project, don't forget to give it a ⭐

**Happy Coding! 🚀**

</p>
