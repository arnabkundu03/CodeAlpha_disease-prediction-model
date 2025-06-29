# 🧠 Disease Prediction Tool

A Machine Learning-based project that predicts the chances of:
- ❤️ **Heart Disease**
- 💉 **Diabetes**
- 🎗️ **Breast Cancer**

using multiple ML models like Logistic Regression, SVM, Random Forest, and XGBoost.

> 🔗 GitHub Repository: [Disease Prediction Tool](https://github.com/arnabkundu03/CodeAlpha_disease-prediction-model/tree/master/Disease-prediction-tool)

---

## 📌 Project Overview

This tool is built to assist in learning ML classification techniques and model evaluation.  
It trains and compares multiple models across three datasets, printing out accuracy and classification metrics for each.

---

## 🧾 Features

- 🔄 Data Preprocessing with `StandardScaler`
- 🤖 Classification Models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - XGBoost
- 📊 Performance Metrics:
  - Accuracy
  - Classification Report (Precision, Recall, F1-Score)

---

## 📁 Dataset Details

| Disease Type     | CSV File             | Target Column |
|------------------|----------------------|---------------|
| Heart Disease     | `heart.csv`          | `target`      |
| Diabetes          | `diabetes.csv`       | `Outcome`     |
| Breast Cancer     | `breast_cancer.csv`  | `target`      |

All datasets are included in this repository.

---

## 🛠️ How to Use

### 1. Clone this repository

```bash
git clone https://github.com/arnabkundu03/CodeAlpha_disease-prediction-model.git
cd CodeAlpha_disease-prediction-model/Disease-prediction-tool
```

### 2. Install dependencies

```pip install -r requirements.txt```

### 3. Run the script

```python disease_prediction.py```
You'll see printed model performance results for each disease dataset.

### 📦 Dependencies

Add the following to ```requirements.txt```

### 📌 Sample Output (CLI)

===== Heart Disease =====
Model: Logistic Regression
Accuracy: 0.8689
...

===== Diabetes =====
Model: Random Forest
Accuracy: 0.7967
...

===== Breast Cancer =====
Model: XGBoost
Accuracy: 0.9474
...

