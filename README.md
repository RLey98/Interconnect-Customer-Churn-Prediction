# 📡 Interconnect — Customer Churn Prediction

## 🧠 Project Overview

This project was developed as part of a data science and machine learning pipeline to help **Interconnect**, a telecommunications company, identify customers likely to **cancel their service** (“churn”).  
By predicting churn in advance, the company can take preventive actions (discounts, plan adjustments, personalized offers) to improve retention and reduce revenue loss.

---

## 🎯 Objective

Build and evaluate a **supervised classification model** capable of predicting customer churn based on their service type, billing method, tenure, and other behavioral features.

---

## 🗂️ Dataset

The dataset contains **7,043 customer records**, each with demographic, contract, and service-related information.

| Column                                                      | Description                                               |
| :---------------------------------------------------------- | :-------------------------------------------------------- |
| `customer_id`                                               | Unique customer identifier                                |
| `gender`                                                    | Male / Female                                             |
| `senior_citizen`                                            | Whether the customer is a senior citizen                  |
| `partner`, `dependents`                                     | Family information                                        |
| `tenure`                                                    | Number of months the customer has stayed with the company |
| `phone_service`, `multiple_lines`                           | Telephone services                                        |
| `internet_service`, `online_security`, `tech_support`, etc. | Internet and additional service types                     |
| `contract`                                                  | Type of contract (Month-to-month, One year, Two year)     |
| `payment_method`                                            | Payment method (electronic, mailed, etc.)                 |
| `monthly_charges`                                           | Monthly billing amount                                    |
| `total_charges`                                             | Total billed amount                                       |
| `service_canceled`                                          | Target variable — 1 if canceled, 0 otherwise              |

---

## 🪜 Project Workflow

### 1️⃣ Data Cleaning & Integration

- Removed duplicates and handled missing values in `TotalCharges`.
- Converted dates and tenure into a new feature: `customer_duration` (in months).
- Cast numerical types appropriately.
- Verified data consistency and structure.

### 2️⃣ Exploratory Data Analysis (EDA)

Explored key statistical properties and relationships:

- **Churn rate:** ~26% of all customers.
- **Contract type:**
  - Month-to-month contracts → 42.7% churn rate
  - One-year contracts → 11.2% churn rate
  - Two-year contracts → 2.8% churn rate
- **Internet service:**
  - Fiber optic → 41.9% churn
  - DSL → 18.9% churn
  - No internet → 7.4% churn
- Strong correlations:
  - `total_charges` ↔ `customer_duration` → 0.82
  - `service_canceled` negatively correlated with `customer_duration` → -0.37

### 3️⃣ Feature Engineering

- Encoded binary categorical features (`Yes/No`, `Male/Female`) into 0/1.
- Applied One-Hot Encoding to nominal categorical columns.
- Scaled numerical variables using `StandardScaler`.
- Addressed class imbalance (26% positive class) via **upsampling** in the training set.

### 4️⃣ Modeling

Trained and compared multiple classification models:

| Model                    | Type                         | Notes                                |
| :----------------------- | :--------------------------- | :----------------------------------- |
| `LogisticRegression`     | Linear baseline              | Quick benchmark                      |
| `RandomForestClassifier` | Ensemble bagging             | Handles nonlinearity                 |
| `CatBoostClassifier`     | Gradient boosting            | Handles categorical data efficiently |
| `XGBClassifier`          | Gradient boosting            | High accuracy, slower to train       |
| `LGBMClassifier`         | Gradient boosting (LightGBM) | Efficient and fast, great ROC AUC    |

---

## ⚙️ Hyperparameter Tuning

GridSearchCV was used to optimize:

- `n_estimators`
- `learning_rate`
- `max_depth`
- `num_leaves` (for LightGBM)
- `colsample_bytree`, `subsample`

Evaluation metric: **ROC-AUC** (main) + **Accuracy / F1-score**.

---

## 📊 Model Evaluation

| Model                | Accuracy (Test) | F1-score | ROC-AUC  |
| :------------------- | :-------------: | :------: | :------: |
| Logistic Regression  |      0.74       |   0.63   |   0.85   |
| Random Forest        |      0.79       |   0.67   |   0.88   |
| CatBoost             |      0.96       |   0.92   |   0.98   |
| XGBoost              |      0.97       |   0.95   |   0.99   |
| **LightGBM (Final)** |    **0.97**     | **0.95** | **0.99** |

### ✅ Final Model: LightGBM

Chosen for its balance between **performance** and **efficiency**.

- Trained in less time than CatBoost/XGBoost.
- Excellent interpretability with feature importances.
- ROC-AUC = **0.99**, Accuracy = **0.97**, showing excellent discrimination between churned and retained customers.

---

## 🔍 Feature Importance

Top predictive features according to LightGBM:

1. `contract_type_Month-to-month`
2. `internet_service_Fiber optic`
3. `customer_duration`
4. `payment_method_Electronic check`
5. `monthly_charges`

---

## 📈 Visual Insights

| Visualization                 | Description                                                     |
| :---------------------------- | :-------------------------------------------------------------- |
| **Churn by Contract Type**    | Month-to-month contracts drive the majority of cancellations.   |
| **Churn by Internet Service** | Fiber optic users cancel more often than DSL users.             |
| **Correlation Heatmap**       | Revealed strong negative relationship between tenure and churn. |
| **Class Distribution**        | Imbalanced target variable (5174 non-churn vs 1869 churn).      |

_(Visuals are available in the Jupyter Notebook.)_

---

## 🧩 Tech Stack

| Category        | Tools                                     |
| :-------------- | :---------------------------------------- |
| Language        | Python 3.10                               |
| Data Processing | Pandas, NumPy                             |
| Visualization   | Matplotlib, Seaborn                       |
| Modeling        | Scikit-learn, LightGBM, XGBoost, CatBoost |
| Optimization    | GridSearchCV                              |
| Evaluation      | ROC-AUC, Accuracy, F1-score               |

---

## 🚀 Results Summary

- The **LightGBM** model accurately identifies potential churners with ROC-AUC ≈ **0.99**.
- The company can leverage this model to **prioritize customer retention** efforts and minimize cancellations.
- The analysis revealed key churn drivers (contract type, internet service, billing method).

---

## 🧾 Conclusion

All project steps — data preparation, EDA, feature engineering, model training and evaluation — were successfully completed.
