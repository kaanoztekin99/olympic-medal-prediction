# Olympic Medal Prediction using Data Mining and Machine Learning

This repository contains the implementation of our course project for **DT085A – Data Mining and Machine Learning** at **Mid Sweden University**.

The goal of this project is to analyze Olympic athlete data and build machine learning models that can predict whether an athlete will win a medal based on their physical characteristics and participation context.

---

## Project Overview

The Olympic Games represent the highest level of athletic performance, but predicting success is challenging due to the many interacting factors involved.

Using a dataset containing **70,000 Olympic athlete records and 15 features**, we investigate whether machine learning can identify patterns that distinguish medal-winning athletes from non-medalists.

Since only a small percentage of athletes win medals, this project also addresses an **imbalanced classification problem**.

Our objective is not only to predict medal outcomes but also to understand the factors that contribute to athletic success.

---

## Dataset

Dataset used in this project:

**Elite Athlete Olympic Records**

Source:
https://www.kaggle.com/datasets/eshummalik/global-olympic-athletes-performance-dataset

The dataset contains information about:

- Athlete age
- Height
- Weight
- Team
- Season
- Sport
- Medal outcome

---

## Methodology

The project follows four main stages:

### 1. Data Preprocessing
- Handling missing values
- Converting medal types into a **binary target (Medal vs No Medal)**
- Normalizing numerical features (Age, Height, Weight)
- Encoding categorical variables
- Removing irrelevant attributes

### 2. Exploratory Data Analysis
- Distribution analysis using histograms and boxplots
- Correlation matrix analysis
- Investigation of class imbalance
- Feature relationship analysis

### 3. Data Mining Techniques

The following machine learning models will be applied:

- Decision Tree
- Random Forest
- k-Nearest Neighbors (k-NN)
- Logistic Regression
- XGBoost

Additionally, clustering and association rule mining will be explored to identify patterns in athlete performance.

### 4. Model Evaluation

Due to class imbalance, evaluation will focus on:

- **F1 Score**
- **ROC-AUC**
- **Confusion Matrix**

Cross-validation will also be used to ensure model generalization.

---

## Handling Class Imbalance

Since medalists represent a minority of athletes, the dataset is highly imbalanced.

To address this issue we will use:

- **SMOTE (Synthetic Minority Over-sampling Technique)**
- Stratified sampling
- Cross-validation

These techniques help the model learn the characteristics of medal-winning athletes more effectively.

---


---

## Project Goals

- Predict Olympic medal outcomes
- Understand how physical attributes influence success
- Compare interpretable and high-performance machine learning models
- Investigate sport-specific patterns in athlete performance

---

## Course Information

Course: **DT085A – Data Mining and Machine Learning**  
University: **Mid Sweden University**

---

## License

This repository is created for academic purposes.
## 👥 Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/kaanoztekin99">
        <img src="https://avatars.githubusercontent.com/kaanoztekin99" width="80" alt="Kaan Tekin Öztekin"/><br />
        <sub><b>Kaan Tekin Öztekin</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/USERNAME_KEVIN">
        <img src="https://avatars.githubusercontent.com/USERNAME_KEVIN" width="80" alt="Kevin Rasmusson"/><br />
        <sub><b>Kevin Rasmusson</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/USERNAME_PETER">
        <img src="https://avatars.githubusercontent.com/USERNAME_PETER" width="80" alt="Peter Zsoldos"/><br />
        <sub><b>Peter Zsoldos</b></sub>
      </a>
    </td>
  </tr>
</table>
