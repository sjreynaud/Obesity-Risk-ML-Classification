**Obesity Risk Classification — Assignment 5**
*Overview*
This repository contains the full machine‑learning workflow developed for Assignment 5: Build and Evaluate Classification Models. The project uses the Multi‑Class Prediction of Obesity Risk dataset from Kaggle to classify individuals into obesity‑risk categories based on lifestyle and biometric features.

**Supervised learning models were implemented:**

  - Multinomial Logistic Regression
  - Quadratic Discriminant Analysis (QDA)
  - Naïve Bayes (GaussianNB)
  - Support Vector Machine (SVM)

Each model includes preprocessing, training, evaluation, visualization, and Kaggle‑ready submission files.

**Repository Structure**

Code

├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│
├── notebooks/
│   ├── Assignment5_ObesityRisk.ipynb
│
├── submissions/
│   ├── submission_logreg.csv
│   ├── submission_svm.csv
│   ├── submission_qda.csv
│   ├── submission_nb.csv
│
├── README.md


**Methods**
  **Preprocessing**
  
    - Numeric features scaled using StandardScaler
    - Categorical features encoded using OneHotEncoder
    - Combined using ColumnTransformer
    - Pipelines built for reproducibility and consistency across models

**Models**

   - **Logistic Regression:** Multinomial classification with LBFGS optimizer
   - **QDA:** Nonlinear boundaries with class‑specific covariance matrices
   - **Naïve Bayes:** Gaussian likelihoods for each feature per class
   - **SVM:** RBF kernel for nonlinear separation

**Evaluation**

Each model was evaluated using:

  - Accuracy
  - Confusion matrix
  - Classification report (precision, recall, F1‑score)
  - Visual diagnostics including:
    - Confusion matrix heatmaps
    - Classification report heatmaps
    - PCA decision‑boundary projections
    - Feature importance and coefficient maps
    - Support‑vector analysis (SVM)
    - Class means and covariance diagnostics (LDA/QDA)
    - Feature mean/variance maps (Naïve Bayes)

**Kaggle Submissions**

Four submission files were generated:
  - submission_logreg.csv
  - submission_svm.csv
  - submission_qda.csv
  - submission_nb.csv

Each file follows the required Kaggle format and was successfully submitted to the competition.
