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
ObesityRisk-ML-Classification/
│
├── [README.md](ca://s?q=Expand_README_structure)
│
├── [data](ca://s?q=Explain_data_folder_structure)/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── [notebooks](ca://s?q=Explain_notebooks_folder_structure)/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── 03_svm_model.ipynb
│   ├── 04_lda_qda.ipynb
│   └── 05_naive_bayes.ipynb
│
├── [models](ca://s?q=Explain_models_folder_structure)/
│   ├── logistic_regression_pipeline.pkl
│   ├── svm_pipeline.pkl
│   ├── lda_model.pkl
│   ├── qda_model.pkl
│   └── naive_bayes_model.pkl
│
├── [submissions](ca://s?q=Explain_submissions_folder_structure)/
│   ├── submission_logreg.csv
│   ├── submission_svm.csv
│   ├── submission_lda.csv
│   ├── submission_qda.csv
│   └── submission_nb.csv
│
├── [src](ca://s?q=Explain_src_folder_structure)/
│   ├── preprocessing.py
│   ├── train_logreg.py
│   ├── train_svm.py
│   ├── train_lda_qda.py
│   └── train_naive_bayes.py
│
├── [reports](ca://s?q=Explain_reports_folder_structure)/
│   ├── assignment5_report.pdf
│   └── figures/
│       ├── class_distribution.png
│       ├── confusion_matrix_logreg.png
│       ├── confusion_matrix_svm.png
│       └── feature_distributions.png
│
└── [requirements.txt](ca://s?q=Explain_requirements_file)

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
