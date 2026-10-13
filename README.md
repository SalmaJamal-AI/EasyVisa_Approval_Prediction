# EasyVisa Approval Prediction | Advanced Machine Learning

## Project Overview

This project applies supervised machine learning to predict whether an employment-based visa application will be **Certified or Denied**, using applicant and employer-related information.

The goal is to help identify applications that are more likely to be certified and support data-driven decision-making. The project explores multiple classification models, class-imbalance handling techniques, hyperparameter tuning, and model evaluation.

## Business Problem

The Office of Foreign Labor Certification (OFLC) processes employment-based visa applications. Identifying factors associated with certification can help stakeholders better understand application outcomes.

This project develops a binary classification solution using historical visa application data to predict `case_status`.

## Objectives

- Perform exploratory data analysis (EDA) to understand applicant and employer characteristics.
- Prepare and transform data for machine learning.
- Engineer relevant features, including company age and standardized prevailing wage.
- Compare multiple classification algorithms.
- Evaluate the effect of class-imbalance handling using SMOTE and undersampling.
- Tune promising models using GridSearchCV.
- Select a final model based on recall, precision, F1 score, and accuracy.
- Identify influential features associated with predicted visa certification outcomes.

## Dataset Overview

The dataset contains **25,480 records and 12 columns** before preprocessing.

The target variable is `case_status`, representing whether an application is Certified or Denied.

Key features include:

- Continent
- Education of employee
- Job experience
- Job training requirement
- Employer and job-related information
- Number of employees
- Year of establishment
- Prevailing wage
- Unit of wage
- Full-time position indicator
- Region of employment

## Project Workflow

### 1. Exploratory Data Analysis

- Examined dataset structure, variable types, and distributions.
- Investigated numerical and categorical features.
- Analyzed relationships between input variables and the target.
- Used visualizations to explore patterns associated with certification outcomes.

### 2. Data Preprocessing and Feature Engineering

- Checked for duplicate records and missing values.
- Examined outliers and addressed negative values in `no_of_employees`.
- Removed the unique identifier `case_id`.
- Created a `company_age` feature from `yr_of_estab`.
- Standardized prevailing wage values to an annual basis.
- Encoded categorical variables for model development.
- Split the data into training, validation, and test sets using stratification.

### 3. Handling Class Imbalance

Compared model performance across three data configurations:

- Original dataset
- Oversampled dataset using SMOTE
- Undersampled dataset

### 4. Model Building

Compared multiple classification algorithms, including:

- Decision Tree
- Random Forest
- Bagging Classifier
- AdaBoost
- XGBoost

### 5. Hyperparameter Tuning

Used `GridSearchCV` to tune selected models, including AdaBoost, Random Forest, and XGBoost, and compare their validation performance.

### 6. Model Selection

Selected the tuned XGBoost classifier as the final model based on its overall performance across accuracy, precision, recall, and F1 score.

## Evaluation Metrics

**Recall was prioritized** because missing an application that should be classified as Certified can be costly in the intended use case. Precision and F1 score were also considered to evaluate false positives and the balance between precision and recall.

The final model was evaluated on the held-out test set.

| Metric | Test Performance |
|---|---:|
| Accuracy | 73.55% |
| Precision | 76.49% |
| Recall | 87.19% |
| F1 Score | 81.49% |

These results are from the notebook's final test-set evaluation.

## Key Findings

- The tuned XGBoost model achieved an accuracy of **73.55%** and an F1 score of **81.49%** on the test set.
- Recall was **87.19%**, indicating that the model identified a substantial proportion of actual Certified cases.
- Feature-importance analysis highlighted employee education and prior job experience as influential features.
- Continent, employment region, and prevailing wage also contributed to the model's predictions.
- The experiments demonstrated the importance of evaluating precision and recall together rather than relying on accuracy alone.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- Jupyter Notebook / Google Colab

## Repository Structure

```text
EasyVisa_Approval_Prediction/
│
├── README.md
└── Project_Advanced_Machine_Learning_EasyVisa_SalmaJamal.ipynb
```

## How to Explore the Project

Open the notebook to review the complete workflow, from exploratory data analysis and preprocessing to model comparison, hyperparameter tuning, and final test-set evaluation.

**[View the Jupyter Notebook](Project_Advanced_Machine_Learning_EasyVisa_SalmaJamal.ipynb)**

The dataset is not included in this repository unless added separately. Dataset access and any environment-specific paths may need to be configured before rerunning the notebook.

## Limitations

- The model reflects patterns in the historical dataset and may not generalize to future applications or policy changes.
- Feature importance indicates predictive contribution, not causation.
- Predictions should not be treated as official visa decisions or as a substitute for legal or regulatory review.
- Further validation, fairness assessment, and monitoring would be needed before any real-world use.

## Author

**Salma Jamal**

Bioinformatics professional transitioning into AI/ML, with experience in scientific research, data analysis, and machine learning applications.

