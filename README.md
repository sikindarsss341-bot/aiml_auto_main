# AutoML Studio

An end-to-end Automated Machine Learning (AutoML) platform designed for tabular data, built with Scikit-Learn, FastAPI, and Streamlit.

Developed by Sikindar.

---

## Overview

AutoML Studio automates the full machine learning lifecycle from raw dataset ingestion to interactive inference:
- Automatic detection of problem type (Classification vs. Regression).
- Automated end-to-end data preprocessing and feature engineering.
- Multi-model tournament training across linear models, decision trees, ensemble methods, gradient boosters, and support vector machines.
- Performance evaluation and side-by-side metric comparison.
- Cross-validation based hyperparameter tuning for top candidate models.
- Interactive web interface for custom real-time inference with automated feature scaling.

---

## Key Features

### 1. Automated Preprocessing Pipeline
- Data Cleaning: Removes empty rows/columns, eliminates duplicate records, and drops irrelevant/high-cardinality ID columns.
- Missing Value Imputation: Numeric features imputed via mean/median strategies; categorical features imputed via mode/constant strategies.
- Categorical Encoding: Automatic label encoding and one-hot encoding with artifact tracking.
- Feature Selection: Variance threshold filtering and correlation-based redundancy pruning.
- Feature Scaling: StandardScaler normalization with mean and scale parameters tracked for downstream inference.
- Dataset Partitioning: Stratified train/test splits for classification; random splits for regression.

### 2. Supported Algorithms

#### Classification:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Gradient Boosting Classifier
- Naive Bayes (GaussianNB)
- XGBoost Classifier
- LightGBM Classifier
- Support Vector Classifier (SVC with CalibratedClassifierCV)

#### Regression:
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- Support Vector Regressor (SVR)

### 3. Model Evaluation & Comparison
- Classification Metrics: Accuracy, Weighted Precision, Weighted Recall, F1 Score, ROC-AUC, Confusion Matrix, and full Classification Report.
- Regression Metrics: R2 Score, Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Squared Error (MSE).
- Benchmark Leaderboard: Sorted model ranking with primary metric comparison bar charts.
- Feature Importance: Ranked feature drivers extracted per trained algorithm.

### 4. Hyperparameter Tuning
- Automated screening of top-performing candidate models.
- Hyperparameter grid search with cross-validation.
- Baseline vs. tuned CV score reporting and best parameter extraction.

### 5. Interactive Prediction Engine
- Dynamic form generation adapting to the dataset schema.
- Dropdown selections for categorical/discrete features and bounded inputs for continuous features.
- Real-time normalization using pipeline scaling parameters.
- Probability distribution outputs for classification predictions.

---

## Project Structure

```text
aiml_auto-main/
|-- .streamlit/
|   `-- config.toml              # Streamlit theme and server configuration
|-- .vscode/
|   `-- settings.json            # Editor settings
|-- app/
|   |-- comparison/
|   |   |-- model_comparison.py       # Classification comparison logic
|   |   `-- regression_comparison.py  # Regression comparison logic
|   |-- evaluation/
|   |   |-- model_evaluation.py       # Classification evaluation metrics
|   |   `-- regression_evaluation.py  # Regression evaluation metrics
|   |-- models/
|   |   |-- model_training.py         # Classification training routines
|   |   `-- regression_training.py    # Regression training routines
|   |-- modules/
|   |   `-- dataset_analyzer.py       # Dataset analysis and loading utilities
|   |-- preprocessing/
|   |   |-- clean_data.py             # Column cleaning and duplicate removal
|   |   |-- encoding.py               # Label and one-hot encoders
|   |   |-- feature_selection.py      # Variance and correlation selectors
|   |   |-- missing_values.py         # Missing value imputation
|   |   |-- sample.py                 # Sampling utilities
|   |   |-- scaling.py                # StandardScaler routines
|   |   `-- split_data.py             # Train-test split routines
|   `-- tuning/
|       |-- __init__.py
|       `-- hyperparameter_tuner.py   # Cross-validation tuning engine
|-- backend/
|   |-- __init__.py
|   |-- main.py                       # Core AutoML execution pipeline
|   |-- parameter_grids.py            # Search space definitions for tuning
|   `-- requirements.txt              # Backend dependencies
|-- .gitignore
|-- .python-version
|-- api_server.py                     # Standalone FastAPI API server entry point
|-- main.py                           # Streamlit dashboard application
|-- README.md
`-- runtime.txt
```

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sikindarsss341-bot/aiml_auto_main.git
cd aiml_auto_main
```

2. Install the required dependencies:
```bash
pip install -r backend/requirements.txt
pip install streamlit
```

### Running the Application

#### Streamlit Web Dashboard:
```bash
python -m streamlit run main.py
```
Open your browser and navigate to:
```
http://localhost:8501
```

#### FastAPI Backend API Server (Optional):
```bash
python api_server.py
```
API documentation available at:
```
http://localhost:8000/docs
```

---

## Usage Workflow

1. Open the Streamlit dashboard in your browser.
2. Upload your tabular dataset (.csv, .xlsx, .xls, or .json) via the sidebar or select an existing sample.
3. Confirm or change the Target Variable from the dropdown.
4. Click "Run AutoML Pipeline".
5. Explore results across the dashboard tabs:
   - Dashboard: High-level KPI metrics and podium winners.
   - Dataset: Preview, data types, missing values, and statistics.
   - Preprocessing: Detailed transformation and imputation audits.
   - Model Training: Feature importance charts and trained models.
   - Model Comparison: Leaderboard rankings and benchmark charts.
   - Evaluation: Confusion matrices and detailed classification reports.
   - Hyperparameter Tuning: CV score improvements and best parameters.
   - Prediction: Interactive inference using the winning model.

---

## License

This project is licensed under the MIT License.
