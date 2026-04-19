# Customer Churn Prediction

A machine learning project that predicts customer churn using Random Forest classification. This repository includes both a training pipeline and an interactive Streamlit web application for real-time predictions.

## Overview

This project implements a complete ML pipeline for predicting whether a customer will churn (leave the service) based on demographic, account, and service usage data. The solution handles class imbalance, performs automated preprocessing, and provides an intuitive web interface for making predictions.

## Features

- **Random Forest Classification**: Robust ensemble learning with optimized hyperparameters
- **Class Imbalance Handling**: Automatic computation of class weights for balanced training
- **Automated Preprocessing**: Label encoding for categorical variables and standard scaling for numerical features
- **Interactive Web Application**: Streamlit-based UI with organized input sections
- **Model Persistence**: Save and load trained models using joblib
- **Performance Analytics**: Feature importance analysis and comprehensive evaluation metrics

## Dataset

The project uses `churn.csv` containing customer information with the following features:

**Customer Demographics:**
- `Gender`: Male/Female
- `Age`: Customer age in years
- `Partner`: Whether the customer has a partner (Yes/No)
- `Dependents`: Whether the customer has dependents (Yes/No)

**Account Information:**
- `Tenure`: Number of months with the company
- `Contract Type`: Month-to-month, One year, Two year
- `Payment Method`: Electronic check, Mailed check, Bank transfer, Credit card

**Services:**
- `Internet Service`: DSL, Fiber optic, or No internet service
- `Phone Service`: Yes/No
- `Multiple Lines`: Yes/No
- `TV`: Yes/No
- `Streaming`: Yes/No

**Financial:**
- `Monthly Charges`: Monthly amount charged to customer
- `Total Charges`: Total amount charged to customer

**Target Variable:**
- `Churn`: Whether the customer churned (Yes/No)

## Installation

1. Clone the repository:
```bash
2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:
```bash
pip install -r requirements.txt
## Usage

### Training the Model

Run the training script to train the model, view performance metrics, and save artifacts:

```bash
python train_model.py
This will:
- Load and preprocess the data
- Train a Random Forest classifier with class balancing
- Display accuracy, classification report, and confusion matrix
- Show top 5 most important features
- Save model artifacts to the `models/` directory

### Running the Web Application

Launch the interactive Streamlit application:

```bash
streamlit run app.py
The web interface features:
- **Two-column layout**: Personal Details (left) and Account Details (right)
- Real-time churn prediction based on user inputs
- Model accuracy display in the sidebar
- Interactive input fields for all customer features

## Model Technical Details

- **Algorithm**: Random Forest Classifier
  - `n_estimators=100`
  - `max_depth=10`
  - `min_samples_split=20`
  - `min_samples_leaf=10`
  - `class_weight`: Balanced (computed automatically)

- **Preprocessing**:
  - Label Encoding for categorical features
  - StandardScaler for numerical features

- **Training Configuration**:
  - Train/Test split: 80/20 with stratification
  - Random state: 42 (for reproducibility)
  - Class balancing using `sklearn.utils.class_weight`

- **Evaluation Metrics**:
  - Accuracy Score
  - Classification Report (Precision, Recall, F1-score)
  - Confusion Matrix
  - Feature Importance Analysis

## File Structure

```
churn-prediction/
├── train_model.py       # Training script with model export
├── app.py              # Streamlit web application
├── churn.csv           # Dataset file
├── requirements.txt    # Python dependencies
├── models/            # Directory for saved model artifacts (created automatically)
└── README.md          # This file
## Requirements

- Python 3.7+
- streamlit
- pandas
- scikit-learn
- matplotlib
- joblib

See `requirements.txt` for specific package versions.

## Performance

The model displays accuracy metrics in the Streamlit sidebar and provides detailed classification reports including precision, recall, and F1-score for churn prediction. Feature importance analysis helps identify the most influential factors in customer churn.

## Future Improvements

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Additional ML algorithms (XGBoost, Logistic Regression, SVM)
- Cross-validation for more robust evaluation
- SHAP values for model interpretability
- Docker containerization for easy deployment
- REST API endpoint for programmatic access

## Author

**Rahul Mahasetha**
- GitHub: [@rahulmahasetha](https://github.com/rahulmahasetha)
- Repository: [churn-prediction](https://github.com/rahulmahasetha/churn-prediction.git)

## License

This project is open source and available under the MIT License.
