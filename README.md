# Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)

A machine learning project for predicting customer churn using Random Forest classification. This project includes a complete training pipeline with class imbalance handling and an interactive Streamlit web application for real-time predictions.

## Features

- **Random Forest Model**: Robust ensemble classifier with optimized hyperparameters
- **Class Imbalance Handling**: Automatic computation of class weights for imbalanced datasets
- **Feature Importance Analysis**: Identification of key factors driving customer churn
- **Interactive Streamlit UI**: User-friendly web interface with organized input forms
- **Data Preprocessing Pipeline**: Automated encoding and scaling of features
- **Model Persistence**: Save and load trained models, scalers, and encoders

## Prerequisites

- Python 3.7 or higher
- pip package manager
- `churn.csv` dataset in the root directory

## Installation

1. Clone the repository:
```bash
2. Install required dependencies:
```bash
3. Ensure the dataset `churn.csv` is present in the root directory.

## Usage

### Training the Model

Run the training script to preprocess data, train the model, and save artifacts:

```bash
This will:
- Create a `models/` directory
- Save the trained model, scaler, and encoders
- Display model performance metrics and feature importance rankings
- Generate classification report and confusion matrix

### Running the Web Application

Launch the interactive Streamlit app:

```bash
The application features:
- Two-column layout for organized data input
- Real-time churn probability prediction
- Model accuracy display in sidebar
- Support for all categorical and numerical features

## Project Structure

```
## Model Details

### Algorithm
- **Type**: Random Forest Classifier
- **Parameters**:
  - `n_estimators=100`: Number of trees in the forest
  - `max_depth=10`: Maximum depth of trees
  - `min_samples_split=20`: Minimum samples required to split a node
  - `min_samples_leaf=10`: Minimum samples required at leaf node
  - `class_weight`: Balanced weights for handling imbalanced classes
  - `random_state=42`: Reproducibility seed
### Data Preprocessing
1. **Categorical Encoding**: Label Encoding for all object-type columns
2. **Feature Scaling**: StandardScaler for numerical feature normalization
3. **Train/Test Split**: 80/20 split with stratification to maintain class distribution
4. **Class Imbalance Handling**: Computation of balanced class weights using `sklearn.utils.class_weight`

### Input Data Format

The `churn.csv` dataset should contain the following:

**Target Variable:**
- `Churn`: Binary indicator (0 = No, 1 = Yes)

**Categorical Features:**
- `Gender`: Customer gender
- `Partner`: Whether customer has a partner (Yes/No)
- `Dependents`: Whether customer has dependents (Yes/No)
- Other categorical columns (automatically encoded)

**Numerical Features:**
- `Age`: Customer age
- `Tenure`: Number of months as customer
- Other numerical columns (automatically scaled)

## Outputs

### Model Performance Metrics
- **Accuracy Score**: Overall prediction accuracy
- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: True vs predicted class counts

### Feature Importance
Ranked list of features contributing most to churn prediction, helping identify key business drivers.

### Predictions
- Binary churn prediction (Yes/No)
- Probability scores for churn risk assessment

## Dependencies

- `streamlit`: Web application framework
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning algorithms and preprocessing
- `matplotlib`: Visualization (for feature importance plots)
- `joblib`: Model serialization and deserialization
- `numpy`: Numerical computations

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
