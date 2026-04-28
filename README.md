 # Customer Churn Prediction
 
 ![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
 ![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
 ![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
 
 A machine learning project for predicting customer churn using Random Forest classification. This project includes a complete training pipeline with class imbalance handling and an interactive Streamlit web application for real-time predictions.
 
 
 ## Installation
 
 1. Clone the repository:
 ```bash
git clone https://github.com/rahulmahasetha/churn-prediction.git
cd churn-prediction
 ```

 2. Install required dependencies:
 ```bash
pip install -r requirements.txt
 ```

 3. Ensure the dataset `churn.csv` is present in the root directory.
 
 ## Usage
 
 Run the training script to preprocess data, train the model, and save artifacts:
 
 ```bash
python train_model.py
```
 
 This will:
 - Create a `models/` directory
 
 Launch the interactive Streamlit app:
 
 ```bash
streamlit run app.py
```
 
 The application features:
 - Two-column layout for organized data input
 ## Project Structure
 
 ```
churn-prediction/
├── churn.csv                 # Dataset file
├── train_model.py           # Model training script
├── app.py                   # Streamlit web application
├── requirements.txt         # Python dependencies
├── models/                  # Directory for saved model artifacts
│   └── (generated files)
└── README.md               # Project documentation
```

## How It Works

The ML pipeline follows these steps:

1. **Data Loading**: Load customer data from `churn.csv`
2. **Preprocessing**:
   - **Label Encoding**: Convert categorical features to numerical values
   - **StandardScaler**: Normalize numerical features for optimal model performance
3. **Class Imbalance Handling**: Compute balanced class weights to address uneven churn distribution
4. **Model Training**: Train Random Forest classifier with 100 estimators and optimized hyperparameters
5. **Evaluation**: Generate accuracy, precision, recall metrics and feature importance analysis
6. **Prediction**: Deploy trained model in Streamlit app for real-time churn prediction
 
 ## Model Details
 
   - `n_estimators=100`: Number of trees in the forest
   - `max_depth=10`: Maximum depth of trees
   - `min_samples_split=20`: Minimum samples required to split a node
  - `min_samples_leaf=10`: Minimum samples required at leaf node
   - `class_weight`: Balanced weights for handling imbalanced classes
   - `random_state=42`: Reproducibility seed
 
 
 **Categorical Features:**
 - `Gender`: Customer gender
- `Partner`: Whether customer has a partner (Yes/No)
- `Dependents`: Whether customer
- `Partner`: Whether customer has a partner (Yes/No)
- `Dependents`: Whether customer has dependents (Yes/No)
- `Contract Type`: Type of contract (Month-to-month/One year/Two year)
- `Payment Method`: Payment method (Electronic check/Mailed check/Bank transfer/Credit card)
- `Internet Service`: Type of internet service (DSL/Fiber optic/No)
- `Phone Service`: Whether customer has phone service (Yes/No)
- `Multiple Lines`: Whether customer has multiple lines (Yes/No)
- `TV`: Whether customer has TV service (Yes/No)
- `Streaming`: Whether customer has streaming service (Yes/No)

**Numerical Features:**
- `Age`: Customer age (1-120)
- `Tenure`: Number of months as customer (0-72)
- `Monthly Charges`: Monthly charges amount
- `Total Charges`: Total charges amount

## Results & Evaluation

The model provides comprehensive evaluation metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision**: Ratio of correctly predicted churn cases
- **Recall**: Ability to identify actual churn cases
- **Classification Report**: Detailed per-class metrics
- **Feature Importance**: Visualization of which features most influence churn predictions

Top predictive features typically include contract type, tenure, monthly charges, and payment method.

## Sample Usage Examples

### Example 1: Low Risk Customer
- **Profile**: 2-year contract, 60 months tenure, automatic bank transfer
- **Prediction**: Likely to stay (No Churn)
- **Confidence**: High

### Example 2: High Risk Customer  
- **Profile**: Month-to-month contract, 2 months tenure, electronic check payment
- **Prediction**: Likely to churn (Yes)
- **Confidence**: High

### Interpreting Results
- **Churn = No**: Customer is predicted to remain with the service
- **Churn = Yes**: Customer is at risk of leaving; consider retention strategies

## Troubleshooting

### Common Issues

**1. Missing churn.csv file**
```
FileNotFoundError: churn.csv not found
```
**Solution**: Ensure the dataset file is in the root directory. Download or create the CSV file with the required columns.

**2. Module not found errors**
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution**: Install dependencies using `pip install -r requirements.txt`

**3. Permission errors when creating models/ directory**
```
PermissionError: [Errno 13] Permission denied: 'models/'
```
**Solution**: Ensure you have write permissions in the project directory, or manually create the `models/` directory.

**4. Streamlit port already in use**
```
Address already in use
```
**Solution**: Use a different port: `streamlit run app.py --server.port 8502`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/) for machine learning
- Web interface powered by [Streamlit](https://streamlit.io/)
- Dataset inspired by telecom customer churn scenarios
