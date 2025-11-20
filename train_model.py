# train_model.py (IMPROVED)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight
import joblib
import os

# Create models directory
os.makedirs("models", exist_ok=True)

def main():
    print("Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv("churn.csv")
    print(f"Dataset shape: {df.shape}")
    
    # Create processed copy
    df_processed = df.copy()
    
    # Encode categorical features
    label_cols = df_processed.select_dtypes(include=['object']).columns
    encoders = {}
    
    for col in label_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        encoders[col] = le
    
    # Split data
    X = df_processed.drop("Churn", axis=1)
    y = df_processed["Churn"]
    
    # Handle class imbalance
    classes = np.unique(y)
    class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
    weight_dict = {classes[0]: class_weights[0], classes[1]: class_weights[1]}
    
    print(f"Class distribution: {dict(zip(classes, class_weights))}")
    
    # Train/Test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with better parameters
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight=weight_dict,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())
    
    # Save model and artifacts
    model_artifacts = {
        "model": model,
        "scaler": scaler,
        "encoders": encoders,
        "feature_columns": X.columns.tolist(),
        "accuracy": accuracy,
        "feature_importance": feature_importance
    }
    
    joblib.dump(model_artifacts, "models/churn_model.pkl")
    print(f"\nModel saved successfully to models/churn_model.pkl")
    print(f"Final Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()