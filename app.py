import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ------------------------
# LOAD & PREPARE DATA (FIXED)
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("churn.csv")
    return df

@st.cache_resource
def train_model(df):
    # Create a copy to avoid modifying cached data
    df_processed = df.copy()
    
    # Encode all categorical columns
    label_cols = df_processed.select_dtypes(include=['object']).columns
    encoders = {}
    
    for col in label_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        encoders[col] = le
    
    # Split data PROPERLY
    X = df_processed.drop("Churn", axis=1)
    y = df_processed["Churn"]
    
    # Handle class imbalance
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numeric data (fit only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use Random Forest for better accuracy
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight=weight_dict,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, encoders, X.columns.tolist(), accuracy

# Load data
df = load_data()

# Train model
model, scaler, encoders, feature_columns, accuracy = train_model(df)

# ------------------------
# STREAMLIT APP UI
# ------------------------
st.title("📊 Customer Churn Prediction App (Improved)")
st.write("Enter customer details to check if the customer will churn.")

# Display model accuracy
st.sidebar.metric("Model Accuracy", f"{accuracy:.2%}")

# User Inputs - organized in columns for better UX
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Details")
    gender = st.selectbox("Gender", encoders["Gender"].classes_)
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    partner = st.selectbox("Partner", encoders["Partner"].classes_)
    dependents = st.selectbox("Dependents", encoders["Dependents"].classes_)
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)

with col2:
    st.subheader("Service Details")
    contract = st.selectbox("Contract Type", encoders["Contract Type"].classes_)
    payment = st.selectbox("Payment Method", encoders["Payment Method"].classes_)
    internet = st.selectbox("Internet Service", encoders["Internet Service"].classes_)
    phone = st.selectbox("Phone Service", encoders["Phone Service"].classes_)
    multiple = st.selectbox("Multiple Lines", encoders["Multiple Lines"].classes_)
    tv = st.selectbox("TV", encoders["TV"].classes_)
    streaming = st.selectbox("Streaming", encoders["Streaming"].classes_)

st.subheader("Financial Details")
monthly = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total = st.number_input("Total Charges", min_value=0.0, value=500.0)

# Convert inputs into a DataFrame row with proper feature order
input_data = {
    "Gender": encoders["Gender"].transform([gender])[0],
    "Age": age,
    "Partner": encoders["Partner"].transform([partner])[0],
    "Dependents": encoders["Dependents"].transform([dependents])[0],
    "Tenure": tenure,
    "Contract Type": encoders["Contract Type"].transform([contract])[0],
    "Payment Method": encoders["Payment Method"].transform([payment])[0],
    "Internet Service": encoders["Internet Service"].transform([internet])[0],
    "Phone Service": encoders["Phone Service"].transform([phone])[0],
    "Multiple Lines": encoders["Multiple Lines"].transform([multiple])[0],
    "TV": encoders["TV"].transform([tv])[0],
    "Streaming": encoders["Streaming"].transform([streaming])[0],
    "Monthly Charges": monthly,
    "Total Charges": total
}

# Ensure correct column order
row = pd.DataFrame([input_data])[feature_columns]

# Scale the new row
row_scaled = scaler.transform(row)

# Predict
if st.button("🔍 Predict Churn"):
    pred = model.predict(row_scaled)[0]
    prob = model.predict_proba(row_scaled)[0]
    
    churn_prob = prob[1] * 100
    no_churn_prob = prob[0] * 100
    
    # Display results with better visualization
    st.subheader("Prediction Results")
    
    if pred == 1:
        st.error(f"❌ HIGH RISK: Customer will likely CHURN")
    else:
        st.success(f"✅ LOW RISK: Customer will likely STAY")
    
    # Probability bars
    st.write("### Confidence Levels:")
    col_prob1, col_prob2 = st.columns(2)
    
    with col_prob1:
        st.metric("No Churn Probability", f"{no_churn_prob:.1f}%")
    
    with col_prob2:
        st.metric("Churn Probability", f"{churn_prob:.1f}%")
        
    
    # Feature importance (for insight)
  #  if st.checkbox("Show feature importance for this prediction"):
    #    feature_importance = model.feature_importances_
     #   importance_df = pd.DataFrame({
      #      'Feature': feature_columns,
        #    'Importance': feature_importance
     #   }).sort_values('Importance', ascending=False)
        
     #   st.write("Top factors influencing this prediction:")
     #   st.dataframe(importance_df.head(5))
     
# Add data exploration section
#if st.sidebar.checkbox("Show Data Overview"):
   # st.subheader("Dataset Overview")
   # st.write(f"Total customers: {len(df)}")
   # st.write(f"Churn rate: {(df['Churn'].mean() * 100):.1f}%")
   # st.dataframe(df.head())'''