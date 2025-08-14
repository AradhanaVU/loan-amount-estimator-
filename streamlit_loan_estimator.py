# streamlit_loan_estimator.py

# 1️⃣ Import libraries
import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

# 2️⃣ Load CSV training data
train_df = pd.read_csv("loan_train.csv")  # make sure the file is in the project folder

# 3️⃣ Define target and features
target = 'LoanAmount'
features = train_df.drop(columns=['Loan_ID', 'LoanAmount'])

categorical_cols = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'
]
numerical_cols = [
    'ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term', 'Credit_History'
]

# Fill missing values
for col in categorical_cols:
    features[col].fillna(features[col].mode()[0], inplace=True)
for col in numerical_cols:
    features[col].fillna(features[col].median(), inplace=True)

y = train_df[target]

# Remove rows where y is NaN
mask = ~y.isna()
features = features[mask]
y = y[mask]

# 4️⃣ Build preprocessing + model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model.fit(features, y)

# 5️⃣ Streamlit UI
st.title("🏦 Loan Amount Estimator")
st.write("Predict the estimated loan amount based on applicant details.")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
app_income = st.number_input("Applicant Income", min_value=0)
coapp_income = st.number_input("Coapplicant Income", min_value=0)
loan_term = st.number_input("Loan Amount Term (in months)", min_value=6)  # realistic minimum
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert input into dataframe
input_data = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [app_income],
    'CoapplicantIncome': [coapp_income],
    'Loan_Amount_Term': [loan_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area]
})

# 6️⃣ Predict button
if st.button("Predict Loan Amount"):
    pred = model.predict(input_data)[0]
    st.success(f"💰 Estimated Loan Amount: {pred:.2f}")
    
    # SHAP explanation
    rf_model = model.named_steps['regressor']
    preprocessor = model.named_steps['preprocessor']
    input_transformed = preprocessor.transform(input_data)
    # Use a sample of the training data as background for SHAP
    background = preprocessor.transform(features.sample(100, random_state=42))
    explainer = shap.Explainer(rf_model, background)
    shap_values = explainer(input_transformed)
    
    st.write("### Feature Contributions")
    shap.plots.waterfall(shap_values[0], max_display=8, show=False)
    st.pyplot(plt.gcf())

# 7️⃣ Batch prediction on test data
st.write("---")
st.header("📄 Batch Prediction on Test Data")

uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    required_cols = categorical_cols + numerical_cols
    missing_cols = [col for col in required_cols if col not in test_df.columns]
    if missing_cols:
        st.error(f"Missing columns in uploaded file: {missing_cols}")
    else:
        # Fill missing values in test data
        for col in categorical_cols:
            test_df[col].fillna(features[col].mode()[0], inplace=True)
        for col in numerical_cols:
            test_df[col].fillna(features[col].median(), inplace=True)
        # Predict
        predictions = model.predict(test_df[required_cols])
        test_df['PredictedLoanAmount'] = predictions
        st.write("### Predictions")
        st.dataframe(test_df[['PredictedLoanAmount'] + required_cols])
        st.download_button(
            label="Download predictions as CSV",
            data=test_df.to_csv(index=False),
            file_name="loan_predictions.csv",
            mime="text/csv"
        )