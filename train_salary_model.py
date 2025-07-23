import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler # Added StandardScaler for potential feature scaling
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np # Import numpy for sqrt

# Load the dataset
df = pd.read_csv("adult 3 (1).csv")

# Initialize encoders dictionary for categorical features
encoders = {}

# List of categorical columns to encode
categorical_columns = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "gender", "native-country"
]

# Encode categorical features and store encoders
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = df[col].astype(str).str.strip()
    le.fit(df[col])
    df[col] = le.transform(df[col])
    encoders[col] = le

# --- IMPORTANT: Refined Income Mapping for Regression Target ---
df['income'] = df['income'].astype(str).str.strip()
income_mapping = {
    '<=50K': 35000,  # Adjusted representative salary for <=50K (e.g., mid-point)
    '>50K': 80000    # Adjusted representative salary for >50K (e.g., mid-point)
}
df['income'] = df['income'].map(income_mapping)

# Handle any unmapped values (e.g., if there are other categories in 'income' not in mapping)
if df['income'].isnull().any():
    print("Warning: Some 'income' values could not be mapped. Dropping rows with NaN income.")
    df.dropna(subset=['income'], inplace=True)
# --- End of Income Mapping ---

# Split features and target
X = df.drop("income", axis=1)
y = df["income"]

# --- Feature Scaling for Numerical Features ---
# This section scales your numerical input features.
# It's crucial that the scaler is saved and then loaded in the Streamlit app.
# Corrected column name from 'education-num' to 'educational-num'
numerical_columns = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
scaler = StandardScaler()
# Fit scaler only on training data to prevent data leakage
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
# --- IMPORTANT: Ensure this line is NOT commented out! ---
joblib.dump(scaler, "feature_scaler.pkl")
# --- End of Feature Scaling ---


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the regression model
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate the regression model
predictions = model.predict(X_test)

print("--- Model Evaluation (Regression Metrics) ---")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, predictions):.2f}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, predictions):.2f}")
# Calculate RMSE manually by taking the square root of MSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}") # Corrected line
print(f"R-squared (R2) Score: {r2_score(y_test, predictions):.2f}")
print("------------------------------------------")

# Save the model and encoders
joblib.dump(model, "salary_model.pkl")
joblib.dump(encoders, "label_encoders.pkl")

print("✅ Regression model, encoders, and feature scaler saved successfully.")
print("Note: The 'income' column was mapped to numerical values for regression. For true exact salaries, a dataset with actual numerical salary values is highly recommended.")
