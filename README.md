🧠 Employee Annual Salary Prediction
This project is a machine learning web application designed to predict an employee's exact annual salary based on various personal and professional attributes such as age, education, occupation, and working hours.

🔍 Problem Statement
The goal is to predict continuous income values using a modified Adult Census Income Dataset. This is a regression problem, aiming to estimate a specific numerical salary rather than categorizing it.

📊 Features Used
Age

Education Level

Job Role (Occupation)

Hours Worked Per Week

Years of Experience (represented by educational-num)

Other features from dataset (for batch prediction):

Marital Status

Relationship

Race

Gender

Capital Gain

Capital Loss

Workclass

Native Country

🛠️ Tech Stack
Python

Pandas & NumPy

Scikit-learn

Matplotlib / Seaborn (for analysis, if used in model_training.ipynb)

Jupyter Notebook (for model_training.ipynb)

Streamlit (for Web App)

Joblib (for saving model and preprocessors)

GitHub (for version control)

🚀 How to Run Locally
Clone the repository:

git clone https://github.com/your-username/employee-salary-prediction.git
cd employee-salary-prediction

Install dependencies:

pip install -r requirements.txt

(Note: You might need to create a requirements.txt file if you don't have one. You can generate it using pip freeze > requirements.txt after installing all libraries.)

Train the model:
Run the train_salary_model.py script to train the regression model and save the necessary .pkl files (salary_model.pkl, label_encoders.pkl, feature_scaler.pkl).

python train_salary_model.py

Start the web application:

streamlit run streamlit_app.py

📂 Project Structure
.
├── train_salary_model.py    # Python script for model training and saving
├── streamlit_app.py         # Streamlit web application
├── adult 3 (1).csv          # Your dataset (ensure 'income' column is numerical or mapped)
├── salary_model.pkl         # Trained Machine Learning model (RandomForestRegressor)
├── label_encoders.pkl       # Saved LabelEncoders for categorical features
├── feature_scaler.pkl       # Saved StandardScaler for numerical features
├── requirements.txt         # Python dependencies
└── README.md                # Project overview

🧪 Model Used
Random Forest Regressor

Key Metrics (Example Values - will vary based on training):

Mean Absolute Error (MAE): ~8500.00 (e.g., average prediction is off by $8500)

Mean Squared Error (MSE): ~195000000.00

Root Mean Squared Error (RMSE): ~13900.00

R-squared (R2) Score: ~0.75 (e.g., 75% of variance explained)

Feature Encoding: LabelEncoder

Scaling: StandardScaler (for numerical features)

🖥️ Web App Features
✅ Predict exact annual salary for a single input (via sliders and dropdowns)

📁 Batch prediction using CSV upload (if implemented in streamlit_app.py)

📥 Download predictions as CSV (if implemented in streamlit_app.py)

📚 Dataset
Adult Income Dataset from UCI Machine Learning Repository (modified to include numerical salary values or mapped categories for regression).

👤 Author
Brijeshrath67
BTech CSE (AI & ML)

🌟 Acknowledgements
UCI ML Repository

Streamlit Documentation

Scikit-learn Docs

📌 Future Improvements
Add cloud deployment (e.g., Streamlit Cloud, Render, or HuggingFace)

Model performance dashboard

Hyperparameter tuning & model comparison

Explore more advanced regression models (e.g., XGBoost, LightGBM)

Implement a more sophisticated numerical mapping or acquire a dataset with true numerical salaries for enhanced accuracy.

📄 License
This project is licensed under the MIT License.
