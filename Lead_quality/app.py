# ==========================================
# FicZon Inc - Flask App for Lead Quality Prediction
# ==========================================

from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# -------------------------------
# Load trained model and encoder
# -------------------------------
model = joblib.load("Best_FicZon_Lead_Model_Random_Forest.pkl")
label_encoder = joblib.load("Lead_LabelEncoder.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from HTML form
        data = {
            'Lead_Source': request.form['Lead_Source'],
            'Lead_Type': request.form['Lead_Type'],
            'Lead_Sector': request.form['Lead_Sector'],
            'Lead_Country': request.form['Lead_Country'],
            'Lead_Engagement_Score': float(request.form['Lead_Engagement_Score']),
            'Lead_Visits': int(request.form['Lead_Visits']),
            'Time_on_Site_Min': float(request.form['Time_on_Site_Min']),
            'Lead_Age_Days': int(request.form['Lead_Age_Days']),
            'Lead_Email_Opened': int(request.form['Lead_Email_Opened']),
            'Lead_Call_Attended': int(request.form['Lead_Call_Attended']),
            'Lead_Score': float(request.form['Lead_Score']),
            'Lead_Region': request.form['Lead_Region'],
            'Lead_Employee_Count': request.form['Lead_Employee_Count']
        }

        # Convert form data into DataFrame
        input_df = pd.DataFrame([data])

        # Make prediction
        pred = model.predict(input_df)[0]
        decoded_pred = label_encoder.inverse_transform([pred])[0]

        return render_template('index.html',
                               prediction_text=f"Predicted Lead Quality: {decoded_pred}")

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
