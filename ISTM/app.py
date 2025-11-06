from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("Priority_Prediction_Model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        data = {
            'CI_Cat': request.form['CI_Cat'],
            'CI_Subcat': request.form['CI_Subcat'],
            'Status': request.form['Status'],
            'Impact': request.form['Impact'],
            'Urgency': float(request.form['Urgency']),
            'Category': request.form['Category'],
            'Alert_Status': request.form['Alert_Status'],
            'No_of_Reassignments': float(request.form['No_of_Reassignments']),
            'Handle_Time_hrs': float(request.form['Handle_Time_hrs']),
            'Closure_Code': request.form['Closure_Code'],
            'No_of_Related_Interactions': float(request.form['No_of_Related_Interactions']),
            'No_of_Related_Incidents': float(request.form['No_of_Related_Incidents']),
            'No_of_Related_Changes': float(request.form['No_of_Related_Changes']),
            'Resolution_Time_hrs': float(request.form['Resolution_Time_hrs']),
        }

        input_df = pd.DataFrame([data])

        # ✅ Add default numeric column
        input_df['number_cnt'] = 0.5  # Default mean-like value

        # ✅ Fill all missing columns expected by the model
        expected_cols = [
            'CI_Cat', 'CI_Subcat', 'Status', 'Impact', 'Urgency',
            'number_cnt', 'Category', 'Alert_Status', 'No_of_Reassignments',
            'Handle_Time_hrs', 'Closure_Code', 'No_of_Related_Interactions',
            'No_of_Related_Incidents', 'No_of_Related_Changes', 'Resolution_Time_hrs'
        ]

        for col in expected_cols:
            if col not in input_df.columns:
                # Fill text-type columns with placeholder
                if col in ['CI_Cat', 'CI_Subcat', 'Status', 'Impact', 'Category',
                           'Alert_Status', 'Closure_Code']:
                    input_df[col] = 'Unknown'
                else:
                    input_df[col] = 0  # numeric defaults

        # ✅ Reorder columns to match training order
        input_df = input_df[expected_cols]

        # ✅ Predict
        prediction = model.predict(input_df)[0]

        return render_template('index.html', prediction_text=f"Predicted Priority: {prediction}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
