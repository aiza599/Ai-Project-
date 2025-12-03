# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('model_svr.pkl', 'rb'))

# Agar model_svc.pkl nahi hai toh fallback (optional)
if not os.path.exists('model_svr.pkl'):
    print("Warning: model_svr.pkl not found! Using dummy prediction.")
    model = None

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            education = int(request.form['education'])
            job_title = int(request.form['job_title'])
            experience = float(request.form['experience'])
            
            # Input array
            features = np.array([[age, education, job_title, experience]])
            
            # Prediction
            if model:
                predicted_salary = model.predict(features)[0]
                prediction = f"₹{predicted_salary:,.0f}"
            else:
                prediction = "₹1,25,000 (Demo Mode)"
                
        except Exception as e:
            prediction = "Invalid input! Please try again."
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)