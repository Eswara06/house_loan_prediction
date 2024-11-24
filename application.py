import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

model = pickle.load(open('models/model.pkl','rb'))

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'POST':
       
        data = request.form

        
        features = [
            data.get('Gender'),
            data.get('Married'),
            data.get('Dependents'),
            data.get('Education'),
            data.get('Self_Employed'),
            data.get('ApplicantIncome'),
            data.get('CoapplicantIncome'),
            data.get('LoanAmount'),
            data.get('Loan_Amount_Term'),
            data.get('Credit_History'),
            data.get('Property_Area')
        ]

        
        input_array = np.array(features).reshape(1, -1)

        
        scaler = StandardScaler()
        input_array_scaled = scaler.fit_transform(input_array)
        prediction = model.predict(input_array_scaled)
        result = "Approved" if prediction[0] == 1 else "Rejected"
        return render_template('home.html', result=result)

    else:
        
        return render_template('home.html', result=None)


if __name__=="__main__":
    app.run(port=5001)