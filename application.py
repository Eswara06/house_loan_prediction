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

if __name__=="__main__":
    app.run(host='0.0.0.0')