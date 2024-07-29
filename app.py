import pickle
from flask import Flask , render_template , url_for ,request
import pandas as pd
import numpy as np


app = Flask(__name__)

Crop_model = pickle.load(open('RandomForestClassifier.pkl' , 'rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/results' , methods = ["GET" , "POST"])
def results():
    Nitrogen = request.form.get('nitrogen')
    Phosphorus = request.form.get('phosphorus')
    Potassium = request.form.get('potassium')
    temperature = request.form.get('temperature')
    humidity = request.form.get("humidity")
    ph = request.form.get("ph")
    rainfall = request.form.get("rainfall")
    
    data = np.array([[Nitrogen , Phosphorus , Potassium , temperature , humidity , ph , rainfall]])
    pred_data = Crop_model.predict(data)
    #print(pred_data)
    
    return render_template('index.html' , data = pred_data[0].upper())


if __name__ == '__main__':
    app.run(debug=True)