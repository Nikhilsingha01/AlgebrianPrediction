import pickle
from flask import Flask , request , jsonify , render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# # create a flask app innstance
application = Flask(__name__)
app = application
 
 
print("Starting app...")

try:
    ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
    print("Loaded ridge model")
    standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    print("Loaded scaler")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata' , methods=['GET' , 'POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        
    
        new_data_scaled= standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html' , results=result[0])
    

    else:
        return render_template('home.html')

if __name__ == "__main__":
    print("Launching Flask server...")
    app.run(debug=True, host="127.0.0.1", port=5000)
