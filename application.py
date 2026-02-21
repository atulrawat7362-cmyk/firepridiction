import pickle

from flask import Flask,request,jsonify,redirect,render_template 
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

## import ridge regressor and  standard scaler pickle 
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))



@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoints():
    if request.method=="POST":
        Temperature = request.form.get('Temperature')  # ye name se math hona ciye us form ke 
        RH= request.form.get('RH')
        WS = request.form.get('WS')
        Rain = request.form.get('Rain')
        FFMC = request.form.get('FFMC')
        DMC = request.form.get('DMC')
        ISI = request.form.get('ISI')
        Classes = request.form.get('Classes')
        Region= request.form.get('Region')

        new_data_scaled=standard_scaler.transform([[Temperature,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',ans=result[0])






    else :
        return render_template('home.html')



if __name__=="__main__":
    app.run(host="0.0.0.0")


