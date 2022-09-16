import re
import pandas as pd
from flask import Flask, render_template, request
import pickle 
import numpy as np

app = Flask(__name__)
data=pd.read_csv("Cleaned_data.csv")
pipe = pickle.load(open('RandomForest_model.pkl','rb'))


@app.route('/')
def index():
    locations=sorted(data["Location"].unique())
    return render_template('index.html',locations=locations)


@app.route('/predict',methods=['POST'])
def predict():
    location=request.form.get('location')
    bedroom=request.form.get('bhk')
    total_area=request.form.get('area')
    sale=request.form.get('sales')
    parking=request.form.get('car')
    lift=request.form.get('lift')
    maintenance=request.form.get('maintenance')
    clubhouse=request.form.get('clubhouse')
    security=request.form.get('security')
    gas=request.form.get('gas')
    if(gas==None):
        gas=0
    if(parking==None):
        parking=0
    if(security==None):
        security=0
    if(maintenance==None):
        maintenance=0
    if(security==None):
        security=0
    if(clubhouse==None):
        clubhouse=0
    if(lift==None):
        lift=0
    print(total_area,location,bedroom,sale,lift,parking,maintenance,security,clubhouse,gas)
    print(type(gas))
    input=pd.DataFrame([[total_area,location,bedroom,sale,lift,parking,maintenance,security,clubhouse,gas]],columns=['Area','Location','No. of Bedrooms','New/Resale','Lift Available','Car Parking','Maintenance Staff','24x7 Security','Clubhouse','Gas Connection'])
    prediction = pipe.predict(input)[0]
    print(np.round(prediction,2))

    return str(np.round(prediction,2))

if __name__=='__main__':
    app.run(debug=True, port=8001)


