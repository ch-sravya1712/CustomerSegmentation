#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas 

from flask import Flask, request, jsonify, render_template
app = Flask(__name__)
model =pickle.load(open('adb_model.pkl','rb'))
scalar=pickle.load(open('scaler.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html') #rendering the home page
@app.route ('/predict', methods=["POST", "GET"])  #route to show the predictions in a web UI
def predict():

#reading the inputs given by the user 
    input_feature =[float(x) for x in request.form.values()]
    features_values = [np.array(input_feature)]
    feature_names =[[ 'Sex', 'Marital status', 'Age', 'Education', 'Income','Occupation','Settlement size']]
    print(len(input_feature))
    print(len(feature_names)) 
    data = pandas.DataFrame(features_values,columns=feature_names)
    data =scalar.fit_transform(features_values)
    #predictions using the Iceded model file
    prediction= model.predict(data)
    print(prediction)
    if (prediction ==0):
        return render_template("index.html",prediction_text= "not a potential customer")

    elif (prediction ==1):
        return render_template("index.html",prediction_text ="Potential culitomer")
    else:
        return render_template("index.html",prediction_text ="Highly potential customer")

    #showing the prediction results in a UI
if __name__=="__main__":
    app.run(debug=True)

    

