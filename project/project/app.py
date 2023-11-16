from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd  # Corrected import statement

app = Flask(__name__)
model = pickle.load(open('adb_model.pkl', 'rb'))
scalar = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    input_features = [float(request.form[x]) for x in request.form]  # Retrieving form values correctly
    input_features = np.array(input_features).reshape(1, -1)  # Reshape features for prediction
    scaled_features = scalar.transform(input_features)  # Scale the features

    prediction = model.predict(scaled_features)
    output = ""

    if prediction == 0:
        output = "Not a potential customer"
    elif prediction == 1:
        output = "Potential customer"
    else:
        output = "Highly potential customer"

    return render_template('index.html', prediction_text=output)

# running your application
if __name__ == "__main__":
    app.run()