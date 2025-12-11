from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [
        float(request.form['engine']),
        float(request.form['cylinders']),
        float(request.form['city']),
        float(request.form['hwy']),
        float(request.form['fuel'])
    ]
    prediction = model.predict([data])[0]
    return render_template('result.html', emission=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
