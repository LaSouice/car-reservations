from flask import Flask, request, jsonify
import os
import pickle
import pandas as pd

app = Flask(__name__)

##loading the model from the saved file
pkl_filename = "model/regressor.pickle"
with open(pkl_filename, 'rb') as f_in:
    model = pickle.load(f_in)

def predict_reservations(data:pd.DataFrame):
    
    if type(data) == dict:
        df = pd.DataFrame(data,index=[0])
    else:
        df = pd.DataFrame(data)
    y_pred = model.predict(df)
    return y_pred

@app.route('/')
def welcome():
    return 'Welcome to the Model Serving API!'

@app.route('/prediction', methods=['POST'])
def prediction():
    data = request.get_json()
    predict = predict_reservations(data)
    predictOutput = predict
    return jsonify({'predictions': predictOutput.tolist()})



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000,debug=True)