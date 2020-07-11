import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

import pickle

app = Flask(__name__)
model = pickle.load(open('MSA.pkl', 'rb'))
un = pickle.load(open('union.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    int_features =  request.form.get('hotelrev')

    sent = str(int_features)
    #output=u'' + output.decode('utf-8')
    sent=sent.rstrip()
    sent=sent.lstrip()
    sent2 = un.transform([sent])
    prediction =model.predict(sent2)
    prediction=str(prediction).replace("['", "")
    prediction=str(prediction).replace("']", "")
    outputstatment = prediction# + "\n  " + str(int_features)

    return render_template('index.html', prediction_text=outputstatment)



if __name__ == "__main__":
    app.run(debug=True)
