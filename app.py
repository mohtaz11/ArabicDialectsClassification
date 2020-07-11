import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
import bz2
import _pickle as cPickle
import pickle

app = Flask(__name__)

# Load any compressed pickle file
def decompress_pickle(file):
 data = bz2.BZ2File(file, 'rb')
 data = cPickle.load(data)
 return data

loaded_model = decompress_pickle('MSA.pbz2') 
union = decompress_pickle('union.pbz2') 

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
    sent2 = union.transform([sent])
    prediction =loaded_model.predict(sent2)
    prediction=str(prediction).replace("['", "")
    prediction=str(prediction).replace("']", "")
    outputstatment = prediction + "\n  " + str(int_features)

    return render_template('index.html', prediction_text=outputstatment)



if __name__ == "__main__":
    app.run(debug=True)
