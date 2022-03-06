from flask import Flask,render_template,request
import pickle
import librosa
import pandas as pd 
import numpy as np 
import glob 
import os
import sys
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


ALLOWED_EXTENSIONS = set(['wav'])

app = Flask(__name__)
ser_model=pickle.load(open("final_ser.pkl","rb"))

def allowed_file(filename):
    return '.' in filename and \
    filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload():
    return render_template('index.html')

@app.route('/output',methods=['GET','POST'])
def output():
    if request.method == 'POST':
        file = request.files['file']

    if file and allowed_file(file.filename):
        X, sample_rate = librosa.load(file, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,n_mfcc=40).T, axis=0)
        feature_data = mfccs.reshape(1,-1)
        scaler=StandardScaler()
        scale=scaler.fit_transform(feature_data)
        x = ser_model.predict(scale)
    return render_template("predict.html",result=x)


if __name__=='__main__':
    app.run(debug=True)
        