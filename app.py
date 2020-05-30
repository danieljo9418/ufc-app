import flask
# from flask import Flask
# import tensorflow
# import pandas as pd
# from tensorflow import keras
# from load import *

# from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import sys 
import os
import base64
sys.path.append(os.path.abspath("./model"))
from load import * 

global model 

model = init()

# Use pickle to load in the pre-trained model.
# with open(f'tf_model3/saved_model.pb', 'rb') as f:
#     model = keras.models.load_model(f)

# with open(f'model/model.json', 'r') as f:
#     model = json.load(f)

app = flask.Flask(__name__)
# app = Flask(__name__)

# app = flask.Flask(__name__, template_folder='templates')
def index_view():
    return render_template('main.html')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        R_Weight = flask.request.form['R_Weight']
        R_Height = flask.request.form['R_Height']
        R_Age = flask.request.form['R_Age']
        B_Weight = flask.request.form['B_Weight']
        B_Height = flask.request.form['B_Height']
        B_Age = flask.request.form['B_Age']
        RPrev = 0
        BPrev = 0
        BStreak = 0
        RStreak = 0

        input_variables = pd.DataFrame([[BPrev, BStreak, B_Age,B_Height,B_Weight,RPrev, RStreak, R_Age,R_Height,R_Weight]],
                                       columns=['BPrev','BStreak','B_Age','B_Height','B_Weight','RPrev','RStreak','R_Age','R_Height','R_Weight'],
                                       dtype=float,
                                       index=['input'])
        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',
                                     original_input={'R_Weight' :R_Weight,
                                                    'R_Height' :R_Height,
                                                    'R_Age' :R_Age,
                                                    'B_Weight' :B_Weight,
                                                    'B_Height' :B_Height,
                                                    'B_Age' :B_Age
                                                     },
                                     result=prediction,
                                     )
        # print(prediction)

# @app.route('/predict/',methods=['GET','POST'])
# def predict():
# 	imgData = request.get_data()
# 	convertImage(imgData)
# 	x = imread('output.png',mode='L')
# 	x = np.invert(x)
# 	x = imresize(x,(28,28))
# 	x = x.reshape(1,28,28,1)

# 	with graph.as_default():
# 		out = model.predict(x)
# 		print(out)
# 		print(np.argmax(out,axis=1))

# 		response = np.array_str(np.argmax(out,axis=1))
# 		return response	

if __name__ == '__main__':
    app.run(debug=True, port=8000)

# if __name__ == '__main__':
#     app.run(debug=True)