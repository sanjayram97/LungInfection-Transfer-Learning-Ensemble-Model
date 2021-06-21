
import flask
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from prediction import *
from PIL import Image
import sys
import os

app = flask.Flask(__name__)

def prediction_vote(file_path):
    normal_vote=0
    infected_vote=0
    print('Inception Resnet V2')
    pred,normal_vote,infected_vote = pred_InceptionResnetV2(file_path,normal_vote,infected_vote)
    print(pred)
    print(normal_vote)
    print(infected_vote)
    
    print('Inception V3')
    pred,normal_vote,infected_vote = pred_InceptionV3(file_path,normal_vote,infected_vote)
    print(pred)
    print(normal_vote)
    print(infected_vote)
    
    print('Resnet 50')
    pred,normal_vote,infected_vote = pred_Resnet50(file_path,normal_vote,infected_vote)
    print(pred)
    print(normal_vote)
    print(infected_vote)
    
    print('Resnet152 V2')
    pred,normal_vote,infected_vote = pred_Resnet152V2(file_path,normal_vote,infected_vote)
    print(pred)
    print(normal_vote)
    print(infected_vote)
    
    print('VGG16')
    pred,normal_vote,infected_vote = pred_Vgg16(file_path,normal_vote,infected_vote)
    print(pred)
    print(normal_vote)
    print(infected_vote)
    
    print('VGG19')
    pred,normal_vote,infected_vote = pred_Vgg19(file_path,normal_vote,infected_vote)
    print(pred)
    print(normal_vote)
    print(infected_vote)
    
    print('---------------------------------')
    print('NORMAL VOTE    :',normal_vote)
    print('INFECTED VOTE  :',infected_vote)
    print('---------------------------------')
    preds='NORMAL' if normal_vote>=infected_vote else 'INFECTED'
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    preds=prediction_vote(file_path)
    result=preds
    return result


if __name__ == '__main__':
    app.run(port=5000,debug=True)