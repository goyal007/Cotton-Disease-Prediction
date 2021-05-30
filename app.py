#!/usr/bin/env python
# coding: utf-8

# In[12]:


import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
MODEL_PATH ='model_inception.h5'
model = load_model(MODEL_PATH)
def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The leaf is diseased cotton leaf"
    elif preds==1:
        preds="The leaf is diseased cotton plant"
    elif preds==2:
        preds="The leaf is fresh cotton leaf"
    else:
        preds="The leaf is fresh cotton plant"    
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        print(f)
        basepath = os.path.dirname(os.path.abspath('__file__'))
        print(basepath)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        file_path = file_path.replace(os.sep, '/')
        print(file_path)
        f.save(file_path)
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None
app.run()


# In[ ]:




