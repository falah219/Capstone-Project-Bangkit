# Importing Libraries
import os
import json
import shutil
import time
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow.keras as keras
import pandas as pd

from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, jsonify

# Flask Object
app = Flask(__name__)

app.config['UPLOAD_FOLDER']="images"

# Load the Machine Learning Model
model = tf.keras.models.load_model("CapstoneModel90val.h5", custom_objects={'KerasLayer':hub.KerasLayer})

# Server Test Functions
@app.route("/")
def hello():
    return 'Hello World'

# Function to preprocess, route and to predict the image
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        shutil.rmtree('images')
        os.makedirs('images')
        upload_image=request.files['images']
        filepath=os.path.join(app.config['UPLOAD_FOLDER'],upload_image.filename)
        upload_image.save(filepath)

        # arahkan ke gambar beserta nama filenya
        fname = "images/{}".format(os.listdir('images/')[0])
        
        df = pd.read_csv("model/label.csv", sep = ";")
        def return_label(array):
            largest = 0
            for x in range(0, len(array)):
                if(array[x] > largest):
                    largest = array[x]
                    y = x
            return y

        # Read the images
        image_size = (145, 145)
        test_image = image.load_img(fname, target_size = image_size)
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        label = return_label(result[0])
        if label == 0:
            id = int(str(time.time()).replace('.', '')[5:75])
            label_name = 'Ayam Goreng'
            calories = df.loc[df["Label"] == label_name]["Calories"][0]
            carbohydrate = df.loc[df["Label"] == label_name]["Carbohydrate"][0]
            protein = df.loc[df["Label"] == label_name]["Protein"][0]
            fat = df.loc[df["Label"] == label_name]["Fat"][0]
        elif label == 1:
            id = int(str(time.time()).replace('.', '')[5:75])
            label_name = 'Bakso'
            calories = df.loc[df["Label"] == label_name]["Calories"][1]
            carbohydrate = df.loc[df["Label"] == label_name]["Carbohydrate"][1]
            protein = df.loc[df["Label"] == label_name]["Protein"][1]
            fat = df.loc[df["Label"] == label_name]["Fat"][1]  
        elif label == 2:
            id = int(str(time.time()).replace('.', '')[5:75])
            label_name = 'Batagor'
            calories = df.loc[df["Label"] == label_name]["Calories"][2]
            carbohydrate = df.loc[df["Label"] == label_name]["Carbohydrate"][2]
            protein = df.loc[df["Label"] == label_name]["Protein"][2]
            fat = df.loc[df["Label"] == label_name]["Fat"][2]  
        elif label == 3:
            id = int(str(time.time()).replace('.', '')[5:75])
            label_name = 'Bubur'
            calories = df.loc[df["Label"] == label_name]["Calories"][3]
            carbohydrate = df.loc[df["Label"] == label_name]["Carbohydrate"][3]
            protein = df.loc[df["Label"] == label_name]["Protein"][3]
            fat = df.loc[df["Label"] == label_name]["Fat"][3]
        elif label == 4:
            id = int(str(time.time()).replace('.', '')[5:75])
            label_name = 'Burger'
            calories = df.loc[df["Label"] == label_name]["Calories"][4]
            carbohydrate = df.loc[df["Label"] == label_name]["Carbohydrate"][4]
            protein = df.loc[df["Label"] == label_name]["Protein"][4]
            fat = df.loc[df["Label"] == label_name]["Fat"][4]
        elif label == 5:
            id = int(str(time.time()).replace('.', '')[5:75])
            label_name = 'Capcay'
            calories = df.loc[df["Label"] == label_name]["Calories"][5]
            carbohydrate = df.loc[df["Label"] == label_name]["Carbohydrate"][5]
            protein = df.loc[df["Label"] == label_name]["Protein"][5]
            fat = df.loc[df["Label"] == label_name]["Fat"][5] 
        elif label == 6:
            id = int(str(time.time()).replace('.', '')[5:75])
            label_name = 'Crepes'
            calories = df.loc[df["Label"] == label_name]["Calories"][6]
            carbohydrate = df.loc[df["Label"] == label_name]["Carbohydrate"][6]
            protein = df.loc[df["Label"] == label_name]["Protein"][6]
            fat = df.loc[df["Label"] == label_name]["Fat"][6]
        elif label == 7:
            id = int(str(time.time()).replace('.', '')[5:75])
            label_name = 'Cumi Goreng Tepung'
            calories = df.loc[df["Label"] == label_name]["Calories"][7]
            carbohydrate = df.loc[df["Label"] == label_name]["Carbohydrate"][7]
            protein = df.loc[df["Label"] == label_name]["Protein"][7]
            fat = df.loc[df["Label"] == label_name]["Fat"][7]
        elif label == 8:
            id = int(str(time.time()).replace('.', '')[5:75])
            label_name = 'Fu Yung Hai'
            calories = df.loc[df["Label"] == label_name]["Calories"][8]
            carbohydrate = df.loc[df["Label"] == label_name]["Carbohydrate"][8]
            protein = df.loc[df["Label"] == label_name]["Protein"][8]
            fat = df.loc[df["Label"] == label_name]["Fat"][8]
        elif label == 9:
            id = int(str(time.time()).replace('.', '')[5:75])
            label_name = 'Gado-Gado'
            calories = df.loc[df["Label"] == label_name]["Calories"][9]
            carbohydrate = df.loc[df["Label"] == label_name]["Carbohydrate"][9]
            protein = df.loc[df["Label"] == label_name]["Protein"][9]
            fat = df.loc[df["Label"] == label_name]["Fat"][9]
        elif label == 10:
            id = int(str(time.time()).replace('.', '')[5:75])
            label_name = 'Gudeg'
            calories = df.loc[df["Label"] == label_name]["Calories"][10]
            carbohydrate = df.loc[df["Label"] == label_name]["Carbohydrate"][10]
            protein = df.loc[df["Label"] == label_name]["Protein"][10]
            fat = df.loc[df["Label"] == label_name]["Fat"][10]
        elif label == 11:
            id = int(str(time.time()).replace('.', '')[5:75])
            label_name = 'Ikan Bakar'
            calories = df.loc[df["Label"] == label_name]["Calories"][11]
            carbohydrate = df.loc[df["Label"] == label_name]["Carbohydrate"][11]
            protein = df.loc[df["Label"] == label_name]["Protein"][11]
            fat = df.loc[df["Label"] == label_name]["Fat"][11]
        elif label == 12:
            id = int(str(time.time()).replace('.', '')[5:75])
            label_name = 'Kebab'
            calories = df.loc[df["Label"] == label_name]["Calories"][12]
            carbohydrate = df.loc[df["Label"] == label_name]["Carbohydrate"][12]
            protein = df.loc[df["Label"] == label_name]["Protein"][12]
            fat = df.loc[df["Label"] == label_name]["Fat"][12] 
        elif label == 13:
            id = int(str(time.time()).replace('.', '')[5:75])
            label_name = 'Rendang'
            calories = df.loc[df["Label"] == label_name]["Calories"][13]
            carbohydrate = df.loc[df["Label"] == label_name]["Carbohydrate"][13]
            protein = df.loc[df["Label"] == label_name]["Protein"][13]
            fat = df.loc[df["Label"] == label_name]["Fat"][13] 
        elif label == 14:
            id = int(str(time.time()).replace('.', '')[5:75])
            label_name = 'Sate'
            calories = df.loc[df["Label"] == label_name]["Calories"][14]
            carbohydrate = df.loc[df["Label"] == label_name]["Carbohydrate"][14]
            protein = df.loc[df["Label"] == label_name]["Protein"][14]
            fat = df.loc[df["Label"] == label_name]["Fat"][14]           
        os.remove("images/{}".format(upload_image.filename))
        return jsonify(id=id, label=label_name, calories=calories, carbohydrate=carbohydrate, protein=protein, fat=fat)
    else:
        return "Internal Server Error, Using Method Get but not Run the Code"

# Functions for Food Dictionary
@app.route("/dictionary")
def dictionary():
    f = open('label.json')
    data = json.load(f)
    return data   

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)