#from keras.applications.vgg16 import decode_predictions
from tarfile import TarError
from keras.applications.regnet import preprocess_input
import tensorflow as tf
from keras.utils import load_img
import pandas as pd 
import numpy as np 
from tensorflow import keras
from tensorflow.keras.utils import img_to_array
import logging
import os 


def _load_image(img_dir: os.path, cleaned_df: os.path) -> None:
    df = pd.read_csv(cleaned_df, header=None)
    raw_data = []
    for num , img in enumerate(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img)
        im  = load_img(img_path, color_mode="rgb", target_size=(224,224,3))
        im = img_to_array(im)
        im = preprocess_input(im)
        raw_data.append([im, [df.loc[num][0], df.loc[num][1], df.loc[num][2], df.loc[num][3]]])
    return raw_data

def seperate_x_y_from_raw_d(li: list) -> list:
    
    x = []
    y = []
    for i , data in enumerate(li):
        x.append(li[i][0])
        y.append(li[i][1])

    x = np.array(x)
    y = np.array(y)
    return x , y

def seperate_y(y: np.abs) -> list:
    """
    It will seperate all the labels so that we can train a mutioutput model.
    """
    a,b,c,d = [], [], [], [] 
    for i in y:
        a.append(i[0]) , b.append(i[1]), c.append(i[2]), d.append(i[3])
    return np.array(a), np.array(b), np.array(c), np.array(d)


def download_resent_model():
    """
    This will load resnet model and returns the output which can be used for trainning
    """
    resnet_base = tf.keras.applications.ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet')

    for layer in resnet_base.layers:
        layer.trainable = False

    return resnet_base


def load_keras_model(model_path: os.path) :
    """
    This will load keras Model
    """

    model = keras.models.load_model(model_path)
    return model

def load_input_img(img_path: os.path):
    img  = load_img(img_path, target_size=(224,224,3))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = img.reshape(1, img.shape[0], img.shape[1], 3)
    return img

def seperate_prediction(arr: np.array):
    li = []
    li.append([arr[0][0], arr[1][0], arr[2][0], arr[3][0]])
    a, b ,c , d = li[0][0][0], li[0][1][0], li[0][2][0] , li[0][3][0]

    li2 = [a,b,c,d]
    li3 = []
    for i in li2:
        if (i>0.5):
            li3.append(1)
        if (i<0.5):
            li3.append(0)
    return li2, li3
