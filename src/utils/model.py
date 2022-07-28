#from keras.applications.vgg16 import decode_predictions
from copyreg import pickle
from keras.applications.regnet import preprocess_input
import tensorflow as tf
from keras.utils import load_img
import pandas as pd 
import numpy as np 
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
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


def load_resnet_model():
    """
    This will load resnet model and returns the output which can be used for trainning
    """
    resnet_base = tf.keras.applications.ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet')

    for layer in resnet_base.layers:
        layer.trainable = False

    return resnet_base
