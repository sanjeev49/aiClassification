#from keras.applications.vgg16 import decode_predictions
from keras.applications.regnet import preprocess_input
import tensorflow as tf
from keras.utils import load_img
import pandas as pd 
import numpy as np 
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
tf.data.experimental.enable_debug_mode()


import argparse
import os
import logging
from venv import create
from src.utils.common import read_yaml, create_directories
from src.utils.model import (
    _load_image, 
    seperate_x_y_from_raw_d
)
import random


STAGE = "create_model" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def create_model(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    prepare_data_dir = os.path.join(artifacts_dir, artifacts["PREPARED_DATA_DIR"])

    root_dir = os.getcwd()
    raw_data_dir = os.path.join(root_dir, "Multilabel")
    raw_label_file = os.path.join(raw_data_dir, "labels.txt")
    raw_img_dir = os.path.join(raw_data_dir, "photos")

    cleaned_csv = os.path.join(prepare_data_dir, 'cleaned.csv')

    np_saved_dir  = os.path.join(artifacts_dir, "Trainning_data")
    create_directories([np_saved_dir])
    label_np_file = os.path.join(np_saved_dir, "y_data.npy")
    img_np_file = os.path.join(np_saved_dir, "X_data.npy")

    if not os.path.exists(img_np_file):
        logging.info("Tranning data does not found creating one at location {}".format(np_saved_dir))
        raw_d = _load_image(raw_img_dir, cleaned_csv)

        X, y = seperate_x_y_from_raw_d(raw_d)

        np.save(label_np_file, y, allow_pickle=True)
        np.save(img_np_file, X, allow_pickle=True)
        logging.info(f"Trinning data has been saved at location {np_saved_dir}")
    else:
        logging.info("Trainning data found loading it from {}".format(np_saved_dir))
    
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    
    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        create_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e