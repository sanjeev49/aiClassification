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
    raw_data_dir = os.path.join(root_dir, artifacts["RAW_DATA_DIR"])
    raw_label_file = os.path.join(raw_data_dir, artifacts["RAW_LABEL_FILE"])
    raw_img_dir = os.path.join(raw_data_dir, artifacts["RAW_IMAGE_DIR"])

    cleaned_csv = os.path.join(prepare_data_dir, artifacts["CLEANED_CSV_FILE"])

    training_data_dir  = os.path.join(artifacts_dir, artifacts["TRAINING_DATA_DIR"])
    create_directories([training_data_dir])
    label_np_file = os.path.join(training_data_dir, artifacts["LABEL_TRAING_DATA"])
    img_np_file = os.path.join(training_data_dir, artifacts["IMG_TRAINING_DATA"])


    if not os.path.exists(img_np_file):
        logging.info("Tranning data does not found creating one at location {}".format(training_data_dir))
        raw_d = _load_image(raw_img_dir, cleaned_csv)

        X, y = seperate_x_y_from_raw_d(raw_d)

        np.save(label_np_file, y, allow_pickle=True)
        np.save(img_np_file, X, allow_pickle=True)
        logging.info(f"Trinning data has been saved at location {training_data_dir}")
    else:
        logging.info("Trainning data found, loading it from {}".format(training_data_dir))
    
    
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