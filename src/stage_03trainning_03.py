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
from src.utils.common import read_yaml, create_directories
from src.utils.model import seperate_y, download_resent_model
from src.utils.callback import save_callback


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

    EPOCHS = params["EPOCHS"]

    model_dir = os.path.join(artifacts_dir, "Models")
    create_directories([model_dir])

    multioutput = os.path.join(model_dir, "multioutput.h5")

    np_saved_dir  = os.path.join(artifacts_dir, "Trainning_data")
    create_directories([np_saved_dir])
    label_np_file = os.path.join(np_saved_dir, "y_data.npy")
    img_np_file = os.path.join(np_saved_dir, "X_data.npy")

    X = np.load(img_np_file, allow_pickle=True)
    y = np.load(label_np_file, allow_pickle=True)
    # Seperating all the columns of label. 
    a,b,c,d = seperate_y(y)

    # lOading pretrained model resnet 50
    resnet_base = download_resent_model()
    
    xinput = Input(shape=(224, 224, 3))
    inp = resnet_base(xinput)

    prediction1 = layers.Dense(1, activation='sigmoid', name='a')(inp)
    prediction2 = layers.Dense(1, activation='sigmoid', name='b')(inp)
    prediction3 = layers.Dense(1, activation='sigmoid', name='c')(inp)
    prediction4 = layers.Dense(1, activation='sigmoid', name='d')(inp)

    model = Model(xinput,[prediction1, prediction2, prediction3, prediction4])
    tb_callback = save_callback()


    logging.info(f"The summary of the model is {model.summary()}")

    model.compile(loss = ["binary_crossentropy", "binary_crossentropy","binary_crossentropy","binary_crossentropy"], optimizer="adam", metrics = ['accuracy'])

    model.fit(X, [a,b,c,d], epochs = EPOCHS, validation_split = 0.2, callbacks =tb_callback)

    model.save(multioutput)
    logging.info(f"MOdel is build and saved at {multioutput}")

    logging.info(f"Callbacks are at logs directory and model is at {model_dir}")
    logging.info("Trainning Complete")





    
    
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