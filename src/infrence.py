from tensorflow import keras
import tensorflow as tf 
tf.data.experimental.enable_debug_mode()


import argparse
import os
import logging
from src.utils.common import read_yaml
from src.utils.model import  load_keras_model, load_input_img, seperate_prediction
from src.utils.callback import save_callback


STAGE = "generate_infrence" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def prediction_model(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    model_dir = os.path.join(artifacts_dir, "Models")
    root_dir = os.getcwd()
    raw_data_dir = os.path.join(root_dir, "Multilabel")
    raw_img_dir = os.path.join(raw_data_dir, "photos")
    img_path = os.path.join(raw_img_dir, "image_012.jpg")

    multioutput = os.path.join(model_dir, "multioutput.h5")
    model = load_keras_model(multioutput)

    loaded_img = load_input_img(img_path)

    prediction = model.predict(loaded_img)
    a, pred= seperate_prediction(prediction)
    logging.info(f"The probabilites of these claases are {a}")
    logging.info(f"The actual prediction for the img {img_path} is {pred}")
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    
    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        prediction_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e