from tensorflow import keras
import tensorflow as tf 
tf.data.experimental.enable_debug_mode()


import argparse
import os
import logging
from src.utils.common import read_yaml
from src.utils.model import loadData


STAGE = "generate_infrence" 

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

    model_dir = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    root_dir = os.getcwd()
    raw_data_dir = os.path.join(root_dir, artifacts["RAW_DATA_DIR"])
    raw_img_dir = os.path.join(raw_data_dir, artifacts["RAW_IMAGE_DIR"])
    # Change it to any image from Multilabel/photos folder
    img_path = os.path.join(raw_img_dir, "image_968.jpg")
    # cleaned csv location
    cleaned_csv_file = os.path.join(root_dir, artifacts["CLEANED_CSV_FILE"])

    multioutput_resnet_model = os.path.join(model_dir, artifacts["BASE_MODEL_NAME"])

    # intializing object of load_data class
    predata = loadData(img_dir=raw_img_dir, cleaned_csv_location=cleaned_csv_file,
                       resnet_50_model_location=multioutput_resnet_model)

    model = predata.load_keras_model()
    loaded_img = predata.load_input_img(img_path)

    prediction = model.predict(loaded_img)
    # Generating Prob_pred for probabilites pred is using a thrshold of 0.5 to assign a label 
    pred, prob_pred = predata.separate_prediction(prediction)
    logging.info(f"The probabilites of these claases are {prob_pred}")
    logging.info(f"The actual prediction for the img {img_path} is {pred}")
    print(pred)
    

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