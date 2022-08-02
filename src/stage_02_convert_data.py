import tensorflow as tf
import numpy as np
import argparse
import os
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.model import loadData
tf.data.experimental.enable_debug_mode()

STAGE = "create_model"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)


def create_model(config_path, params_path):
    # read config files
    config = read_yaml(config_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    prepare_data_dir = os.path.join(artifacts_dir, artifacts["PREPARED_DATA_DIR"])

    root_dir = os.getcwd()
    raw_data_dir = os.path.join(root_dir, artifacts["RAW_DATA_DIR"])
    raw_img_dir = os.path.join(raw_data_dir, artifacts["RAW_IMAGE_DIR"])

    cleaned_csv_file = os.path.join(prepare_data_dir, artifacts["CLEANED_CSV_FILE"])

    training_data_dir = os.path.join(artifacts_dir, artifacts["TRAINING_DATA_DIR"])
    create_directories([training_data_dir])
    label_np_file = os.path.join(training_data_dir, artifacts["LABEL_TRAINING_DATA"])
    img_np_file = os.path.join(training_data_dir, artifacts["IMG_TRAINING_DATA"])
    model_dir = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    multioutput_model = os.path.join(model_dir, artifacts["BASE_MODEL_NAME"])

    # Creating an object for prepData Class
    predata = loadData(img_dir=raw_img_dir, cleaned_csv_location=cleaned_csv_file, resnet_50_model_location=multioutput_model)

    if not os.path.exists(img_np_file):
        logging.info("Training data does not found creating one at location {}".format(training_data_dir))
        raw_x_y_combined = predata.load_img_and_cleaned_label()

        x, y = predata.separate_x_y_from_raw_d(raw_x_y_combined)

        np.save(label_np_file, y, allow_pickle=True)
        np.save(img_np_file, x, allow_pickle=True)
        logging.info(f"Training data has been saved at location {training_data_dir}")
    else:
        logging.info("Training data found, loading it from {}".format(training_data_dir))


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
