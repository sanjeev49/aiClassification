import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
import argparse
import os
import logging
from src.utils.callback import save_callback
from src.utils.model import loadData
from src.utils.common import read_yaml, create_directories
tf.data.experimental.enable_debug_mode()

STAGE = "training_model"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def create_model(config_path, params_path):
    # read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    prepare_data_dir = os.path.join(artifacts_dir, artifacts["PREPARED_DATA_DIR"])
    cleaned_csv_file = os.path.join(prepare_data_dir, artifacts["CLEANED_CSV_FILE"])


    
    model_dir = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    create_directories([model_dir])

    resnet50_multioutput_model = os.path.join(model_dir, artifacts["BASE_MODEL_NAME"])

    training_data_dir = os.path.join(artifacts_dir, artifacts["TRAINING_DATA_DIR"])
    label_np_file = os.path.join(training_data_dir, artifacts["LABEL_TRAINING_DATA"])
    img_np_file = os.path.join(training_data_dir, artifacts["IMG_TRAINING_DATA"])

    root_dir = os.getcwd()
    raw_data_dir = os.path.join(root_dir, artifacts["RAW_DATA_DIR"])
    raw_img_dir = os.path.join(raw_data_dir, artifacts["RAW_IMAGE_DIR"])

    # Creating an object for prepData Class
    predata = loadData(img_dir=raw_img_dir, cleaned_csv_location=cleaned_csv_file,
                       resnet_50_model_location=resnet50_multioutput_model)

    # Loading params
    # INPUT_SHAPE = params["INPUT_SHAPE"]
    activation = params["ACTIVATION"]
    validation_split = params["VALIDATION_SPLIT"]
    loss = params["LOSS"]
    epochs = params["EPOCHS"]
    optimizer = params["OPTIMIZER"]
    batch_size = params["BATCH_SIZE"]

    try:
        x = np.load(img_np_file, allow_pickle=True)
        y = np.load(label_np_file, allow_pickle=True)
        logging.info("Training data is loaded from {}".format(training_data_dir))
    except Exception:
        logging.info("Training Data is not available at {}".format(training_data_dir))
        raise Exception
    # Separating all the columns of label.
    a, b, c, d = predata.separate_label_into4_col(y)

    # loading pretrained model resnet 50
    resnet_base = predata.download_resent_model()
    
    xinput = Input(shape=(224, 224, 3))
    inp = resnet_base(xinput)

    prediction1 = layers.Dense(1, activation=activation, name='a')(inp)
    prediction2 = layers.Dense(1, activation=activation, name='b')(inp)
    prediction3 = layers.Dense(1, activation=activation, name='c')(inp)
    prediction4 = layers.Dense(1, activation=activation, name='d')(inp)

    model = Model(xinput, [prediction1, prediction2, prediction3, prediction4])
    tb_callback = save_callback()

    logging.info(f"The summary of the model is {model.summary()}")

    model.compile(loss=[loss, loss, loss, loss], optimizer=optimizer, metrics=['accuracy'])

    model.fit(x, [a, b, c, d], epochs=epochs, validation_split=validation_split, batch_size=batch_size,  callbacks=tb_callback)

    model.save(resnet50_multioutput_model)
    logging.info(f"Model is build and saved at {resnet50_multioutput_model}")

    logging.info(f"Callbacks are at logs directory and model is at {model_dir}")
    logging.info("Training Complete")





    
    
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
