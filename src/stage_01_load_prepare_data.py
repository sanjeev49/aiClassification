import argparse
import os
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.prepare import prepareData

STAGE = "Prepare_data"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def prepare_data(config_path, params_path):
    # read config files
    config = read_yaml(config_path)
    # Initialize the Class
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    prepare_data_dir = os.path.join(artifacts_dir, artifacts["PREPARED_DATA_DIR"])
    create_directories([artifacts_dir, prepare_data_dir])

    root_dir = os.getcwd()
    raw_data_dir = os.path.join(root_dir, artifacts["RAW_DATA_DIR"])
    raw_label_file = os.path.join(raw_data_dir, artifacts["RAW_LABEL_FILE"])
    raw_img_dir = os.path.join(raw_data_dir, artifacts["RAW_IMAGE_DIR"])

    csv_w_nan = os.path.join(prepare_data_dir, artifacts["RAW_CSV_FILE"])
    cleaned_csv = os.path.join(prepare_data_dir, artifacts["CLEANED_CSV_FILE"])

    # Creating Object of prepData Class
    prepdata = prepareData(img_dir=raw_img_dir, label_file=raw_label_file, csv_w_nan_location=csv_w_nan,
                           cleaned_csv_location=cleaned_csv)

    updated_label_img_name = prepdata.updated_label_img_name()
    prepdata.rename_img_before_0_99()
    updated_label_acc_img = prepdata.fix_wrong_img_text_index(updated_label_img_list=updated_label_img_name)

    # Asserting for image_dir image_name and label_file image_name
    for img_name, text_data in zip(os.listdir(raw_img_dir), updated_label_acc_img):
        assert img_name == text_data[0]
    # Removing Image name from label file for creating training data
    prepdata.save_to_csv(updated_label_img_name=updated_label_acc_img)
    # Handling Nan Values
    prepdata.handle_nan_value()


    #
    # update_label_val = updated_label_img_name(raw_label_file)
    #
    # rename_img_before_0_99(raw_img_dir)
    #
    # a = fix_wrong_img_text_index(os.listdir(raw_img_dir), update_label_val)

    # checking for equality of image and label

    # for img, data in zip(os.listdir(raw_img_dir ), a):
    #     assert img == data[0]
    # # Save clean label to pandas dataframe to handle nan vaules
    # save_to_csv(a, csv_w_nan)
    #
    # # Handling NaN in labels
    # handle_nan_value(csv_w_nan, cleaned_csv)
    #
    # logging.info("Handles Missing values and cleaned file is at location {}".format(cleaned_csv))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    
    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        prepare_data(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
