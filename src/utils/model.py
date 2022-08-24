from keras.applications.regnet import preprocess_input
import tensorflow as tf
from keras.utils import load_img
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import img_to_array
import logging
import os


class loadData:
    def __init__(self, img_dir: str, cleaned_csv_location: str, resnet_50_model_location: str):
        self.img_dir_name = img_dir
        self.cleaned_csv_location = cleaned_csv_location
        self.resnet_50_model_location = resnet_50_model_location
        self.img_dir_list = os.listdir(img_dir)

    def load_img_and_cleaned_label(self) -> list:
        """
        Method Name: load_img_and_cleaned_label
        Description: This Method load the image data from image folder and cleaned_csv label from cleaned_csv directory
                    and store it in a list and return the list .
        Output: List
        On Failure: Raise FileNotFoundError, Exception
        Written By: Sanjeev Kumar
        Version: 1.0
        Revisions: None
        """
        try:
            df = pd.read_csv(self.cleaned_csv_location, header=None)
            raw_data = []
            # Calling Variable of prepData
            for num, img in enumerate(self.img_dir_list):
                img_path = os.path.join(self.img_dir_name, img)
                im = load_img(img_path, color_mode="rgb", target_size=(224, 224, 3))
                im = img_to_array(im)
                im = preprocess_input(im)
                raw_data.append([im, [df.loc[num][0], df.loc[num][1], df.loc[num][2], df.loc[num][3]]])
            return raw_data
        except FileNotFoundError:
            logging.info(f"File not exist at location {self.img_dir_name}")
            raise FileNotFoundError
        except Exception as e:
            logging.info(f"Something Bad {e}")
            raise e

    @staticmethod
    def separate_x_y_from_raw_d(raw_X_y_list: list):
        """
        Method Name: separate_x_y_from_raw_d
        Description: This Method take combine list of X_data and Y_data separate them and return two numpy array
                    For X and Y.
        Output: Numpy Array of X_data and Numpy Array of Y_data
        On Failure: Raise FileNotFoundError, Exception
        Written By: Sanjeev Kumar
        Version: 1.0
        Revisions: None
        """
        try:
            x_data = []
            y_data = []
            for idx, data in enumerate(raw_X_y_list):
                x_data.append(raw_X_y_list[idx][0])
                y_data.append(raw_X_y_list[idx][1])

            x_data = np.array(x_data)
            y_data = np.array(y_data)
            return x_data, y_data
        except NameError:
            logging.info(f"List {raw_X_y_list} is not defined.")
            raise NameError
        except Exception as e:
            logging.info("Something Went Wrong.")
            raise e

    @staticmethod
    def separate_label_into4_col(label_combined_array: list):
        """
        Method Name: separate_label_into4_col
        Description: This method take combine Numpy array of Label data and return single array for every colum.
        Output: Numpy Array of X_data and Numpy Array of Y_data
        On Failure: Raise FileNotFoundError, Exception
        Written By: Sanjeev Kumar
        Version: 1.0
        Revisions: None
        """
        try:
            a, b, c, d = [], [], [], []
            for i in label_combined_array:
                a.append(i[0]), b.append(i[1]), c.append(i[2]), d.append(i[3])
            return np.array(a), np.array(b), np.array(c), np.array(d)
        except Exception:
            logging.info("Some Ex`ception happens .")
            raise Exception

    @staticmethod
    def download_resent_model():
        """
        Method Name: download_resent_model
        Description: This method Download And returns the Resnet Model with the parameter include_top = False,
                    Uses average pooling and uses weights of the imagenet.
        Output: Tensorflow Model
        On Failure: ConnectionError, Exception
        Written By: Sanjeev Kumar
        Version: 1.0
        Revisions: None
        """
        try:
            resnet_base = tf.keras.applications.ResNet50(include_top=False, pooling='avg', weights='imagenet')
            for layer in resnet_base.layers:
                layer.trainable = False
            return resnet_base
        except ConnectionError:
            logging.info("Please Check Your internet Connection")
            raise ConnectionError
        except Exception:
            logging.info(f"Something Bad Happens. {Exception}")
            raise Exception

    def load_keras_model(self)-> list:
        """
        Method Name: load_keras_model
        Description: This method Loads Keras Model from the given file and return the tensorflow model
        Output: Tensorflow Model
        On Failure: ConnectionError, Exception
        Written By: Sanjeev Kumar
        Version: 1.0
        Revisions: None
        """
        try:
            model = keras.models.load_model(self.resnet_50_model_location)
            return model
        except FileNotFoundError:
            logging.info(f"The model is not present at {self.resnet_50_model_location}.")
            raise FileNotFoundError
        except Exception:
            logging.info(f"Something bad happened {Exception}")
            raise Exception

    @staticmethod
    def load_input_img(img_path: str):
        """
        Method Name: load_input_img
        Description: This method Loads image and preprocess it so that it can be utilized during model prediction
        Output: Tensorflow data format
        On Failure: ConnectionError, Exception
        Written By: Sanjeev Kumar
        Version: 1.0
        Revisions: None
        """
        try:
            img = load_img(img_path, target_size=(224, 224, 3))
            img = img_to_array(img)
            img = preprocess_input(img)
            img = img.reshape(1, img.shape[0], img.shape[1], 3)
            return img
        except FileNotFoundError:
            logging.info(f"File not found at location {img_path}")
            raise FileNotFoundError
        except Exception:
            logging.info(f"something bad happens {Exception}")
            raise Exception

    @staticmethod
    def separate_prediction(pred_array: list):
        """
        Method Name: load_input_img
        Description: This method takes the combined prediction list and apply a threshold of 0.5 on it, so it will
                    Give the output in 0 and 1 format.
        Output: 0 if prediction_prob is less or equal to than 0.5 , 1 if prediction is greater than 0.5
        On Failure: Exception
        Written By: Sanjeev Kumar
        Version: 1.0
        Revisions: None
        """
        try:
            filtered_pred = [[pred_array[0][0], pred_array[1][0], pred_array[2][0], pred_array[3][0]]]
            pred_a, pred_b, pred_c, pred_d = filtered_pred[0][0][0], filtered_pred[0][1][0], filtered_pred[0][2][0], \
                                             filtered_pred[0][3][0]

            li2 = [pred_a, pred_b, pred_c, pred_d]
            ones_and_zeros = []
            for i in li2:
                if i > 0.5:
                    ones_and_zeros.append(1)
                if i <= 0.5:
                    ones_and_zeros.append(0)
            logging.info("All the prediction probabilities are converted into 1 and zero by threshold of 0.5")
            return ones_and_zeros, filtered_pred
        except Exception:
            logging.info(f"Something bad happens at separate_prediction method. {Exception}")
            raise Exception
