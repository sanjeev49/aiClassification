import logging
import os
import pandas as pd
import numpy as np

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)


class prepareData:

    def __init__(self, img_dir: str, label_file: str, csv_w_nan_location: str, cleaned_csv_location: str):
        self.img_dir_name = img_dir
        self.img_dir_list = os.listdir(img_dir)
        self.label_file = label_file
        self.csv_w_nan_location = csv_w_nan_location
        self.cleaned_csv_location = cleaned_csv_location

    def updated_label_img_name(self) -> list:
        """
        Method Name: updated_label_img_name
        Description: This function take label_file input and rename a column inside it. Ex: image_1.jpg to image_001.jpg
        Output: Updated List if Images in the label file
        On Failure: FileNotFoundError, Exception
        Written By: Sanjeev Kumar
        Version: 1.0
        Revisions: None
        """
        try:
            with open(self.label_file, 'r') as f:
                label_list = f.readlines()
            updated_list = []
            for i in label_list:
                split_val = i.split()
                img_var = split_val[0]
                img_raw_name, jpg_raw_name = img_var.split(".")[0], img_var.split(".")[1]
                img_, img_num = img_raw_name.split("_")[0], img_raw_name.split("_")[1]
                if len(img_num) == 1:
                    updated_img_num = ".".join(["_".join([img_, img_num.rjust(3, '0')]), jpg_raw_name])
                    split_val[0] = updated_img_num
                    updated_list.append(split_val)
                if len(img_num) == 2:
                    updated_img_num2 = ".".join(["_".join([img_, img_num.rjust(3, '0')]), jpg_raw_name])
                    split_val[0] = updated_img_num2
                    updated_list.append(split_val)
                if len(img_num) == 3:
                    updated_list.append(split_val)
            logging.info(f"All image name in {self.label_file} between 0 and 99 are renamed to correct format.")
            return updated_list
        except FileNotFoundError:
            logging.info(f"The file {self.label_file} is not available. ")
            raise FileNotFoundError
        except Exception:
            logging.info(f"Something bad happens \n {Exception}")
            raise Exception
        finally:
            f.close()

    def rename_img_before_0_99(self) -> None:
        """
        Function Name: rename_img_before_0_99
        Description: This function take img folder paths input and rename images between 0 and 99.
                    Ex:- image_1.jpg to image_001.jpg
        Output: None
        On Failure: Raise FileNotFoundError, Exception
        Written By: Sanjeev Kumar
        Version: 1.0
        Revisions: None
        """
        try:
            for img in self.img_dir_list:
                path = os.path.join(self.img_dir_name, img)
                img_, acc = path.split("_")[0], path.split("_")[1]
                num_, replace = acc.split(".")[0], acc.split(".")[1]
                if len(num_) == 1:
                    # getting old name
                    replaced_num = num_.rjust(3, '0')
                    new_half_name = ".".join([replaced_num, replace])
                    new_full_name = "_".join([img_, new_half_name])
                    os.rename(path, new_full_name)
                if len(num_) == 2:
                    replaced_num = num_.rjust(3, '0')
                    new_half_name = ".".join([replaced_num, replace])
                    new_full_name = "_".join([img_, new_half_name])
                    os.rename(path, new_full_name)
            logging.info(f"All image file in {self.img_dir_name} folders are renamed with correct format. ")
        except FileNotFoundError:
            logging.info(f"The file {self.img_dir_name} is not available. ")
            raise FileNotFoundError
        except Exception:
            logging.info(f"Something wrong happens")
            raise Exception

    def fix_wrong_img_text_index(self, updated_label_img_list: list) -> list:
        """
        Function Name: fix_wrong_img_text_index
        Description: This function delete the row from label_text file which doesn't match with image_name
                        present in the image dir.
        Output: updated_label_img_list
        On Failure: Exception
        Written By: Sanjeev Kumar
        Version: 1.0
        Revisions: None
        """
        try:
            for i in range(970):
                if self.img_dir_list[i] != updated_label_img_list[i][0]:
                    del updated_label_img_list[i]
            logging.info("All incorrect index position with compare to image and index are fixed. ")
            return updated_label_img_list
        except Exception:
            logging.info(f"Exception is {Exception}")
            raise Exception

    def save_to_csv(self, updated_label_img_name: list) -> None:
        """
        Function Name: save_to_csv
        Description: This function takes a list of label columns and remove the image name from it and save it as a csv
                        file to given location. Ex: (image_001, 1,0,0,1) to (1,0,0,1).
        Output: None
        On Failure: Raise FileNotFoundError, Exception
        Written By: Sanjeev Kumar
        Version: 1.0
        Revisions: None
        """
        try:
            val_without_img_name = []
            for i in updated_label_img_name:
                val_without_img_name.append([i[1], i[2], i[3], i[4]])

            df = pd.DataFrame(val_without_img_name, columns=["a", 'b', 'c', 'd'])
            df.to_csv(self.csv_w_nan_location, index=False)
            logging.info(f"Image name removed from file and saved it as DataFrame at {self.csv_w_nan_location}")
        except FileNotFoundError:
            logging.info(f"File not found at {self.csv_w_nan_location}")
            raise FileNotFoundError
        except Exception:
            logging.info(f"something gone wrong {Exception}")
            raise Exception

    def handle_nan_value(self) -> None:
        """
        Method Name: handle_nan_value
        Description: This function take a dataframe and replace all the NA values of it  with max in the column and
                        save clean data to directory given.
        Output: Cleaned Data
        On Failure: Raise ValueError,  KeyError,Exception
        Written By: Sanjeev Kumar
        Version: 1.0
        Revisions: None
        """
        df = pd.read_csv(self.csv_w_nan_location)

        df.a = df.a.replace("NA", np.nan)
        df.b = df.b.replace("NA", np.nan)
        df.c = df.c.replace("NA", np.nan)
        df.d = df.d.replace("NA", np.nan)

        for i in df.columns:
            logging.info("Null value in column {} is {}: ".format(df[i].name, df[i].isnull().sum()))
            logging.info("***********************")
            logging.info("Filling all the missing values with max, in column {} The max value is {}".format(df[i].name
                                                                                    ,df[i].value_counts().index[0]))
            logging.info("Filling values with for column {} ".format(df[i].name))
            df[i] = df[i].fillna(df[i].value_counts().index[0])
        logging.info("Dealt with missing values in the data")

        a = df.isnull().sum()
        logging.info("Checking Null found {} {} {} {} values.".format(a[0], a[1], a[2], a[3]))

        df.to_csv(self.cleaned_csv_location, index=False, header=None)
