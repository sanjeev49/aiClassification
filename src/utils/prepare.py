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


root_dir = os.getcwd()
chal = os.path.join(root_dir , "Multilabel")

img_dir = os.path.join(chal, "photos")

def updated_label_img_name(label_file: list) -> list:
    with open(label_file, 'r') as f:
        f = f.readlines()
    updated_list = []
    for i in f:
        
        splitted_val = i.split()
        img_var = splitted_val[0]
        img_raw_name, jpg_raw_name = img_var.split(".")[0], img_var.split(".")[1]
        img_, img_num = img_raw_name.split("_")[0], img_raw_name.split("_")[1]
        if len(img_num) == 1:
            updated_img_num = ".".join(["_".join([img_, img_num.rjust(3, '0')]), jpg_raw_name])
            splitted_val[0] = updated_img_num
            updated_list.append(splitted_val)
        if len(img_num) == 2:

            updated_img_num2 = ".".join(["_".join([img_, img_num.rjust(3, '0')]), jpg_raw_name])
            splitted_val[0] = updated_img_num2
            updated_list.append(splitted_val)
        if len(img_num) == 3:
            updated_list.append(splitted_val)
    return updated_list

def rename_img_before_0_99(folder_path):
    for img in os.listdir(folder_path):
        path = os.path.join(folder_path, img)
        img_, acc = path.split("_")[0], path.split("_")[1]
        num_, replace  = acc.split(".")[0], acc.split(".")[1]
        if len(num_)==1:
            # getting old name 
            replaced_num = num_.rjust(3, '0')
            new_half_name = ".".join([replaced_num , replace])
            new_full_name = "_".join([img_, new_half_name])
            os.rename(path, new_full_name)
        if len(num_) == 2:
            replaced_num = num_.rjust(3, '0')
            new_half_name = ".".join([replaced_num , replace])
            new_full_name = "_".join([img_, new_half_name])
            os.rename(path, new_full_name)


def fix_wrong_img_text_index(img_list , text_img_list) -> None:
    for i in range(970):
        if img_list[i] != text_img_list[i][0]:
            del text_img_list[i]
    return text_img_list


def save_to_csv(li: list, Location: str):
    """
    this function will take the list and convert it to the DataFrame and save it at location given.
    """
    val_without_img_name = []
    for i in li:
        val_without_img_name.append([i[1], i[2], i[3], i[4]])
    
    df = pd.DataFrame(val_without_img_name, columns=["a",'b','c','d'])
    df.to_csv(Location, index=None)

def handle_nan_value(path: os.path, save_cleaned_csv_location: os.path) -> pd.DataFrame :
    """this function will take the dataframe and returns a cleaned dataframe """
    df = pd.read_csv(path)

    df.a = df.a.replace("NA", np.nan)
    df.b = df.b.replace("NA", np.nan)
    df.c = df.c.replace("NA", np.nan)
    df.d = df.d.replace("NA", np.nan)

    for i in df.columns:
        logging.info("Nulll value in column {} is {}: ".format(df[i].name , df[i].isnull().sum()))
        logging.info("***********************")
        logging.info("Filling all the missing values with max, in column {} The max value is {}".format(df[i].name,df[i].value_counts().index[0]))
        logging.info("Filling values with for column {} ".format(df[i].name))
        df[i] =  df[i].fillna(df[i].value_counts().index[0])
    logging.info("Dealt with missing values in the data")

    a = df.isnull().sum()
    logging.info("No values in the columns are {} {} {} {} null now".format(a[0], a[1], a[2], a[3]))

    df.to_csv(save_cleaned_csv_location, index = None, header= None)
 

    

