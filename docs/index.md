# Multivariate Classification
Multivariate Classification

## 📝 Description
- This implementation is based on official **resnet50** 
- In this project we have used **Pretrained Model** and **tensorboard** for image classification and checking the accuracy of the model.


## STEPS -


### STEP 01- Create a conda environment after opening the repository in VSCODE

```bash
conda create --prefix ./env python=3.8.13 -y
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

### STEP 03- initialize the dvc project
```bash
dvc init
```
It will run for 100 Epochs. You can change this into params.yaml section for the number of records.

- This repository represents **" MultiVariate Classification  "**.
- With the help of this project we can Classifiy 4 Attributes of An Image .
  

## ⏳ Dataset
- Download the dataset for custom training
- https://drive.google.com/file/d/1mV7EP-maKTNu2RNv6wYRnaoON9dOqLNt/view?usp=sharing


## :desktop_computer:	Installation


### :hammer_and_wrench: Requirements
* Python 3.8+
* Tensorflow 2.9.1
* Keras 
* Pandas 
* Numpy 
* Os 




## :gear: Setup
1. Create virtual environment.
```bash
$ conda create --prefix ./env python=3.8.13 -y
```
2. Activate virtual enviroment. 
```bash
conda activate ./env
```
OR
```bash
$ source activate ./env
```

4. Run setup.py 
```bash
$ pip install -e.
```
5. Initialize `DVC` 
```
$ dvc init
```
6. Run All the steps for DVC
```
$ dvc repro
```
After that our model will trained on the given dataset for 10 epochs.
We can modify the parameters in the `params.yaml` file in the root 
directory of the folder. 
## 🎯 Inference demo

1. Testing with **Images** (Put test inages in anywhere and give the location of this image to **img_path** parameter inside prediction model function in src/infrence.py file)

![infrence_example](https://github.com/sanjeev49/aiClassification/blob/master/docs/img/infrence_example2.png)

```bash
$ python src/infrence.py 

```
In img_path give the path of the image that you want to get prediction. 

## To run Tensorboard 

```bash
tensorboard --logdir ./logs
```
## Data Augmentation
For Data Augmentation We can use
* changes in Angels, Rotation and lighting
* Changes in Lighting and direction + Flipping about the vertical axis
* Flipping about the horizontal axis and rotating by 90, 180, 270 degress. 

## Contributor <img src="https://raw.githubusercontent.com/TheDudeThatCode/TheDudeThatCode/master/Assets/Developer.gif" width=35 height=25> 
- Sanjeev Kumar