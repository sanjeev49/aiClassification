

- This repository represents **" MultiVariate Classification  "**.
- With the help of this project we can Classifiy 4 Attributes of An Image .
  
## 📝 Description
- This implemantation is based on official **resnet50** 
- In this project we have used **Pretrained Model** and **tensorboard** for image classification and checking the accuracy of the model.

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
1. Create virtual enviroment
```bash
$ conda create --prefix ./env python=3.8.13 -y
```
2. Activate conda enviroment 
```bash
$ conda activate ./env
```

3. Install Required libraries
```bash
$ pip install requirements.txt
```

4. Run setup.py 
```bash
$ pip install -e.
```
5. Run src/infrence.py To get the prediction.

## 🎯 Inference demo

1. Testing with **Images** (Put test inages in anywhere and give the location of this image to **img_path** parameter inside prediction model function in src/infrence.py file)

![infrence_example](https://github.com/sanjeev49/aiClassification/blob/master/docs/img/infrence_example.png)

```bash
$ python src/infrence.py 

```
In img_path give the path of the image that you want to get prediction. 

## To run Tensorboard 

```bash
tensorboard --logdir ./logs
```


## Contributors <img src="https://raw.githubusercontent.com/TheDudeThatCode/TheDudeThatCode/master/Assets/Developer.gif" width=35 height=25> 
- Sanjeev Kumar