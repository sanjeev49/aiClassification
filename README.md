![Untitled design (4)](https://user-images.githubusercontent.com/62059604/99800421-5818bf80-2b5a-11eb-83ad-c0fe6a2d48be.png)

![Untitled design (5)](https://user-images.githubusercontent.com/62059604/99800592-9e6e1e80-2b5a-11eb-8f70-4796dd0ee36a.png)

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
1. Install PyTorch :-
```bash
$ pip3 install tensorflow

```
2. Install :-
```bash
$ pip3 install pandas

```
```bash
$ pip3 install matplotlib

```

```bash
$ pip3 install -e.

```
## 🎯 Inference demo
1. Testing with **Images** ( Put test images in **AlphaPose/examples/demo/** )  :-
```bash
$ python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/ --save_img

```



## Contributors <img src="https://raw.githubusercontent.com/TheDudeThatCode/TheDudeThatCode/master/Assets/Developer.gif" width=35 height=25> 
- Sanjeev Kumar