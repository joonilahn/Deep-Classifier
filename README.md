# Deep Classifier - PyTorch based image classification model trainer

## Introduction
deep_classifier is a PyTorch based image classification model trainer.

You can train your own image classification model using a built-in config file which is a yacs based yaml file.

## Requirements
- yacs
- efficientnet_pytorch
- Pillow
- tensorboardX
- tqdm
- matplotlib
- pandas
- seaborn

## Supported Models
- Most of models are imported from torchvision.models.

- [x] ResNet
- [x] DenseNet
- [x] Inception-v3
- [x] Inception-v4
- [x] MobileNet
- [x] PNASNet
- [x] EfficientNet

## How to train your own model
- Let's say your root directory for your dataset is 'dataset/'.
- You have to make sub-directories under the root directory to store images with different class.
- If you have dog-cat images, then you should create 'dataset/0/' for one class, and 'dataset/1/' for another.
- Then make a yaml file and set all the hyperparameters to train a model. See default.py(deep_classifier/config/default.py)  file for the details.
- Run train.py
```
python train.py {path to config file}
```

- If your want to reduce usage of gpu memory or training time, consider using amp. (torch>=1.5)
```
python train.py {path to config file} --fp16
```
