# Audio_Classification
In this project we will create a model to classify the audios, the dataset that we will be using for this project is taken from 
https://urbansounddataset.weebly.com/urbansound8k.html. This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, 
car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music. Firstly, we will perform Exploratory Data Analysis 
on the data inorder to understand the various characteritics, then we will perform Data Preprocessing to convert 
the data which can be used to develop a model which can classify the audios and finally we will create the model based on the requirements and evaluate its performance.

# Important Libraries required 
- import numpy as np
- import pandas as pd
- import matplotlib.pyplot as plt
  %matplotlib inline

- import os
- from scipy.io import wavfile as wav
- import IPython.display as ipd
- import librosa
- import librosa.display
- from tqdm import tqdm
- from sklearn.model_selection import train_test_split
- from sklearn.preprocessing import LabelEncoder
- import tensorflow as tf
- from tensorflow.keras.utils import to_categorical
- from tensorflow.keras.models import Sequential
- from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
- from tensorflow.keras.optimizers import Adam
- from sklearn import metrics
