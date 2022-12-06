import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2                                                                                                                                
from tdqm import tdqm
import warnings
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.utils.vis_utils import plot_model
from glob import glob

warnings.filterwarnings('ignore')

train_path = 'DATASET/TRAIN'
test_path = 'DATASET/TEST'

# visualize dataset
x_data = []
y_data = []

for category in glob(train_path + '/*'):
    for file in tdqm(glob(category + '/*')):
        img_array = cv2.imread(file)
        img_array = cv2.cvtColor(img_array, cv2.BGR2RGB)
        x_data.append(img_array)
        y_data.append(category.split('/')[-1])

data = pd.DataFrame({'image': x_data, 'label': y_data})