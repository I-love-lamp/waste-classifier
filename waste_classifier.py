import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2                                                                                                                                
import warnings
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.utils import image_dataset_from_directory, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from tensorflow import convert_to_tensor
from glob import glob

warnings.filterwarnings('ignore')

train_path = 'C://Users//QF626RJ//OneDrive - EY//Documents//Projects//waste-classifier//DATASET//DATASET//TRAIN'
test_path = 'C://Users//QF626RJ//OneDrive - EY//Documents//Projects//waste-classifier//DATASET//DATASET//TEST'

# visualize dataset
x_data = []
y_data = []

def create_dataset(path):
    for category in glob(path + '/*'):
        for file in glob(category + '/*'):
            img_array = cv2.imread(file)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            x_data.append(img_array)
            y_data.append(category.split('\\')[-1])

    data = pd.DataFrame({'image': x_data, 'label': y_data})
    return data

# create training and test datasets 
train = create_dataset(train_path)
test = create_dataset(test_path)

# build a CNN to classify images
input_shape = (224,224,3)
model = Sequential()

# first convolutional layer
model.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu'))
model.add(MaxPooling2D())

# second convolutional layer
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D())

# flatten
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Output layer
num_classes = len(glob(train_path + '/*'))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
batch_size = 256

# prepare dataset for training
train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_generator =train_gen.flow_from_directory(train_path,
                                                batch_size=batch_size,
                                                target_size=(224,224),
                                                color_mode='rgb',
                                                class_mode='categorical')

test_generator =test_gen.flow_from_directory(test_path,
                                                batch_size=batch_size,
                                                target_size=(224,224),
                                                color_mode='rgb',
                                                class_mode='categorical')
# early stopping 
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# train the model
img_classifier = model.fit(train_generator, epochs=30, validation_data=test_generator, workers=-1)

# predictions from the classifier
def predict_waste_material_from_img(img):
    # plot image
    plt.figure(figsize=(6,4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [-1, 224, 224,3])
    result = np.argmax(model.predict(img))
    prediction = ''

    print(model.predict(img))
    print("Result ---> ", result)
    if result == 0:
        prediction = 'R'
    elif result == 1:
        prediction = 'O'
    
    return prediction
