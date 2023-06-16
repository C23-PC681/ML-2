import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

data_dir = '../capstone/imagedata' # call data directory
classes = os.listdir(data_dir)

height = 128 # image height
width = 128 # image width

with open('workout_label.txt', 'w') as f:
    for workout_class in classes:
        f.write(f'{workout_class}\n')

data = []
labels = []   

for dirname, _, filenames in os.walk(data_dir):
    data_class = dirname.split(os.path.sep)[-1]
    for filename in filenames:
        img_path = os.path.join(dirname, filename)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height , width))
        
        data.append(image)
        labels.append(classes.index(data_class))
        
data = np.array(data)
labels = np.array(labels)

labels = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
train_data = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    zoom_range = 0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

train_data.fit(X_train)

test_data = ImageDataGenerator(rescale = 1./255)

test_data.fit(X_test)

def create_model():
    model = tf.keras.models.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(width, height, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation = 'softmax'))
    model.summary()
    return model
workout_model = create_model()

workout_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
workout_model.fit(X_train, y_train, epochs=30, batch_size=128, validation_data=(X_test, y_test))

workout_model.save('workout_model')

converter = tf.lite.TFLiteConverter.from_saved_model('./workout_model')
tflite_model = converter.convert()

with open('workout_model.tflite', 'wb') as f:
    f.write(tflite_model)
