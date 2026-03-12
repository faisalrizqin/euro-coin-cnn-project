import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

train_path = "dataset/coins-dataset/classified/train"
test_path = "dataset/coins-dataset/classified/test"

img_size = 128
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(
    rescale = 1./255
)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size = (img_size, img_size),
    batch_size = batch_size,
    class_mode = 'categorical'
)

test_data = test = test_datagen.flow_from_directory(
    test_path,
    target_size = (img_size, img_size),
    batch_size = batch_size,
    class_mode = 'categorical'
)

print("Classes:", train_data.class_indices)

model = Sequential()

# Convolution layer 1
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (img_size, img_size, 3)))
model.add(MaxPooling2D(2, 2))

# Convolution layer 2
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(2, 2))

# Convolution layer 3
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(2, 2))

# Convolution layer 4
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(2, 2))

# Flatten
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))

# Output layer (8 kelas koin)
model.add(Dense(8, activation = 'softmax'))

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.summary()

history = model.fit(
    train_data,
    epochs = 30,
    validation_data = test_data
)

model.save("model/coin_model.h5")

print("Model berhasil disimpan!")