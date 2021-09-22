import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

# Load data from remote URL
# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_path = pathlib.Path(r"C:\Users\vadim\.keras\datasets\flower_photos")

# Count number of .jpg images
image_count = len(list(data_path.glob('*/*.jpg')))
print(image_count)

# Define ML batch size (how many things it processes at once), and target image size
batch_size = 32
img_height = 180
img_width = 180

# Load 80% of data as a training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Load 20% of data as a testing dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Get names of things we can classify
class_names = train_ds.class_names
print(class_names)

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
# plt.show()

AUTOTUNE = tf.data.AUTOTUNE

# Shuffle data and prefetch it into memory for preformance reasons
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Color is on a scale from 0-255, not good for a NN so we scale it down to between 0-1
normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)

# Distort input data randomly to prevent overfitting
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal",
                                                     input_shape=(img_height,
                                                                  img_width,
                                                                  3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)

# Number of classificiation classes
num_classes = len(class_names)

# Define model
model = Sequential([
    data_augmentation,
    # layers.experimental.preprocessing.Rescaling(1. / 255),
    normalization_layer,
    # Aggressively spam layers, IDK what these do its hard math I just copied it from sample
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    # Dropout layer to help prevent overfitting
    layers.Dropout(0.2),
    # ????
    layers.Flatten(),
    # More calculus I don't really get
    layers.Dense(128, activation='relu'),
    # Output layer with correct number of outputs
    layers.Dense(num_classes)
])

# Compile the model with stuff IDK what any of this is
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Print a summary
model.summary()

# How many times to run the model during training
epochs = 15

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Save model
model.save("trained_model")

# Get statistics for graphing later
# Model training accuracy
acc = history.history['accuracy']
# Model testing accuracy
val_acc = history.history['val_accuracy']

# Model training loss
loss = history.history['loss']
# Model testing loss
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# Create graph of training and validation accuracy and loss
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
