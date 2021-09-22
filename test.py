import pathlib

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load some sunflower dataset
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

# You saw these in the other file
batch_size = 32
img_height = 180
img_width = 180

# Load the model
model = keras.models.load_model('./trained_model')

# Load an image from the dataset and resize appropriately
img = keras.preprocessing.image.load_img(
    sunflower_path, target_size=(img_height, img_width)
)

# Convert it to something the model understands
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

# Ask the model to predict what it thinks the image is
predictions = model.predict(img_array)
# Get the confidence level of the most likely item
score = tf.nn.softmax(predictions[0])

# This is literally just so we can get the class names of the training
data_path = pathlib.Path(r"C:\Users\vadim\.keras\datasets\flower_photos")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
class_names = train_ds.class_names

# Helpful print message to show what the model thinks the image is (Should be sunflower with very high confidence)
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
)
