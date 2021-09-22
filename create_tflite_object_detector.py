# Code from https://www.tensorflow.org/lite/tutorials/model_maker_object_detection

import numpy as np
import os

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf

# Ask tensorflow to only report errors
tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

spec = model_spec.get('efficientdet_lite4')

# Load predone data from google cloud
# In order to do this, we need to do a bunch of authorization pain that I don't have time form rn
train_data, validation_data, test_data = object_detector.DataLoader.from_csv('gs://cloud-ml-data/img/openimage/csv/salads_ml_use.csv')

# Create the model
model = object_detector.create(train_data,
                               model_spec=spec,
                               batch_size=8,
                               train_whole_model=True,
                               validation_data=validation_data)

# Test the model
model.evaluate(test_data)
model.export(export_dir='./tflite_object_detector')

# Evaluate again to make sure it exported correctly
model.evaluate_tflite('./tflite_object_detectormodel.tflite', test_data)
