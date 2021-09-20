from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

# To load the model look at:
# https://www.tensorflow.org/lite/guide/android

# Also, technically, this is all wrong. This is an image classification algorithm, we want an object detector,
# Which does exist at tflite_model_maker.object_detector, but has some different data format we need to look into.

# Load input data specific to an on-device ML app.
data = DataLoader.from_folder(r"C:\Users\vadim\.keras\datasets\flower_photos")
train_data, test_data = data.split(0.9)

# Customize the TensorFlow model.
model = image_classifier.create(train_data,
                                epochs=15,
                                batch_size=32,
                                dropout_rate=0.2,
                                shuffle=True)

# Evaluate the model.
loss, accuracy = model.evaluate(test_data)

# Export to Tensorflow Lite model and label file in `export_dir`.
model.export(export_dir='./tflite/')
