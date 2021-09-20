from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

# To load the model look at:
# https://www.tensorflow.org/lite/guide/android

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
