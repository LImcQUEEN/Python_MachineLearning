import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)
get_label_name = metadata.features['label'].int2str  # creates a function object that we can use to get labels

# display 2 images from the dataset
for image, label in raw_train.take(5):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))
IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(img, labl):
    img = tf.cast(img, tf.float32)                # 1. Convert to float32
    img = (img/127.5) - 1                        # 2. Normalize pixels to [-1, 1]
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)) # 3. Resize to fixed size
    return img, labl


