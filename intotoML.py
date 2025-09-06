from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
train_path = tf.keras.utils.get_file(#this file is for training our model
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(#this is file is for testing our model if we feed the same data that we used to test the model
    # would score great as it as seen the data before but would fail if we give it unseen data
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
#pandas data website mein sey read kerney ke liye aur tensor slow us data ko model ke liye ready kerney ke liye
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
train_y = train.pop('Species')#we remove the label of data so the model have to guess that label from the data
test_y = test.pop('Species')
def input_fn(features, labels, training=True, batch_size=256):
# ye function hamarey data ko tayyar kerta hai kei we can use it to train our model
# we don't give our model direct data but rather we give it a function for better and efficient training
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

model = keras.Sequential([
    layers.Dense(30, activation='relu', input_shape=(train.shape[1],)),
    layers.Dense(10, activation='relu'),
    layers.Dense(3, activation='softmax')   # n_classes = 3
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',   # since labels are integers (0,1,2)
    metrics=['accuracy']
)
model.fit(train, train_y, epochs=50, batch_size=32, verbose=1) #model is now trained
loss, accuracy = model.evaluate(test, test_y, verbose=0)
sepal_length = float(input("Enter Sepal Length: "))
sepal_width = float(input("Enter Sepal Width: "))
petal_length = float(input("Enter Petal Length: "))
petal_width = float(input("Enter Petal Width: "))
sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
predictions = model.predict(sample)
predicted_class = np.argmax(predictions[0])
predicted_species = SPECIES[predicted_class]
probability = predictions[0][predicted_class] * 100
print(f"Predicted species: {predicted_species}")
print(f"Confidence: {probability:.2f}%")
