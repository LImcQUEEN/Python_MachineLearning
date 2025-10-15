import tensorflow as tf
from  tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
fashion_mnist = keras.datasets.fashion_mnist
(trainImage, trainLabels), (testImage, testLabels) = fashion_mnist.load_data()
# print(trainImage.shape)
# print(type(trainImage))
## we are dealing with greyscale image where each pixel is between 0 - 255 where 0 represent black and 255 represent
# plt.figure()
# plt.imshow(trainImage[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()
classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
trainImage = trainImage / 255.0
testImage = testImage / 255.0
model = keras.Sequential([##humney type of layer defind kerdi aur connection bhi define kerdiye in short we have build architecture of our model
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation ='relu'),
    keras.layers.Dense(10, activation='softmax')## kiyun kei hamarey paas 10 possible classes hain of our data isliye output mein 10 neurons hongey
    ## softmax make sure that all values are between 0 and 1
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(trainImage, trainLabels, epochs=10)

# test_loss, test_acc = model.evaluate(testImage, testLabels, verbose=1)
# print("Test Accuracy:", test_acc)
predictions = model.predict(testImage)
temp = np.argmax(predictions[0])
print(classNames[np.argmax(predictions[0])])
plt.figure()
plt.imshow(trainImage[temp])
plt.colorbar()
plt.grid(False)
plt.show() 