import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()#jab humney data set ko load kiya
#to its returns two tuples dataset one for testing and one for training
train_images, test_images = train_images / 255.0, test_images / 255.0#image pixel ki hoti hai and each pixel color will
#vary so to make things simpler we devide the pixel color number by 255 so if for example a pixel has a value of 240
#it would become 0.941 after 240/255
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
# IMG_INDEX = 88
# plt.imshow(train_images[IMG_INDEX])
# plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
# plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))#we have created the first ye asal mei 32 nhi 2 pixel padding kei hai
#hidden layer whichs has 32 filters of 3 by 3 size and the input shape will be 32, 32, 3 humein apney model ko input size
#batana perta hai takey woh apney wieght adjust ker sakey imagine the model is prepared to handle greyscale images
#and you feed it RGB images the models won't work then
model.add(layers.MaxPooling2D((2, 2)))#this is a pooling layer jo kei hamarey feature map pey max pooling apply keregi
#yani kei max values of the array or matrix will be filter out meaning if a matrix ix |2.3 0.2|
#                                                                                     |1.3 4.4| we will get 4.4 then
#relu is the activation function
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
# model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=4,
                    validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)