from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import Model, datasets
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import argparse
from preprocess import preprocess_dataset

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="f",
	help="f, m, c, or cc for fashion, mnist, cifar10, or cifar100. leave blank \
	for fashion.")
ap.add_argument("-e", "--num_epochs", type=int, default=2, 
	help="how many epochs do you want to train for?")
ap.add_argument("-b", "--batch_size", type=int, default=200, 
	help="how many samples per batch?")
args = vars(ap.parse_args())
    
class MyModel(Model):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation="relu")
        self.mp1 = MaxPooling2D(2)
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(num_classes, activation="softmax")
    
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

#preprocess dataset
dataset = args["dataset"]
if dataset == "cc":
    num_classes = 100
else:
    num_classes = 10
batch_size = args["batch_size"]
train_data, test_data = preprocess_dataset(dataset, batch_size)

#create an instance of the model
model = MyModel(num_classes)

#choose loss function and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

#metrics
train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        #training=True is only needed if there are layers with different
        #behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
    
@tf.function
def test_step(images, labels):
    #training=False is only needed if there are layers with different
    #behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS = args["num_epochs"]

for epoch in range(EPOCHS):
    #reset metrics at start of each epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    
    for train_images, train_labels in train_data:
        train_step(train_images, train_labels)
    
    for test_images, test_labels in test_data:
        test_step(test_images, test_labels)
        
    template = 'Epoch {}, \nTrain Loss: {}, Train Accuracy: {}%, \nTest Loss: {}, Test Accuracy: {}%\n'
    print(template.format(epoch+1,
                          round(float(train_loss.result())),
                          round(float(train_accuracy.result()*100), 3),
                          round(float(test_loss.result())),
                          round(float(test_accuracy.result()*100), 3)))