import tensorflow as tf
from tensorflow.keras import datasets

def preprocess_dataset(ds, bs):
    #read dataset from tensorflow.keras.datasets
    if ds == "f":
        (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
        print("You are training and testing on the fashion mnist dataset.")
    elif ds == "m":
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        print("You are training and testing on the mnist dataset.")
    elif ds == "c":
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        print("You are training and testing on the cifar10 dataset.")
    elif ds == "cc":
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
        print("You are training and testing on the cifar100 dataset.")
    else:
        raise Exception("the dataset options are fashion mnist, mnist, cifar10, and cifar100")
    
    #normalize pixel values from 0-255 to 0-1
    x_train, x_test = x_train / 255., x_test / 255.

    
    #add channels dimension (size of 1) if grayscale dataset
    if "c" not in ds:
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    
    #shuffle train set and create batch datasets for train and test
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(bs)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(bs)
    
    return (train_ds, test_ds)