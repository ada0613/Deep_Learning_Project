import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE
    x_train = np.zeros((0,3072)) #50000,3072
    y_train = np.zeros(0) #50000
    for i in range(5):
        with open(os.path.join(data_dir,'data_batch_'+str(i + 1)),'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
            x_train = np.vstack((x_train,data[b'data']))
            y_train = np.hstack((y_train,data[b'labels']))

    with open(os.path.join(data_dir,'test_batch'),'rb') as f:
        data = pickle.load(f,encoding='bytes')
        x_test = data[b'data']
        y_test = np.array(data[b'labels'])
    ### END CODE CODE HERE
    #check size before return
    # assert x_train.shape == (50000, 3072)
    # assert y_train.shape == (50000,)
    # assert x_test.shape == (10000, 3072)

    ### END CODE HERE

    return x_train, y_train, x_test, y_test


def load_testing_images(private_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 3072].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    # private_dir = os.path.join(private_dir, "Private")
    x_test = np.load(os.path.join(private_dir, 'private_test_images_2022.npy'))


    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, split_index=45000):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE
    # split_index = int(y_train.shape[0] * train_ratio)
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

