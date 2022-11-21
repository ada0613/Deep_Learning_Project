import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    depth_major = record.reshape((3, 32, 32))
    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training) # If any.

    image = np.transpose(image, [2, 0, 1])

    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [3, 32, 32].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32]. The processed image.
    """
    ### YOUR CODE HERE
    if training:
        image = np.pad(image,((4,4),(4,4),(0,0)),'constant')
        # Randomly crop a [32, 32] section of the image.
        x = np.random.randint(0,9)
        y = np.random.randint(0,9)
        crop_image = image[x:x+32,y:y+32,:]
        # Randomly flip the image horizontally.
        horizontal_flip = np.random.randint(0,2)
        if horizontal_flip == 0:
            crop_image = np.flip(crop_image,1)
    mean = np.mean(image, axis=(0,1), keepdims=True)
    std = np.std(image, axis=(0,1), keepdims=True)
    image = (image - mean)/std
    
    return image


def visualize(image, save_name='test.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    image = image.reshape((3, 32, 32))
    image = np.transpose(image, [1, 2, 0])
    ### YOUR CODE HERE
    plt.imshow(image)
    plt.savefig(save_name)
    return image

def preprocess_test(image):
    """
    Preprocess a single private test data image
    Args:
        image: An array of shape [32, 32, 3]

    Returns:
        image: A processed image of tensor format with shape [3, 32, 32]
    """
    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])
    image = Image.fromarray(image)
    image = transform_image(image)
    return image

### END CODE HERE