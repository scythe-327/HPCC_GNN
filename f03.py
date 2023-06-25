import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def resize_image(image, target_size):
    image = tf.image.resize(image, target_size)
    return image

def crop_fill(image, target_size):
    image_shape = tf.shape(image)[:2]
    target_height, target_width = target_size

    # Calculate the scaling factor for resizing
    scale_factor = tf.cast(target_height / image_shape[0], tf.float32)

    # Resize the image maintaining aspect ratio
    new_width = tf.cast(scale_factor * tf.cast(image_shape[1], tf.float32), tf.int32)
    image = resize_image(image, (target_height, new_width))

    # Pad or crop the image to match the desired size
    image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)

    return image

def stretch_shrink(image, target_size):
    return resize_image(image, target_size)

def fit(image, target_size):
    image_shape = tf.shape(image)[:2]
    target_height, target_width = target_size

    # Calculate the scaling factors for resizing
    scale_factor_h = tf.cast(target_height / image_shape[0], tf.float32)
    scale_factor_w = tf.cast(target_width / image_shape[1], tf.float32)
    scale_factor = tf.minimum(scale_factor_h, scale_factor_w)

    # Resize the image maintaining aspect ratio
    new_height = tf.cast(scale_factor * tf.cast(image_shape[0], tf.float32), tf.int32)
    new_width = tf.cast(scale_factor * tf.cast(image_shape[1], tf.float32), tf.int32)
    image = resize_image(image, (new_height, new_width))

    # Pad or crop the image to match the desired size
    image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)

    return image

def fitv(image, target_size):
    image_shape = tf.shape(image)[:2]
    target_height, target_width = target_size

    # Calculate the scaling factor for resizing
    scale_factor = tf.cast(target_height / image_shape[0], tf.float32)

    # Resize the image maintaining aspect ratio
    new_width = tf.cast(scale_factor * tf.cast(image_shape[1], tf.float32), tf.int32)
    image = resize_image(image, (target_height, new_width))

    # Pad or crop the image to match the desired size
    image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)

    return image

def fith(image, target_size):
    image_shape = tf.shape(image)[:2]
    target_height, target_width = target_size

    # Calculate the scaling factor for resizing
    scale_factor = tf.cast(target_width / image_shape[1], tf.float32)

    # Resize the image maintaining aspect ratio
    new_height = tf.cast(scale_factor * tf.cast(image_shape[0], tf.float32), tf.int32)
    image = resize_image(image, (new_height, target_width))

    # Pad or crop the image to match the desired size
    image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)

    return image

def process_image(image_array, method, target_size):
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

    switcher = {
        "crop_fill": crop_fill,
        "stretch_shrink": stretch_shrink,
        "fit": fit,
        "fitv": fitv,
        "fith": fith
    }

    func = switcher.get(method, None)
    if func:
        image_tensor = func(image_tensor, target_size)
    else:
        print("Invalid method!")

    return image_tensor

# User inputs
image_paths = ["C:/Users/rohan/Desktop/uktkjxhcb/sampleFormats/902587.jpg",
               "C:/Users/rohan/Desktop/uktkjxhcb/sampleFormats/902587.jpg"]

method = "crop_fill"  # Replace with your desired method
target_height = 100  # Replace with your desired target height
target_width = 100  # Replace with your desired target width
target_size = (target_height, target_width)

image_arrays = []
for image_path in image_paths:
    image = plt.imread(image_path)
    image_arrays.append(image)

image_tensors = []
for image_array in image_arrays:
    image_tensor = process_image(image_array, method, target_size)
    image_tensors.append(image_tensor)

image_tensor = tf.stack(image_tensors)

print("Final image tensor shape:", image_tensor.shape)
