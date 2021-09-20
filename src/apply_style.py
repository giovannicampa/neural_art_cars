#! /usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class ApplyStyle:

    # this will take a few minutes to load
    hub_module = hub.load(
        "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
    )

    def tensor_to_image(self, tensor):
        """converts a tensor to an image"""
        tensor_shape = tf.shape(tensor)
        number_elem_shape = tf.shape(tensor_shape)
        if number_elem_shape > 3:
            assert tensor_shape[0] == 1
            tensor = tensor[0]

        tensor = tf.keras.preprocessing.image.array_to_img(tensor)
        tensor = np.asarray(tensor).copy()
        return tensor

    def load_img(self, path_to_img):
        """loads an image as a tensor and scales it to 512 pixels"""
        max_dim = 512
        image = tf.io.read_file(path_to_img)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)

        shape = tf.shape(image)[:-1]
        shape = tf.cast(tf.shape(image)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        image = tf.image.resize(image, new_shape)
        image = image[tf.newaxis, :]
        image = tf.image.convert_image_dtype(image, tf.uint8)

        return image

    def load_images(self, content_path, style_path):
        """loads the content and path images as tensors"""
        content_image = self.load_img("{}".format(content_path))
        style_image = self.load_img("{}".format(style_path))

        return content_image, style_image

    def imshow(self, image, title=None):
        """displays an image with a corresponding title"""
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)

        plt.imshow(image)
        if title:
            plt.title(title)

    def show_images_with_objects(self, images, titles=[]):
        """Displays a row of images with corresponding titles"""

        if len(images) != len(titles):
            return

        plt.figure(figsize=(20, 12))
        for idx, (image, title) in enumerate(zip(images, titles)):
            plt.subplot(1, len(images), idx + 1)
            plt.xticks([])
            plt.yticks([])
            self.imshow(image, title)
