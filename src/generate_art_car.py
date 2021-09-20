#! /usr/bin/python3

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize

from apply_style import ApplyStyle
from segmentation import Segmentation

# set default images
content_path = os.path.join("images", "subjects", "bmw.jpeg")
style_path = os.path.join("images", "styles", "bauhaus.jpeg")

segmentation = Segmentation()
segmentation.read_input_image(content_path)

_ = segmentation.predict_mask()

styliser = ApplyStyle()
content_image, style_image = styliser.load_images(content_path, style_path)

content_image_array = styliser.tensor_to_image(content_image)
content_image_gray = rgb2gray(content_image_array)
mask_dark_areas = content_image_gray > 0.3

mask_dark_areas = np.vstack((mask_dark_areas[0, :], mask_dark_areas))
mask_dark_areas = np.vstack((mask_dark_areas, mask_dark_areas[-1, :]))
mask_dark_areas = np.vstack((mask_dark_areas, mask_dark_areas[-1, :]))

stylized_image = styliser.hub_module(
    tf.image.convert_image_dtype(content_image, tf.float32),
    tf.image.convert_image_dtype(style_image, tf.float32),
)[0]

mask_car = segmentation.upscale_mask_to_final_shape(
    height_final=stylized_image.shape[1], width_final=stylized_image.shape[2]
)

# convert the tensor to image
stylized_image = styliser.tensor_to_image(stylized_image)


content_image_array = resize(content_image_array, (344, 512))
content_image_array = content_image_array * 255

mask_dark_areas_car = np.logical_and(mask_car, ~mask_dark_areas)
mask_pixels_from_original = np.logical_or(mask_dark_areas_car, ~mask_car)

stylized_image[mask_pixels_from_original, :] = content_image_array[
    mask_pixels_from_original, :
]

plt.imshow(stylized_image)
plt.show()
