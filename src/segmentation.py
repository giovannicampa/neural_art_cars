#! /usr/bin/python3

import numpy as np
from keras_segmentation.pretrained import pspnet_101_voc12
from PIL import Image
from skimage.transform import rescale


class Segmentation:

    # load the pretrained model trained on ADE20k dataset

    def __init__(self):
        self.model = pspnet_101_voc12()

    def read_input_image(self, input_image_path):

        self.input_image_path = input_image_path
        self.input = Image.open(input_image_path)
        self.original = np.asarray(self.input)

    # -------------------------------------------------------------------------------
    # - Semantic segmentation
    def predict_mask(self):
        prediction = self.model.predict_segmentation(
            inp=self.input_image_path, out_fname="out" + self.input_image_path + ".png"
        )

        # Select the pixels that belong to the person
        id_car = 7  # Corresponds to cars
        self.car_mask = (prediction == np.ones_like(prediction) * id_car).astype(int)

        return self.car_mask

    def upscale_mask_to_final_shape(self, height_final, width_final):

        # How much we need to upscale the output of the semantic segmentation
        upscale_height_face = self.original.shape[0] / self.car_mask.shape[0]
        upscale_width_face = self.original.shape[1] / self.car_mask.shape[1]

        upscale_height_face = height_final / self.car_mask.shape[0]
        upscale_width_face = width_final / self.car_mask.shape[1]

        # Upscale the segmented image to the shape of the original
        image_rescaled = rescale(
            self.car_mask,
            (upscale_height_face, upscale_width_face),
            anti_aliasing=True,
            preserve_range=True,
        )

        image_rescaled = np.array(image_rescaled, dtype=bool)
        return image_rescaled
