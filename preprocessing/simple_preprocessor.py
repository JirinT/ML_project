import json

import cv2 as cv
import numpy as np

with open("config.json") as f:
    config_preprocessor = json.load(f)["preprocessor"]

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv.INTER_AREA):
        # store the target image width, height, and interpolation method used when resizing
        self.width = width
        self.height = height
        self.inter = inter
        self.coordinates = []
		
    def resize_image(self, image):
        """
        changes the size of the image to the specified width and height, does not crop it
        original size of images is 720x1280

        Args:
            image (numpy.ndarray): The image to be resized.

        Returns:
            numpy.ndarray: The resized image.
        """
        img_resized = cv.resize(image, (self.width, self.height), interpolation=self.inter)

        return img_resized
    
    def crop_image_around_nozzle(self, image, crop_size=config_preprocessor["crop_size"]):
        """
        crop image around the nozzle based on its coordinates

        Args:
            image (numpy.ndarray): The image to be cropped.
            crop_size (int): The size of the cropped image.

        Returns:
            numpy.ndarray: The cropped image (crop_size * crop_size).
        """
        x = self.coordinates[0]
        y = self.coordinates[1]
        half_crop_size = int(np.floor(crop_size/2))
        cropped_image = image[y-half_crop_size : y+half_crop_size, x-half_crop_size-50 : x+half_crop_size] # with -50 works better for images ive tried
    
        return cropped_image
    
    def rgb_to_grayscale(self, image):
        """
        turns the image to gray scale

        Args:
            image (numpy.ndarray): The image to be turned to gray scale.

        Returns:
            numpy.ndarray: The gray scale image.
        """
        gray_img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        gray_img = cv.equalizeHist(gray_img) # increasing contrast, so the printed filament is more visible - this should be changed, does not work very well so far
        return gray_img

    def preprocess(self, image):
        """
        Preprocess the image by applying a series of operations.

        Args:
            image (numpy.ndarray): The image to be preprocessed.

        Returns:
            numpy.ndarray: The preprocessed image.
        """
        img_cropped = self.crop_image_around_nozzle(image)
        img_resized = self.resize_image(img_cropped)
        img_preprocessed = self.rgb_to_grayscale(img_resized)

        return img_preprocessed
    