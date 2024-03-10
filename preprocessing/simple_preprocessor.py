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
        # returns gray scale matrix
        
        alpha = 1.5  # Contrast control (1.0 means no change)
        beta = 30    # Brightness control (0 means no change)

        gray_img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        gray_img = cv.convertScaleAbs(gray_img, alpha=alpha, beta=beta)
       
        return gray_img

    def unsharp_mask(self,image, kernel_size=(5,5), sigma=0, amount=1.0, threshold=1):
        """Return a sharpened version of the image, using an unsharp mask algorithm
        Args:
        sigma: increasing sigma will decrease the impact of the pixels nearest the pixel of interest, e.g. it makes a blurrier image.
        amount: amount of sharpening
        threshold: -is the threshold for the low-contrast mask. The pixels for which the difference between the input and blurred images is less than threshold will remain unchanged.
                   -removes some noise from sharpening process

        Parameters influence each other a lot, so only change one at a time."""
        
        blurred = cv.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

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
        img_rgb2gray = self.rgb_to_grayscale(img_resized)
        img_preprocessed = self.unsharp_mask(img_rgb2gray)

        return img_preprocessed
    