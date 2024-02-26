import cv2 as cv
import numpy as np

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv.INTER_AREA):
        # store the target image width, height, and interpolation method used when resizing
        self.width = width
        self.height = height
        self.inter = inter
        self.coordinates = []
		
    def resize_image(self, image):
        # original size of images is 720x1280
        # this fcn just change the size of the image, does not crop it 
        img_resized = cv.resize(image, (self.width, self.height), interpolation=self.inter)
        return img_resized
    
    def crop_image_around_nozzle(self, image, crop_size=100):
        # implement function to crop image around the nozzle based on its coordinates 
        # returns image with dimensions crop_size x crop_size
        x = self.coordinates[0] # x coordinate of nozzle in image
        y = self.coordinates[1] # y coordinate of nozzle in image
        half_crop_size = int(np.floor(crop_size/2)) # needs to be integer.
        cropped_image = image[y-half_crop_size : y+half_crop_size, x-half_crop_size-50 : x+half_crop_size] # with -50 works better for images ive tried
    
        return cropped_image
    
    def rgb_to_grayscale(self, image):
        # returns gray scale matrix
        gray_img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        return gray_img

    def preprocess(self, image):
        img_cropped = self.crop_image_around_nozzle(image)
        img_resized = self.resize_image(img_cropped)
        img_preprocessed = self.rgb_to_grayscale(img_resized)

        return img_preprocessed
    