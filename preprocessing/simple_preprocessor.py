import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation method used when resizing
        self.width = width
        self.height = height
        self.inter = inter
		
    def resize_image(self, image):
        img_resized = cv2.resize(image, (self.width, self.height), interpolation=self.inter)
		
        return img_resized
    
    def crop_image_around_nozzle(self, image):
        # implement function to crop image around the nozzle based on coordinates here
        return image
    
    def rgb_to_grayscale(self, image):
        # implement function to transform image into grayscale here
        return image

    def preprocess(self, image):
        img_cropped = self.crop_image_around_nozzle(image)
        img_resized = self.resize_image(img_cropped)
        img_preprocessed = self.rgb_to_grayscale(img_resized)

        return img_preprocessed

