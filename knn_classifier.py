import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import numpy as np
import cv2


def image_to_feature_vector(image, size):
    return cv2.resize