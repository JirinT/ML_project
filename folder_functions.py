import os
import shutil
import matplotlib.pyplot as plt

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def delete_files(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def delete_all_in_folder(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

def save_images(path, images):
    for i, image in enumerate(images[1:11]):
        filename = f"sample_{i}.png"
        filepath = os.path.join(path, filename)
        plt.imsave(filepath, image, cmap='gray')
