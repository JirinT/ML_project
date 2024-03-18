import os
import matplotlib.pyplot as plt

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def delete_files(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def save_images(path, images):
    i = 0
    for image in images[1:]:
        if i < 10: # 100 stands for how many images do we want to save
            filename = f"sample_{i}.png"
            filepath = os.path.join(path, filename)
            plt.imsave(filepath, image, cmap='gray')
        i+=1
