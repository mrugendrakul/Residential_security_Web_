import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
from skimage import exposure
import streamlit as st
import csv

def img_to_array(img):
    return np.array(img, dtype='float32') / 255.0

def load_img(path):
    return Image.open(path)

def augment_images(input_folder, output_folder, class_name, num_augmented_images):
    print("in augment image")
    # Create output folder for augmented images
    output_class_folder = os.path.join(output_folder, class_name)
    os.makedirs(output_class_folder, exist_ok=True)

    # Augment each image in the class folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            img = load_img(img_path)
            img_array = img_to_array(img)

            # Reshape for augmentation
            x = img_array.reshape((1,) + img_array.shape)

            # Define an ImageDataGenerator for augmentation
            datagen = ImageDataGenerator(
                
            )

            # Fit the datagen on the current image
            datagen.fit(x)

            # Generate augmented images
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=output_class_folder, save_prefix='aug', save_format='jpg'):
                i += 1
                if i >= num_augmented_images:
                    break

def augment_dataset(input_root_folder, output_root_folder, num_augmented_images):
    # Process each class folder
    print("Augentation dataset")
    for class_name in os.listdir(input_root_folder):
        class_folder = os.path.join(input_root_folder, class_name)
        if os.path.isdir(class_folder):
            augment_images(class_folder, output_root_folder, class_name, num_augmented_images)

if __name__ == "__main__":


    # Folder Format : Main/class1/images, Main/class2/images......
    input_root_folder = "./newUser" 
    output_root_folder = "./newUser"
    num_augmented_images = 2

    start_aug = st.button("start the augmentation")
    if start_aug:
        print("augmentation started!!")
        augment_dataset(input_root_folder, output_root_folder, num_augmented_images)

