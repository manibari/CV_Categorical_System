import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (512, 512))
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_images(images):
    # Unsharp masking
    sharpened_images = []
    for img in images:
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        sharpened_images.append(sharpened)
    return np.array(sharpened_images)

def augment_images(images):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen.flow(images, batch_size=32)

def save_processed_images(images, labels, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, img in enumerate(images):
        label = labels[i]
        label_folder = os.path.join(output_folder, label)
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        cv2.imwrite(os.path.join(label_folder, f'processed_{i}.png'), img)

def main():
    raw_data_folder = 'data/raw'
    processed_data_folder = 'data/processed'
    
    images, labels = load_images_from_folder(raw_data_folder)
    processed_images = preprocess_images(images)
    save_processed_images(processed_images, labels, processed_data_folder)

if __name__ == "__main__":
    main()