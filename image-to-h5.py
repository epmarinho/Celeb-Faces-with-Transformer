# Authors: Eraldo Pereira Marinho and ChatGPT
# Sep 11, 2023, 4:23pm

import h5py
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, features
import pillow_avif
from torchvision.datasets import ImageFolder
# from cnn_transformer_core_h5_v3 import batch_size

batch_size = 64

# Define a function to convert a PyTorch DataLoader to H5 format
def dataloader_to_h5(loader, h5file, dataset_name, class_labels):
    data_list = []
    labels_list = []

    for inputs, labels in loader:
        try:
            data_list.append(inputs.numpy())  # Convert PyTorch tensor to NumPy array
            labels_list.append(labels.numpy())  # Convert PyTorch tensor to NumPy array
        except Exception as e:
            print(f"Error loading data from DataLoader: {e}")
            continue

    if len(data_list) == 0 or len(labels_list) == 0:
        print("No valid data to store in H5 file.")
        return

    data_array = np.concatenate(data_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    try:
        h5file.create_dataset(f"{dataset_name}_data", data=data_array)
        h5file.create_dataset(f"{dataset_name}_labels", data=labels_array)
    except Exception as e:
        print(f"Error creating datasets in H5 file: {e}")

    # Store class labels as an attribute
    try:
        h5file.attrs[f"{dataset_name}_class_labels"] = class_labels
    except Exception as e:
        print(f"Error storing class labels as attributes in H5 file: {e}")

# Image dimensions
img_width = 272
img_height = 272
crop_size = 256

# Transformations for preprocessing
transform_train = transforms.Compose([
    transforms.RandomRotation(30), # Apply a random rotation to the image within the range of -30 to +30 degrees
    transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally (left to right)
    transforms.RandomAdjustSharpness(sharpness_factor=1.5), # Randomly adjust the sharpness of the image, making it 1.5 times sharper
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), # Randomly adjusts brightness, contrast, saturation, and hue.
    transforms.Resize((img_width, img_height)),
    transforms.RandomCrop(crop_size),
    transforms.ToTensor(), # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # transformes color ranges from [0,1] to [-1,1]
])
transform_validation = transform_train

# Define the root directory of your dataset
train_data_root = r'images/train'
validation_data_root = r'images/validation'

# Create ImageFolder datasets to infer class labels
train_dataset = ImageFolder(root=train_data_root, transform=transform_train)
validation_dataset = ImageFolder(root=validation_data_root, transform=transform_validation)

# is_normalized = True  # Assume the dataset is normalized

# for image, _ in train_dataset:
#     min_pixel_value = torch.min(image)
#     max_pixel_value = torch.max(image)

    # if min_pixel_value != 0.0 or max_pixel_value != 1.0:
    #     is_normalized = False
    #     break

# if is_normalized:
#     print("The images are normalized to [0, 1].")
# else:
#     print("The images are not normalized to [0, 1].")
# print(f" Min pix = {min_pixel_value}, max pix = {max_pixel_value}.")

# Get the class labels from the dataset
class_labels = train_dataset.classes

# Check images integrity for training/validation images
from tqdm import tqdm
import os
def check_images(image_folder):
    for subdir in os.listdir(image_folder):
        subdir_path = os.path.join(image_folder, subdir)

        # Check if the item in validation_data_root is a directory
        if os.path.isdir(subdir_path):
            for filename in tqdm(os.listdir(subdir_path), desc=f"Processing Images in {subdir}"):
                image_path = os.path.join(subdir_path, filename)

                try:
                    sample_image = Image.open(image_path).convert("RGB")
                    # Continue with preprocessing and inference here
                except Exception as e:
                    print(f"Error loading image {image_path}: {str(e)}")

# Create data loaders

print("\n### Checking training images\n")
check_images(train_data_root)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("\n### Checking validation images\n")
check_images(validation_data_root)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

# Create an H5 file to store the data

print("\n### Converting images data-loader to H5 format\n")
h5file_path = "datasets.h5"
with h5py.File(h5file_path, "w") as h5file:
    dataloader_to_h5(train_loader, h5file, "train", class_labels)
    dataloader_to_h5(validation_loader, h5file, "validation", class_labels)
print("\n### Done\n")

