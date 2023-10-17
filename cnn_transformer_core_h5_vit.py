# Author: Eraldo Pereira Marinho, Ph.D
# About: the code is a core module to the Visual Transformer, originally designed to classify astronomical images
# Creation: Sep 14, 2023
# Usage, import cnn_transformer_core and its components therein

import torch
import torch.nn as nn
import h5py
from vit_pytorch import SimpleViT

# nmaxpool = 4
img_width = 256
img_height = 256

# Load class labels from the H5 file
def load_class_labels_from_h5(h5file_path, dataset_name):
    with h5py.File(h5file_path, "r") as h5file:
        class_labels = h5file.attrs[f"{dataset_name}_class_labels"]
    return class_labels

# Load class labels from the H5 file
class_labels = load_class_labels_from_h5("datasets.h5", "train")

# Function to load data and labels from H5 file
def load_data_from_h5(h5file_path, dataset_name):
    with h5py.File(h5file_path, "r") as h5file:
        data = h5file[f"{dataset_name}_data"][:]
        labels = h5file[f"{dataset_name}_labels"][:]
    return data, labels

# Load training data and labels from H5 file
train_data, train_labels = load_data_from_h5("datasets.h5", "train")

# Load validation data and labels from H5 file
validation_data, validation_labels = load_data_from_h5("datasets.h5", "validation")

# Create PyTorch tensors from the loaded NumPy arrays
train_data = torch.tensor(train_data)
train_labels = torch.tensor(train_labels)
validation_data = torch.tensor(validation_data)
validation_labels = torch.tensor(validation_labels)

# Create custom PyTorch datasets
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
validation_dataset = torch.utils.data.TensorDataset(validation_data, validation_labels)

# Create data loaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Parameters for ViT model
num_classes = len(class_labels)  # Number of output classes

# Define the classification model using ViT
class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ViTClassifier, self).__init__()
        self.vit_model = SimpleViT(
            image_size=img_width,
            patch_size=16,
            num_classes=num_classes,
            dim=1024,
            depth=4,
            heads=16,
            mlp_dim=2048
        )

    def forward(self, x):
        x = self.vit_model(x)
        return x

# Instantiate the ViT-based model
model = ViTClassifier(num_classes)
