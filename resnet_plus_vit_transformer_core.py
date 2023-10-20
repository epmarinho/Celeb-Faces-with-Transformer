# Author: Eraldo Pereira Marinho, Ph.D.
# Description: This code is a core module for building a ResNet with a transformer, initially designed for classifying astronomical images.
# Created: October 15, 2023
# Usage: Import 'resnet_transformer_core' and its components.

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, ReLU, PReLU
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
# import torch.nn.functional as F
import h5py
import torch.nn.init as init
import os

# class Swish(nn.Module):
#     def __init__(self, beta=1.0):
#         super(Swish, self).__init__()
#         self.beta = beta
#
#     def forward(self, x):
#         return x * torch.sigmoid(self.beta * x)

# Define how to load class labels from the H5 file
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

# Setup the mini-batch size
batch_size = 16

# Create data loaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Define the ResNet + Transformer model class
class ResNetTransformer(nn.Module):
    def __init__(self,
                 resnet_model,
                 num_heads = 16,
                 transformer_layers = 6, # Number of Transformer Encoder attention layers
                 num_dense_layers = 3,
                 encoder_dropout = .1,
        ):
        super(ResNetTransformer, self).__init__()
        self.resnet_model = resnet_model

        # Transformer Encoder Configuration
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=embedding_dimension,
                nhead=num_heads, activation='relu', # Break the Transformer default activation (GELU)
                dropout=encoder_dropout
            ),
            num_layers=transformer_layers,
            enable_nested_tensor=True,
        )

        # Adding dense layers for classification after the ResNet Transformer
        input_size = embedding_dimension
        output_size_2 = input_size // 2** num_dense_layers
        assert output_size_2 > num_classes, f'Error: FC output size = {output_size_2} whereas number of classes = {num_classes}'
        dense_layers = []
        for _ in range(num_dense_layers):
            output_size_1 = input_size // 2
            dense_layers.append(nn.Linear(input_size, output_size_1))
            dense_layers.append(nn.GELU()) # Worked much much better with GELU
            input_size = output_size_1

        self.dense_layers = nn.Sequential(*dense_layers)
        self.fc = nn.Linear(output_size_2, num_classes) if num_dense_layers > 0 else nn.Linear(embedding_dimension, num_classes)

    def forward(self, x):
        # Feature extraction using the ResNet
        features = self.resnet_model(x)

        # Preparing features for input to the Transformer
        features = features.view(features.size(0), features.size(1), -1)  # Flatten the features
        features = features.permute(2, 0, 1)  # Reorder for the Transformer-compatible format

        # Applying the Transformer Encoder on the features
        transformed_features = self.transformer(features)

        # Reverting the shape of features to the original format
        transformed_features = transformed_features.permute(1, 2, 0)
        transformed_features = transformed_features.contiguous().view(transformed_features.size(0), -1)

        # Passing features through the set of dense layers
        transformed_features = self.dense_layers(transformed_features)

        # Final classification layer
        output = self.fc(transformed_features)
        return output

# Note:
"""
# For image classification, a Decoder layer is not necessary. The Transformer Encoder is used to extract useful features from the image,
# and subsequent classification layers are used to make class predictions. This is a suitable design for image classification tasks,
# including the classification of astronomical images.
"""

# Defining the ResNet architecture for feature extraction - I'm using ResNet as a black box
class ResNet(nn.Module):
    def __init__(self, dense_dims, dropout=0.5):
        super(ResNet, self).__init__()
        self.dense_dims = dense_dims
        # Use a pre-trained ResNet model as the feature extractor
        # self.resnet = models.resnet50(pretrained=True)
        self.resnet = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        # Adjust the last classification layer of the ResNet to match your output dimension
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, embedding_dimension)

        # Dense layers for pre-classification
        self.dense_layers = nn.ModuleList()
        in_dim = embedding_dimension
        for out_dim in dense_dims:
            dense_layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(p=dropout),
                nn.ReLU()
            )
            self.dense_layers.append(dense_layer)
            in_dim = out_dim

    def forward(self, x):
        # Propagate images through the ResNet feature extractor
        x = self.resnet(x)
        # Propagate through dense layers
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        return x

# Gets the number of classes from the dataset
num_classes = len(class_labels)

# Transformer Encoder Parameters
embedding_dimension = 128 # Dimension of the feature space, which is an important dimension for encoder attention
# Instantiate the ResNet + Dense layer + Transformer
dense_dims = [4*embedding_dimension, 2*embedding_dimension, embedding_dimension] # List of output dimensions for dense layers # The best for unsorted astronomical image classification
resnet_model = ResNet(dense_dims)
model = ResNetTransformer(resnet_model, num_heads = 16, transformer_layers = 2, num_dense_layers = 2) # The best for unsorted astronomical image classification by now
