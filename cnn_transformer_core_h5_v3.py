""" Code name: cnn_transformer_core_h5_v3.py (core code to be imported by the main code) """

# Author: Eraldo Pereira Marinho, Ph.D.
# Description: This code is a core module for building a VGG-like CNN with a transformer, initially designed for classifying astronomical images.
# Created: August 29, 2023
# Usage: Import 'cnn_transformer_core' and its components.
# In this version, the number of CNN layers is variable, which may result in slower performance compared to the previous version.

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, ReLU, PReLU
import torchvision
import torchvision.transforms as transforms
# import torch.nn.functional as F
#import h5py
import torch.nn.init as init
import os

## Swish unused yet
#class Swish(nn.Module):
    #def __init__(self, beta=1.0):
        #super(Swish, self).__init__()
        #self.beta = beta

    #def forward(self, x):
        #return x * torch.sigmoid(self.beta * x)

## Define how to load class labels from the H5 file
#def load_class_labels_from_h5(h5file_path, dataset_name):
    #with h5py.File(h5file_path, "r") as h5file:
        #class_labels = h5file.attrs[f"{dataset_name}_class_labels"]
    #return class_labels

## Load class labels from the H5 file
#class_labels = load_class_labels_from_h5("datasets.h5", "train")

## Function to load data and labels from H5 file
#def load_data_from_h5(h5file_path, dataset_name):
    #with h5py.File(h5file_path, "r") as h5file:
        #data = h5file[f"{dataset_name}_data"][:]
        #labels = h5file[f"{dataset_name}_labels"][:]
    #return data, labels

## Load training data and labels from H5 file
#train_data, train_labels = load_data_from_h5("datasets.h5", "train")

## Load validation data and labels from H5 file
#validation_data, validation_labels = load_data_from_h5("datasets.h5", "validation")

## Create PyTorch tensors from the loaded NumPy arrays
#train_data = torch.tensor(train_data)
#train_labels = torch.tensor(train_labels)
#validation_data = torch.tensor(validation_data)
#validation_labels = torch.tensor(validation_labels)

## Create custom PyTorch datasets
#train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
#validation_dataset = torch.utils.data.TensorDataset(validation_data, validation_labels)

# Gets the number of classes from the dataset
#num_classes = len(class_labels)


# Define the CNN + Transformer model class
class CNNTransformer(nn.Module):
    def __init__(self,
                 cnn_model,
                 num_heads = 16,
                 transformer_layers = 4, # Number of Transformer Encoder attention layers
                 num_dense_layers = 3,
                 embedding_dimension = 128,
                 encoder_dropout = .01,
                 num_classes = 4,
        ):
        super(CNNTransformer, self).__init__()
        self.cnn_model = cnn_model

        # Transformer Encoder Configuration
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=embedding_dimension,
                nhead=num_heads, activation='relu', # Break the Transformer default activation (GELU)
                dropout=encoder_dropout
            ),
            num_layers=transformer_layers,
            enable_nested_tensor=False, # As suggested by GPT-4
        )

        # Adding dense layers for classification after the CNN Transformer
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
        # Feature extraction using the CNN
        features = self.cnn_model(x)

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

# Defining the CNN architecture for feature extraction
class CNN(nn.Module):
    def __init__(self,
                 cnn_out_dims,
                 dense_dims,
                 embedding_dimension = 128,
                 dropout = 0.5,
                 num_classes = 4,
                 ):
        super(CNN, self).__init__()

        self.cnn_out_dims = cnn_out_dims
        self.dense_dims = dense_dims

        # Convolutional layers to extract features from images
        self.conv_layers = nn.ModuleList()
        in_channels = 3  # Number of input channels, say, (R, G, B)
        for out_dim in cnn_out_dims:

            conv_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.PReLU(), # PReLU worked better than both ReLU and GELU
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.conv_layers.append(conv_layer)
            in_channels = out_dim

        # Global Max Pooling - presuming the input image is (256,256) size with 4 convolutional layers
        # Since the previous output is a collection of (16,16) images, then the global max pooling becomes
        # a grid of 8x8=64 (2,2) adaptive max pooling cells
        self.global_max_pooling = nn.AdaptiveMaxPool2d((8,8)) # This is the best to fit (16,16) convolutional output-layer

        # Dense layers for pre-classification
        self.dense_layers = nn.ModuleList()
        in_dim = out_dim
        for out_dim in dense_dims:
            dense_layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(p=dropout),
                nn.ReLU() # Better results with ReLU
            )
            self.dense_layers.append(dense_layer)
            in_dim = out_dim

        # CNN output layer used as embedding dimension for the Transformer Encoder
        self.embedding_layer = nn.Sequential(
            nn.Linear(in_dim, embedding_dimension),
            # nn.Dropout(p=dropout), # Dropout for FC output didn't work
            nn.GELU() # Some improvement using GELU
        )

    def forward(self, x):
        # Propagating images through the convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Global Max Pooling
        x = nn.functional.adaptive_max_pool2d(x, (1, 1)) # Don't touch this line!

        # Reshaping outputs to be compatible with the dense layers
        x = x.view(x.size(0), -1)

        # Propagating through the dense layers for pre-classification
        for dense_layer in self.dense_layers:
            x = dense_layer(x)

        # Embedding layer
        x = self.embedding_layer(x)

        return x
