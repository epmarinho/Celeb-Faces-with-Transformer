# Author: Eraldo Pereira Marinho, Ph.D.
# Description: This code is a core module for building a ResNet with a transformer, initially designed for classifying celebs images.
# Created: October 17, 2023
# Usage: Import 'resnet_transformer_core' and its components.

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, ReLU, PReLU
from transformers import ViTFeatureExtractor, ViTForImageClassification
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
batch_size = 8

# Create data loaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Gets the number of classes from the dataset
num_classes = len(class_labels)
print(f'Number of classes is {num_classes}')

# Load pre-trained models
resnet50 = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

class ResNetClassifier(nn.Module):
    def __init__(self, resnet, num_classes):
        super(ResNetClassifier, self).__init__()
        self.resnet = resnet
        # Modify the last fully connected layer of ResNet to match the number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        # Use the modified ResNet model directly
        return self.resnet(x)


#vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

## Modify the final classification head for ViT
#vit.classifier = nn.Linear(vit.config.hidden_size, num_classes)

## Combine the models
#class CelebrityClassifier(nn.Module):
    #def __init__(self, resnet, vit, num_classes):
        #super(CelebrityClassifier, self).__init__()
        #self.resnet = resnet
        #self.vit = vit
        #self.fc = nn.Linear(1000 + num_classes, num_classes)

    #def forward(self, x):
        #resnet_features = self.resnet(x)
        #vit_output = self.vit(x)['logits']

        ## Combine features or outputs along the second dimension (dim=1)
        #combined_output = torch.cat((resnet_features, vit_output), dim=1)

        #return self.fc(combined_output)

# Create an instance of the ResNetClassifier with the modified ResNet model
model = ResNetClassifier(resnet50, num_classes)

#model = CelebrityClassifier(resnet50, vit, num_classes)

# # Define the ResNet + Transformer model class
# class ResNetTransformer(nn.Module):
#     def __init__(self,
#                  resnet_model,
#                  num_heads = 16,
#                  transformer_layers = 6, # Number of Transformer Encoder attention layers
#                  # num_dense_layers = 3,
#                  encoder_dropout = .1,
#         ):
#         super(ResNetTransformer, self).__init__()
#         self.resnet_model = resnet_model
#
#         # Transformer Encoder Configuration
#         self.transformer = TransformerEncoder(
#             TransformerEncoderLayer(
#                 d_model=embedding_dimension,
#                 nhead=num_heads, activation='relu', # Break the Transformer default activation (GELU)
#                 dropout=encoder_dropout
#             ),
#             num_layers=transformer_layers,
#             enable_nested_tensor=True,
#         )
#         self.fc = nn.Linear(embedding_dimension, num_classes)
#
#     def forward(self, x):
#         # Feature extraction using the ResNet
#         features = self.resnet_model(x)
#
#         # Preparing features for input to the Transformer
#         features = features.view(features.size(0), features.size(1), -1)  # Flatten the features
#         features = features.permute(2, 0, 1)  # Reorder for the Transformer-compatible format
#
#         # Applying the Transformer Encoder on the features
#         transformed_features = self.transformer(features)
#
#         # Reverting the shape of features to the original format
#         transformed_features = transformed_features.permute(1, 2, 0)
#         transformed_features = transformed_features.contiguous().view(transformed_features.size(0), -1)
#
#         # Final classification layer
#         output = self.fc(transformed_features)
#         return output
#
# # Defining the ResNet architecture for feature extraction - I'm using ResNet as a black box
# class ResNet(nn.Module):
#     def __init__(self, dense_dims, dropout=0.5):
#         super(ResNet, self).__init__()
#         self.dense_dims = dense_dims
#         # Use a pre-trained ResNet model as the feature extractor
#         # self.resnet = models.resnet50(pretrained=True)
#         self.resnet = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
#         # Adjust the last classification layer of the ResNet to match output dimension
#         num_ftrs = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_ftrs, resnet_out_dim)
#
#         # Dense layers for pre-classification
#         self.dense_layers = nn.ModuleList()
#         in_dim = resnet_out_dim
#         for out_dim in dense_dims:
#             dense_layer = nn.Sequential(
#                 nn.Linear(in_dim, out_dim),
#                 nn.Dropout(p=dropout),
#                 nn.ReLU()
#             )
#             self.dense_layers.append(dense_layer)
#             in_dim = out_dim
#
#     def forward(self, x):
#         # Propagate images through the ResNet feature extractor
#         x = self.resnet(x)
#         # Propagate through dense layers
#         for dense_layer in self.dense_layers:
#             x = dense_layer(x)
#         return x

# # Transformer Encoder Parameters
# embedding_dimension = 64 # Dimension of the feature space, which is an important dimension for encoder attention
# # Instantiate the ResNet + Dense layer + Transformer
# dense_dims = [2*embedding_dimension, embedding_dimension] # List of output dimensions for dense layers # The best for unsorted celebs image classification
# resnet_out_dim = 2 * dense_dims[0]
# resnet_model = ResNet(dense_dims)
# model = ResNetTransformer(resnet_model, num_heads = 16, transformer_layers = 1)
