# Author: Eraldo Pereira Marinho, Ph.D
# About: The code is a core module to build a VGG-like CNN with transformer, originally designed to classify astronomical images
# Creation: Aug 29, 2023
# Usage: Import cnn_transformer_core and its components therein
# In the present version, the number of CNN layers is variable,
# which slowed down in comparison with the former version

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, ReLU, PReLU
import torchvision
import torchvision.transforms as transforms
# import torch.nn.functional as F
import h5py
import torch.nn.init as init
import os

class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

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

# Create data loaders
batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Transformer Encoder Parameters
embedding_dimension = 64 # Dimension of the feature space, which is an important dimension for attention

# Define the CNN + Transformer model class
class CNNTransformer(nn.Module):
    def __init__(self,
                 cnn_model,
                 num_heads = 16,
                 transformer_layers = 6, # Number of Transformer Encoder attention layers
                 encoder_dropout = .1,
        ):
        super(CNNTransformer, self).__init__()
        self.cnn_model = cnn_model

        # Transformer Encoder Configuration
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=embedding_dimension,
                                    nhead=num_heads,
                                    activation=PReLU(),
                                    dropout=encoder_dropout),
            num_layers=transformer_layers
        )

        # # Adding dense layers for classification after the CNN Transformer
        # input_size = embedding_dimension
        # output_size_1 = 256
        # output_size_2 = 128
        # assert num_dense_layers > 0, "number of dense layers must be greater than or equal to 1"
        # dense_layers = []
        # for _ in range(num_dense_layers):
        #     dense_layers.append(nn.Linear(input_size, output_size_1))
        #     dense_layers.append(nn.ReLU())
        #     input_size = output_size_1
        #
        # dense_layers.append(nn.Linear(output_size_1, output_size_2))
        # dense_layers.append(nn.ReLU())
        # input_size = output_size_2
        # self.dense_layers = nn.Sequential(*dense_layers)
        # self.fc = nn.Linear(output_size_2, num_classes)
        self.fc = nn.Linear(embedding_dimension, num_classes)

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

        # # Passing features through the set of dense layers
        # transformed_features = self.dense_layers(transformed_features)

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
                 dropout = 0.5,
                 model_checkpoint = "trained_cnn_model.pth"):
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
                nn.PReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            if not os.path.exists(model_checkpoint):
                # Initialize the PReLU activations
                nn.init.normal_(conv_layer[2].weight, mean=0.01, std=0.02)
            self.conv_layers.append(conv_layer)
            in_channels = out_dim

        # Global Max Pooling only if (h_out, w_out) == (1,1)
        self.global_max_pooling = nn.AdaptiveMaxPool2d((3, 3))

        # Dense layers for pre-classification
        self.dense_layers = nn.ModuleList()
        in_dim = out_dim
        for out_dim in dense_dims:
            dense_layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(p=dropout),
                nn.PReLU()
            )
            self.dense_layers.append(dense_layer)
            in_dim = out_dim

        # CNN output layer used as embedding dimension for the Transformer
        self.embedding_layer = nn.Sequential(
            nn.Linear(in_dim, embedding_dimension),
            nn.Dropout(p=dropout),
            nn.ReLU()
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

# Initialize the entire model, including CNN and Transformer layers
def initialize_weights(model, model_checkpoint="trained_cnn_model.pth"):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Check if a pretrained weights file is provided
            if os.path.exists(model_checkpoint):
                checkpoint = torch.load(model_checkpoint)
                # This was necessary to mitigate an inconsistency between saved/loaded weights file
                for name, param in model.named_parameters():
                    if name in checkpoint:
                        param.data.copy_(checkpoint[name])
                # # Commented due to the explanation above
                # module.load_state_dict(checkpoint)
            else:
                # Apply default weight initialization
                if hasattr(module, 'weight'):
                    init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(module, 'bias') and module.bias is not None:
                    init.constant_(module.bias, 0)
                # If the module has PReLU activation, initialize its weight with a small positive value
                if isinstance(module, nn.PReLU):
                    nn.init.normal_(module.activation.weight, mean=0.01, std=0.02)
        elif isinstance(module, nn.TransformerEncoderLayer):
            # Initialize Transformer layers
            init.kaiming_normal_(module.self_attn.in_proj_weight, mode='fan_out', nonlinearity='relu')
            init.kaiming_normal_(module.self_attn.out_proj.weight, mode='fan_out', nonlinearity='relu')
            init.kaiming_normal_(module.linear1.weight, mode='fan_out', nonlinearity='relu')
            init.kaiming_normal_(module.linear2.weight, mode='fan_out', nonlinearity='relu')
            if module.self_attn.in_proj_bias is not None:
                init.constant_(module.self_attn.in_proj_bias, 0)
            if module.self_attn.out_proj.bias is not None:
                init.constant_(module.self_attn.out_proj.bias, 0)
            if module.linear1.bias is not None:
                init.constant_(module.linear1.bias, 0)
            if module.linear2.bias is not None:
                init.constant_(module.linear2.bias, 0)
            # Initialize PReLU activations in the Transformer layer
            if isinstance(module, nn.PReLU):
                nn.init.normal_(module.activation.weight, mean=0.01, std=0.02)

# Gets the number of classes from the dataset
num_classes = len(class_labels)

# Instantiate the CNN + Dense layer + Transformer
cnn_out_dims = [64, 128, 256, 512] # List of output dimensions for convolutional layers
dense_dims = [512, 256, 128]  # List of output dimensions for dense layers
cnn_model = CNN(cnn_out_dims, dense_dims)
model = CNNTransformer(cnn_model, num_heads = 8, transformer_layers = 2)

# # Initialize the model's weights
initialize_weights(model)
