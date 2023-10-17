# Author: Eraldo Pereira Marinho, Ph.D
# About: the code imports ViT to classify astronomical images
# Creation: Sep 14, 2023

import torch
import torch.nn as nn
import h5py
from pytorch_pretrained_vit import ViT
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from utils import Visualizer
import os
from cnn_transformer_core_h5_vit import model
# from cnn_transformer_core_h5_vit import learning_rate
from cnn_transformer_core_h5_vit import train_loader
from cnn_transformer_core_h5_vit import validation_loader
# from cnn_transformer_core_h5 import num_epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"pytorch device: {device}")

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

# Visualizer setup
viz = Visualizer.Visualizer('Astro Classifier', use_incoming_socket=False)

# Definir o vetor de pesos das classes obtido empiricamente da última execução:
galaxies = np.float32(1/699)
globular = np.float32(1/325)
nebulae  = np.float32(1/443)
openclust= np.float32(1/144)
norm_denominator=galaxies + globular + nebulae + openclust
weight_class_0=galaxies/norm_denominator
weight_class_1=globular/norm_denominator
weight_class_2=nebulae/norm_denominator
weight_class_3=openclust/norm_denominator

# Instantiate the class weight tensor
class_weights = torch.tensor([weight_class_0, weight_class_1, weight_class_2, weight_class_3])
class_weights = class_weights.to(device)

# Parâmetros de treinamento
num_epochs = 40
initial_learning_rate = 1e-4
weight_decay = 1e-6
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Mover o modelo para o dispositivo GPU
model.to(device)

# Função de treinamento
def train(model, dataloader, validation_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        scheduler.step()
        if epoch % 2 == 0:
            test(model, validation_loader)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        viz.plot_lines('batch loss', epoch_loss)

# predicted_labels array is used to construct a histogram to reveal how many times each class was used along the evaluation
predicted_labels = []

# Função de validação
def test(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.extend(predicted.tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    viz.plot_lines('Acuracy', accuracy)

# Training loop
train(model, train_loader, validation_loader, criterion, optimizer, num_epochs)
test(model, validation_loader)

# Save the trained model
saved_model_path = 'trained_cnn_model.pth'
torch.save(model.state_dict(), saved_model_path)
print(f"Trained model saved to '{saved_model_path}'")

# plot the histogram for predicted categories - unbalanced histogram means low quality training
unique_labels = set(predicted_labels)
print("unique labels: ", unique_labels)
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
indices = [label_to_idx[label] for label in predicted_labels]
label_counts = torch.bincount(torch.tensor(indices))
plt.bar(torch.arange(len(label_counts)), label_counts)
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.show()
