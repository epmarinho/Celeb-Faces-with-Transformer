# Author: Eraldo Pereira Marinho, Ph.D
# About: The code imports resnet_plus_vitcore to allow Transformer+ResNet to classify celebrity images
# Creation: October, 2023

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
# import torchvision.transforms as transforms
import visdom
from utils import Visualizer
from resnet_core import model
from resnet_core import train_dataloader
from resnet_core import validation_dataloader
import os
import torch.nn.init as init
import numpy as np
# from PIL import Image
import pillow_avif
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch device: {device}")

viz = Visualizer.Visualizer('Celebrity Classifier', use_incoming_socket=False)

# Define the class weight vector empirically obtained from the last run:
# run after the classes histogram:
# galaxies = np.float32(1/1653)
# globular = np.float32(1/845)
# nebulae  = np.float32(1/1132)
# openclust= np.float32(1/507)
# # run this before to have an actual class histogram
# galaxies = np.float32(1)
# globular = np.float32(1)
# nebulae  = np.float32(1)
# openclust= np.float32(1)
# norm_denominator=galaxies + globular + nebulae + openclust
# weight_class_0=galaxies/norm_denominator
# weight_class_1=globular/norm_denominator
# weight_class_2=nebulae/norm_denominator
# weight_class_3=openclust/norm_denominator
# Instantiate the class weight tensor
# class_weights = torch.tensor([weight_class_0, weight_class_1, weight_class_2, weight_class_3])
# print(f"Class weights = {class_weights}")
# Weights tensor must be converted to the adopted device
# class_weights = class_weights.to(device)

# Training parameters

num_epochs = 100

initial_learning_rate = 5e-5 # Larger values caused issues

# Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=1e-7)

# Check if the pretrained file exists
model_checkpoint = "trained_resnet_model.pth"
if os.path.exists(model_checkpoint):
    # Load pretrained weights
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint)
    print(f"Pretrained weights \"{model_checkpoint}\" found.")
else:
    print("No pretrained weights file found. Initializing with PyTorch default weights.")
#     # He initialization in PyTorch
#     # Access all the linear layers (fully connected)
#     # Ensure the model contains only layers that should be initialized with He
#     # for layer in model.children():
#     #     if isinstance(layer, nn.Linear):
#     #         init.kaiming_normal_(layer.weight)

# # Check the loaded weights
# print(model.state_dict())

## Move the model to the GPU device
#model.to(device)

# These are basically my earling stopping proposed in previously unpublished works
class EarlyStoppingBatch:
    def __init__(self, patience,  threshold=.005):
        self.history = []
        self.patience = patience
        self.threshold = threshold

    def update_history(self, new_loss):
        # Update the history array with the new loss value
        self.history.append(new_loss)
        if len(self.history) > self.patience:
            self.history.pop(0) # Discard the earliest one

    def should_stop(self):
        # Check if the minimum loss in the history is repeated or becomes smaller
        if len(self.history) < self.patience:
            return False  # Not enough data to decide
        return self.history[-1] <= min(self.history[:-1]) + self.threshold

class EarlyStoppingValLoss:
    def __init__(self, patience,  threshold=.005):
        self.history = []
        self.patience = patience
        self.threshold = threshold

    def update_history(self, new_loss):
        # Update the history array with the new loss value
        self.history.append(new_loss)
        # Keep only the most recent 'remembrance' elements
        if len(self.history) > self.patience:
            self.history.pop(0) # Discard the earliest one

    def should_stop(self):
        # Check if we have enough data to make a decision
        if len(self.history) < self.patience:
            return False  # Not enough data to decide

        # Check the best loss so far
        min_loss = min(self.history[:-1])

        # Check if the loss has not improved significantly for 'patience' epochs
        plateau_count = sum(1 for x in self.history[-self.patience:] if min_loss - self.threshold <= x <= min_loss + self.threshold)

        # If the loss has been on a plateau for 'patience' consecutive epochs, stop
        return plateau_count >= self.patience

class EarlyStoppingAccuracy:
    def __init__(self, patience,  threshold=.0005):
        self.history = []
        self.patience = patience
        self.threshold = threshold

    def update_history(self, new_accuracy):
        # Update the history array with the new loss value
        self.history.append(new_accuracy)
        if len(self.history) > self.patience:
            self.history.pop(0) # Discard the earliest one

    def should_stop(self):
        '''
        Determine if training should be stopped based on validation loss.

        Returns:
        - Boolean, True if training should be stopped, False otherwise.
        '''
        if len(self.history) < self.patience:
            return False  # Not enough data to decide, continue training

        # Check the best loss so far
        max_accuracy = max(self.history[:-1])

        # Count how many recent losses are within the threshold of the best loss
        plateau_count = sum(1 for x in self.history[-self.patience:] if max_accuracy - self.threshold <= x <= max_accuracy + self.threshold)

        # Stop if the loss hasn't improved for 'patience' consecutive epochs
        return plateau_count >= self.patience

# Configure schedulers for dynamic learning rate adjustment.Fseed
step_size = 10
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
scheduler_by_accuracy = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
scheduler_by_valloss = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Set up early stopping mechanisms based on different performance metrics.
early_stopping_batch = EarlyStoppingBatch(patience=40)
early_stopping_valloss = EarlyStoppingValLoss(patience=10)
early_stopping_accuracy = EarlyStoppingAccuracy(patience=5)

# Move the model to the GPU device before training
model.to(device)

# Training function
def train_and_validate(model, dataloader, validation_loader, criterion, optimizer, max_norm=2):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in dataloader:
            # Move the images and labels to the GPU device
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # Update the learning rate based on the scheduler
        scheduler.step()

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'\nEpoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}\n')
        viz.plot_lines('Batch Loss', epoch_loss)

        early_stopping_batch.update_history(epoch_loss)
        if early_stopping_batch.should_stop():
            print(f'\nEarly stopping triggered at epoch {epoch + 1} for batch loss = {epoch_loss}\n')
            break

        validation_loss, accuracy = validate(model, validation_loader)

        scheduler_by_valloss.step(validation_loss)

        early_stopping_valloss.update_history(validation_loss)
        if early_stopping_valloss.should_stop():
            print(f'\nEarly stopping triggered at epoch {epoch + 1} for validation loss = {validation_loss}\n')
            break

        scheduler_by_accuracy.step(accuracy)

        early_stopping_accuracy.update_history(accuracy)
        if early_stopping_accuracy.should_stop():
            print(f'\nEarly stopping triggered at epoch {epoch + 1} for validation accuracy = {accuracy:.2f}%\n')
            break

from sklearn.metrics import confusion_matrix
predicted_labels = []

# Validation function
def validate(model, dataloader):

    model.eval()  # Set the model to evaluation mode

    # Initialize variables to keep track of counts
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total = 0
    total_loss = 0
    correct = 0

    all_predicted = []
    all_true = []

    with torch.no_grad():
        for images, labels in dataloader:
            # Move the images and labels to the GPU device
            images = images.to(device)
            true_labels = labels.to(device)
            total += true_labels.size(0)

            # Use the trained CNN+Transformer model for the image-set
            outputs = model(images)

            # Compute the Loss function defined in criterion
            loss = criterion(outputs, true_labels)

            # Sum up the computed loss
            total_loss += loss.item()

            # Get the predicted labels
            _, predicted = torch.max(outputs.data, 1)

            # Count the correct ones
            correct += (predicted == true_labels).sum().item()

            predicted_labels.extend(predicted.tolist())

            all_predicted.extend(predicted.tolist())
            all_true.extend(true_labels.tolist())

    validation_loss = total_loss / len(dataloader)

    #early_stopping_batch.update_history(validation_loss)
    #if early_stopping_batch.should_stop():
    ## if early_stopping_batch.early_stop:
        #print(f'\nEarly stopping triggered for validation loss = {validation_loss}\n')
        #break

    accuracy = 100 * correct / total

    cm = confusion_matrix(all_true, all_predicted)

    # Compute precision, recall, specificity and F1-score for each class
    precision = []
    recall = []
    f1_scores = []
    specificity = []

    for i in range(len(cm)):
        true_positive = cm[i, i]
        false_positive = sum(cm[j, i] for j in range(len(cm))) - true_positive
        false_negative = sum(cm[i, j] for j in range(len(cm))) - true_positive
        true_negative = sum(cm[j, k] for j in range(len(cm)) for k in range(len(cm))) - (true_positive + false_positive + false_negative)

        precision_i = true_positive / (true_positive + false_positive + 1e-12)
        recall_i = true_positive / (true_positive + false_negative + 1e-12)
        f1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i + 1e-12)
        specificity_i = true_negative / (true_negative + false_positive + 1e-12)

        precision.append(precision_i)
        recall.append(recall_i)
        f1_scores.append(f1_i)
        specificity.append(specificity_i)

    print(f'Validation Loss: {validation_loss:.6f}, Validation Accuracy: {accuracy:.2f}%')
    print(f'Precision per class: {precision}')
    print(f'Recall per class: {recall}')
    print(f'F1-score per class: {f1_scores}')
    print(f'Specificity per class: {specificity}')
    viz.plot_lines('Validation Loss', validation_loss)
    viz.plot_lines('Validation Accuracy', accuracy)
    viz.plot_lines('Precision', precision)
    viz.plot_lines('Recall', recall)
    viz.plot_lines('F1-scores', f1_scores)
    viz.plot_lines('Specificity', specificity)

    return validation_loss, accuracy

# Train the ResNet+Transformer
train_and_validate(model, train_dataloader, validation_dataloader, criterion, optimizer, num_epochs)

## Validate the ResNet+Transformer
#validate(model, validation_dataloader)

# Save the trained weights
saved_model_path = 'trained_resnet_model.pth'
torch.save(model.state_dict(), saved_model_path)
# print(f"model.state_dict '{model.state_dict()}'")
print(f"Trained model saved to '{saved_model_path}'")
# # Load pretrained weights
# checkpoint = torch.load(model_checkpoint)
# model.load_state_dict(checkpoint)
# print("Pretrained weights loaded successfully.")
# # Check the loaded weights
# torch.save(model.state_dict(), 'pesos_lidos.pth')
# torch.save(model.state_dict(), 'trained_resnet_model.pth')

# Plot the histogram for predicted categories - an unbalanced histogram indicates low-quality training
unique_labels = set(predicted_labels)
print("Unique labels: ", unique_labels)
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
indices = [label_to_idx[label] for label in predicted_labels]
# print("Indices: ", indices)

label_count = torch.bincount(torch.tensor(indices, dtype=torch.int64))

print("Label count: ", label_count)

# plt.bar(torch.arange(len(label_count)), label_count)
# plt.xlabel('Labels')
# plt.ylabel('Frequency')
# plt.show()
