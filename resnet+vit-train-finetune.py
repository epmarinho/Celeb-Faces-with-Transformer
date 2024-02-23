''' Code name: cnn-train-transformer-h5-celeb.py (main code for training)'''

# Author: Eraldo Pereira Marinho, Ph.D
# About: The code imports cnn_transformer_core to allow Transformer+CNN to classify astronomical images
# Creation: Jul 12, 2023
#
# This is a benchmark script to find out the optimal hyperparameters.
#
# Latter changes:
# 3 Feb 2024: adapted to celebrity classification

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import visdom
from utils import Visualizer
from resnet_vit_finetune import class_labels
from resnet_vit_finetune import train_dataset
from resnet_vit_finetune import validation_dataset
from resnet_vit_finetune import CelebrityClassifier, model, vit, resnet50
import os
import torch.nn.init as init
import numpy as np
# from PIL import Image
import pillow_avif
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch device: {device}')

viz = Visualizer.Visualizer('Celebrity Classifier', use_incoming_socket=False)
vis = visdom.Visdom()

## Define the class weight vector empirically obtained from the last run:
## run after the classes histogram:
#galaxies = np.float32(1/190)
#globular = np.float32(1/109)
#nebulae  = np.float32(1/190)
#openclust= np.float32(1/124)
## # run this before to have an actual class histogram
## galaxies = np.float32(1)
## globular = np.float32(1)
## nebulae  = np.float32(1)
## openclust= np.float32(1)
#norm_denominator=galaxies + globular + nebulae + openclust
#weight_class_0=galaxies/norm_denominator
#weight_class_1=globular/norm_denominator
#weight_class_2=nebulae/norm_denominator
#weight_class_3=openclust/norm_denominator

## Instantiate the class weight tensor
#class_weights = torch.tensor([weight_class_0, weight_class_1, weight_class_2, weight_class_3])
#print(f'Class weights = {class_weights}')

## Weights tensor must be converted to the adopted device
#class_weights = class_weights.to(device)

def init_weights(m):
    """ As we say in portuguese, "Isso está uma lambança!" """
    #if isinstance(m, nn.Conv2d):
        ## Use He initialization for convolutional layers
        #init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #if m.bias is not None:
            #init.zeros_(m.bias)
    #elif isinstance(m, nn.Linear):
        ## Use He initialization for linear layers as well
        #init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #if m.bias is not None:
            #init.zeros_(m.bias)

    #if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #init.kaiming_uniform_(m.weight, nonlinearity='relu')

    for layer in m.children():
        if isinstance(layer, nn.Linear):
            init.kaiming_normal_(layer.weight)

    #if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        ## Initialize weights using Xavier uniform initialization
        #init.xavier_uniform_(m.weight)

        # Set biases to zero if they exist
        #if m.bias is not None:
            #init.constant_(m.bias, 0)

def reset_model(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

# Gets the number of classes from the dataset
num_classes = len(class_labels)

num_epochs = 100

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
    #print(f'Precision per class: {precision}')
    #print(f'Recall per class: {recall}')
    #print(f'F1-score per class: {f1_scores}')
    #print(f'Specificity per class: {specificity}')
    viz.plot_lines('Validation Loss', validation_loss)
    viz.plot_lines('Validation Accuracy', accuracy)
    viz.plot_lines('Precision', precision)
    viz.plot_lines('Recall', recall)
    viz.plot_lines('F1-scores', f1_scores)
    viz.plot_lines('Specificity', specificity)

    return validation_loss, accuracy


'''  **** Grid search loop ****  '''

# Define the grid for hyperparameters
step_sizes = [7, 5, 3]
learning_rates = [6e-5]
max_norms = [2.0]
weight_decays = [2e-7]
batch_sizes = [16]

print(f'\nStep sizes = {step_sizes}')
print(f'learning rates = {learning_rates}')
print(f'max norms for gradients clipping = {max_norms}')
print(f'weight_decays = {weight_decays}')
print(f'batch sizes = {batch_sizes}\n')

best_accuracy = 0  # Track the best accuracy
best_hyperparameters = None  # Track the best hyperparameters


''' Here it is defined the training grid for the parameter sets above '''

has_started = False
# Outer loops for hyperparameter tuning. Each loop iterates over a range of values for a specific hyperparameter.
for step_size in step_sizes:
    print(f'step size = {step_size}')
    for learning_rate in learning_rates:
        print(f'learning rate = {learning_rate}')
        for max_norm in max_norms:
            print(f'Max norm for gradients clipping = {max_norm}')
            for weight_decay in weight_decays:
                print(f'Weight decay = {weight_decay}')
                # Inner loops for batch size variations. This affects memory usage, hence the careful handling.
                for batch_size in batch_sizes:
                    print(f'batch size = {batch_size}')
                    # Initialize data loaders with the current batch size for training and validation datasets.
                    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
                    # Reset visualization environment and metrics for each configuration.
                    if has_started:
                        vis.delete_env('Astro Classifier')
                        vis.close()
                        viz.reset_x_axis('Batch Loss')
                        viz.reset_x_axis('Validation Loss')
                        viz.reset_x_axis('Validation Accuracy')
                        viz.reset_x_axis('Precision')
                        viz.reset_x_axis('Recall')
                        viz.reset_x_axis('F1-scores')
                        viz.reset_x_axis('Specificity')
                    else:
                        has_started = True

                    # Ensuring reproducibility by setting a fixed seed and deterministic behavior.
                    torch.manual_seed(3908274)
                    torch.backends.cudnn.deterministic = True

                    # Initialize model weights and move the model to the GPU.
                    model = CelebrityClassifier(resnet50, vit, num_classes)
                    #model.apply(reset_model)

                    # Save the initial state
                    initial_state_dict = model.state_dict()

                    model.to(device)

                    # Set up the loss function and optimizer with hyperparameters.
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                    # Configure schedulers for dynamic learning rate adjustment.
                    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
                    scheduler_by_accuracy = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
                    scheduler_by_valloss = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

                    # Set up early stopping mechanisms based on different performance metrics.
                    early_stopping_batch = EarlyStoppingBatch(patience=40)
                    early_stopping_valloss = EarlyStoppingValLoss(patience=10)
                    early_stopping_accuracy = EarlyStoppingAccuracy(patience=5)

                    # Training and validation loop with exception handling for memory issues.
                    try:
                        train_and_validate(model, train_dataloader, validation_dataloader, criterion, optimizer, max_norm)
                    except RuntimeError as err:
                        if 'out of memory' in str(err):
                            print(f'WARNING: Out of {device} memory. Skipping grid element')
                            # Handle the out-of-memory issue here, e.g., by reducing batch size or skipping
                            continue
                        else:
                            raise err  # Re-raise the exception if it's not a memory error

                    # Evaluate the model and update best_hyperparameters if this model performs the best.
                    _, current_accuracy = validate(model, validation_dataloader)
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        best_hyperparameters = (step_size, learning_rate, max_norm, weight_decay, batch_size)
                        #                           0            1           2             3          4
                        print(f'\nBest accuracy by now = {best_accuracy}\n')

                    # Later, to reset the model to the saved state:
                    model.load_state_dict(initial_state_dict)

# End of all hyperparameter tuning loops.

# Print out the best hyperparameter set and its performance
print('\nBest Hyperparameters:')
print(f'Step size = {best_hyperparameters[0]}')                         # 0
print(f'Learning rate = {best_hyperparameters[1]}')                     # 1
print(f'Max norm for gradients clipping = {best_hyperparameters[2]}')   # 2
print(f'weight decay = {best_hyperparameters[3]}')                      # 3
print(f'Batch Size = {best_hyperparameters[4]}')                        # 4

print(f'\nBest Accuracy: {best_accuracy}\n')
