""" Code: cnn-test-transformer-h5-v3.py (main code for test)"""

# Author: Eraldo Pereira Marinho, Ph.D
# About: The code imports cnn_transformer_core to validate the pre-trained classification of astronomical images
# Creation: Aug 29, 2023

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
# import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import pillow_avif
from utils import Visualizer  # You should import your Visualizer module here
import os
# from cnn_transformer_core_h5_v3 import img_width as imgw
# from cnn_transformer_core_h5_v3 import img_height as imgh
from cnn_transformer_core_h5_v3 import CNN, CNNTransformer
# from cnn_transformer_core_h5_v3 import transform

# Transformer Encoder Parameters
#embedding_dimension = 128 # Dimension of the feature space, which is an important dimension for encoder attention
embedding_dimension = 128

# Full connected layers
# dense_dims = [1024, 512, 256] # List of output dimensions for dense layers # The best for unsorted astronomical image classification
fc_out_dim = embedding_dimension
dense_dims = [fc_out_dim * 4, fc_out_dim * 2, fc_out_dim] # List of output dimensions for dense layers # The best for unsorted astronomical image classification

# Convolutional layers
cnn_out_dim = 2 * dense_dims[0]
cnn_out_dims = [cnn_out_dim // 8, cnn_out_dim // 4, cnn_out_dim // 2, cnn_out_dim] # List of output dimensions for convolutional layers
# cnn_out_dims = [128, 256, 512, 1024] # List of output dimensions for convolutional layers # The best for unsorted astronomical image classification

# Replace 'class_labels' with your actual class labels
class_labels = ['Angelina Jolie', 'Brad Pitt', 'Denzel Washington', 'Hugh Jackman', 'Jennifer Lawrence',
                'Johnny Depp', 'Kate Winslet', 'Keira Knightley', 'Kristen Bell', 'Leonardo DiCaprio', 'Megan Fox', 'Natalie Portman',
                'Nicole Kidman', 'Paris Hilton', 'Robert Downey Jr', 'Sandra Bullock', 'Scarlett Johansson', 'Tom Cruise',
                'Tom Hanks', 'Will Smith']

num_classes = len(class_labels)

# Instantiate the CNN + Dense layer + Transformer
cnn_model = CNN(cnn_out_dims, dense_dims, embedding_dimension, dropout=0.3, num_classes=num_classes)
# Instantiate the composed CNN+Transformer network
model = CNNTransformer(cnn_model, num_heads=8, transformer_layers=6, num_dense_layers=0, embedding_dimension=embedding_dimension, num_classes=num_classes)

# Load the saved model parameters
saved_model_path = 'trained_cnn_model.pth'
model.load_state_dict(torch.load(saved_model_path))

# Move the model to the same device as the input
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the model to evaluation mode
model.eval()

# # Transformations for preprocessing the input image - it might be different from transform within cnn_transformer_core
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),                        # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image tensor RGB color
])

# Path to the validation images directory
validation_dir = 'images/tests'

correct_count = 0
images = os.listdir(validation_dir)
total_images = len(images)
print(f"Total test images = {total_images}")

def get_yes_or_no():
    while True:
        answer = input("Please enter 'Y' or 'N': ")
        if answer.lower() in ['y', 'n']:
            return answer
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")

# Iterate through images in the validation directory
for filename in os.listdir(validation_dir):
    image_path = os.path.join(validation_dir, filename)

    print(f"Processing image: {image_path}")

    # Load and preprocess the image
    sample_image = Image.open(image_path).convert("RGB")
    input_image = transform(sample_image).unsqueeze(0)  # Add an extra dimension for the batch
    input_image = input_image.to(device)  # Move to the same device as the model

    # Perform inference
    with torch.no_grad():
        output = model(input_image)

    # Get the predicted class index
    predicted_class_index = torch.argmax(output).item()

    # Print the inferred class
    print(f"Inferred class: {class_labels[predicted_class_index]}")

    # Display the image using matplotlib
    plt.imshow(sample_image)
    plt.show()

    #response = input("Is the classification correct? (Y/N) / Ctr-C to exit: ").strip()
    #if response.lower() == "y":
        #correct_count += 1
    print("Is the classification correct?", end=' ')
    response = get_yes_or_no()
    if response == 'y':
        correct_count += 1



print(f"Accuracy percentage = 100% * {correct_count} / {total_images} = {100 * correct_count / total_images:.2f}%")
