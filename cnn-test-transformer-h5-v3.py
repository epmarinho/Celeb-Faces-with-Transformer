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
from cnn_transformer_core_h5_v3 import model as loaded_model
# from cnn_transformer_core_h5_v3 import transform

# Load the saved model parameters
saved_model_path = 'trained_cnn_model.pth'
loaded_model.load_state_dict(torch.load(saved_model_path))

# Move the model to the same device as the input
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)

# Set the model to evaluation mode
loaded_model.eval()

# # Transformations for preprocessing the input image - it might be different from transform within cnn_transformer_core
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),                        # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image tensor
])

# Replace 'class_labels' with your actual class labels
class_labels = ['galaxies', 'globular clusters', 'nebulae', 'open clusters']

# Path to the validation images directory
validation_dir = 'images/tests'

correct_count = 0
images = os.listdir(validation_dir)
total_images = len(images)
print(f"Total test images = {total_images}")

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
        output = loaded_model(input_image)

    # Get the predicted class index
    predicted_class_index = torch.argmax(output).item()

    # Print the inferred class
    print(f"Inferred class: {class_labels[predicted_class_index]}")

    # Display the image using matplotlib
    plt.imshow(sample_image)
    plt.show()

    response = input("Is the classification correct? (Y/N): ").strip()
    if response.lower() == "y":
        correct_count += 1

print(f"Accuracy percentage = {100 * correct_count / total_images} %")
