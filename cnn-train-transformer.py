# Author: Eraldo Pereira Marinho, Ph.D
# About: the code imports cnn_transformer_core to allow Transformer+CNN to classify astronomical images
# Creation: Jul 12, 2023

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
# import torchvision.transforms as transforms
import visdom
from utils import Visualizer
from cnn_transformer_core import model
# from cnn_transformer_core import learning_rate
from cnn_transformer_core import train_loader
from cnn_transformer_core import validation_loader
# from cnn_transformer_core import num_epochs
import os
import torch.nn.init as init
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"pytorch device: {device}")

viz = Visualizer.Visualizer('Astro Classifier', use_incoming_socket=False)

# Definir o vetor de pesos das classes obtido empiricamente da última execução:
galaxies = np.float32(1/780)
globular = np.float32(1/295)
nebulae  = np.float32(1/409)
openclust= np.float32(1/147)
norm_denominator=galaxies + globular + nebulae + openclust
weight_class_0=galaxies/norm_denominator
weight_class_1=globular/norm_denominator
weight_class_2=nebulae/norm_denominator
weight_class_3=openclust/norm_denominator
# Instantiate the class weight tensor
class_weights = torch.tensor([weight_class_0, weight_class_1, weight_class_2, weight_class_3])
print(f"Class weights = {class_weights}")
# Weights tensor must be converted to the adopted device
class_weights = class_weights.to(device)

# Parâmetros de treinamento
num_epochs = 120
initial_learning_rate = 1e-4 # Valores menores deram pau
# Definir a função de perda e o otimizador
weight_decay = 1e-5 # Este é um valor razoável
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=weight_decay)
# Defina um scheduler para ajustar a taxa de aprendizado
# Aqui, um scheduler StepLR é usado, que reduz a taxa de aprendizado por um fator gamma após um número fixo de épocas
# Você pode ajustar o fator gamma e o período conforme necessário
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Verifique se o arquivo pré-treinado existe
model_checkpoint = "trained_cnn_model.pth"
if os.path.exists(model_checkpoint):
    # Carregue os pesos pré-treinados
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint)
    print("Pesos pré-treinados carregados com sucesso.")
else:
    print("Nenhum arquivo de pesos pré-treinados encontrado. Inicializando com pesos padrão do PyTorch.")
    # Inicialização de He em PyTorch
    # Acesse todas as camadas lineares (fully connected) em seu modelo
    # Certifique-se de que o modelo contém apenas camadas que devem ser inicializadas com He
    # for layer in model.children():
    #     if isinstance(layer, nn.Linear):
    #         init.kaiming_normal_(layer.weight)

# Mover o modelo para o dispositivo GPU
model.to(device)

# Função de treinamento
def train(model, dataloader, validation_loader, criterion, optimizer, num_epochs):
    model.train()  # Configurar o modelo para o modo de treinamento

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in dataloader:
            # Mover as imagens e rótulos para o dispositivo GPU
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

            optimizer.step()

            # Imprima ou registre os gradientes das camadas intermediárias
            # for name, param in model.named_parameters():
            #     if param.requires_grad and 'weight' in name:
            #         grdnorm = param.grad.norm().item()
            #         if grdnorm < 0.01:
            #             print(f'Layer: {name}, Grad norm: {grdnorm}')

            running_loss += loss.item() * images.size(0)

        # Atualize a taxa de aprendizado com base no scheduler
        scheduler.step()

        if epoch % 10 == 0:
            test(model, validation_loader)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        viz.plot_lines('batch loss', epoch_loss)

# predicted_labels array is used to construct a histogram to reveal how many times each class was used along the evaluation
predicted_labels = []

# Função de validação
def test(model, dataloader):
    model.eval()  # Configurar o modelo para o modo de avaliação
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            # Mover as imagens e rótulos para o dispositivo GPU
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            predicted_labels.extend(predicted.tolist())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    viz.plot_lines('test acuracy', accuracy)

# Mover o modelo para o dispositivo GPU antes do treinamento
model.to(device)

# Treinamento e teste da CNN
train(model, train_loader, validation_loader, criterion, optimizer, num_epochs)
test(model, validation_loader)

# Save the trained model
saved_model_path = 'trained_cnn_model.pth'
torch.save(model.state_dict(), saved_model_path)
# print(f"model.state_dict '{model.state_dict()}'")
print(f"Trained model saved to '{saved_model_path}'")

# plot the histogram for predicted categories - unbalanced histogram means low quality training
unique_labels = set(predicted_labels)
print("unique labels: ", unique_labels)
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
indices = [label_to_idx[label] for label in predicted_labels]
# print("índices: ", indices)

label_counts = torch.bincount(torch.tensor(indices))

print("label count: ", label_counts)

plt.bar(torch.arange(len(label_counts)), label_counts)
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.show()
