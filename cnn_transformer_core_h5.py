# Author: Eraldo Pereira Marinho, Ph.D
# About: the code is a core module to build a VGG like CNN with transformer, originally design to classify astronomical images
# Creation: Aug 29, 2023
# Usage, import cnn_transformer_core and its components therein

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import h5py

nmaxpool = 4
img_width = 256
img_height = 256
img_out_width = img_width // 2**nmaxpool
img_out_height = img_height // 2**nmaxpool

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

cnn_pre_classification = 64 # este é o número de classes intermediárias como saída do modelo CNN

# Parâmetros do Transformer Encoder
embedding_dimension = cnn_pre_classification # dimensão do espaço de recursos, que é uma dimensão importante para a atenção

# Definindo a classe do modelo CNN + Transformer
class CNNTransformer(nn.Module):
    # def __init__(self, cnn_model, num_dense_layers=1):
    def __init__(self,
                 cnn_model,
                 num_heads = 16,
                 transformer_layers = 2, # número de camadas de atenção do Transformer Encoder
                 num_dense_layers = 1,
        ):
        super(CNNTransformer, self).__init__()
        self.cnn_model = cnn_model

        # Configuração do Transformer Encoder
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=embedding_dimension, nhead=num_heads, activation=F.relu),
            num_layers=transformer_layers
        )

        # Adicionando camadas densas para classificação após o CNN Transformer
        dense_layers = []
        input_size = embedding_dimension
        output_size_1 = 256
        output_size_2 = 128
        assert num_dense_layers > 0, "number of dense layers must be greater than or equal to 1"

        for _ in range(num_dense_layers):
            dense_layers.append(nn.Linear(input_size, output_size_1))
            dense_layers.append(nn.ReLU())
            input_size = output_size_1

        dense_layers.append(nn.Linear(output_size_1, output_size_2))
        dense_layers.append(nn.ReLU())
        input_size = output_size_2
        self.dense_layers = nn.Sequential(*dense_layers)
        self.fc = nn.Linear(output_size_2, num_classes)
        # self.fc = nn.Linear(embedding_dimension, num_classes)

    def forward(self, x):
        # Extração de características usando a CNN
        features = self.cnn_model(x)

        # Preparação das características para entrada no Transformer
        features = features.view(features.size(0), features.size(1), -1)  # Achata as características
        features = features.permute(2, 0, 1)  # Reordena para o formato adequado para o Transformer

        # Aplicação do Transformer Encoder nas características
        transformed_features = self.transformer(features)

        # Revertendo a forma das características para o formato original
        transformed_features = transformed_features.permute(1, 2, 0)
        transformed_features = transformed_features.contiguous().view(transformed_features.size(0), -1)

        # Passagem das características pelo conjunto de camadas densas
        transformed_features = self.dense_layers(transformed_features)

        # Camada final de classificação
        output = self.fc(transformed_features)
        return output

# Observação:
"""
# Para classificação de imagens, uma camada Decoder não é necessária. O Encoder do Transformer é usado para extrair recursos úteis da imagem
# e as camadas de classificação subsequentes são usadas para fazer a predição das classes. Este é um design adequado para tarefas de
# classificação de imagem, incluindo a classificação de imagens astronômicas
"""

cnn_n_out_1 = 16
cnn_n_out_2 = 32
cnn_n_out_3 = 64
cnn_n_out_4 = 128
dense_l_1 = 512
dense_l_2 = 256
dense_l_3 = 128

# Definindo a arquitetura da CNN para extração de características
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Camadas convolucionais para extrair características das imagens
        self.features = nn.Sequential(
            # Bloco convolutivo 1 - saída maxpool tem metade das dimensões lineares da imagem de entrada redimensionada
            nn.Conv2d(3, cnn_n_out_1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_n_out_1),  # Normalização por lotes para estabilizar o treinamento
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # Camada de max pooling para reduzir a resolução
            nn.AdaptiveMaxPool2d((img_width // 2, img_height // 2)),

            # Bloco convolutivo 2 - saída maxpool tem 1/4 das dimensões lineares da imagem de entrada redimensionada
            nn.Conv2d(cnn_n_out_1, cnn_n_out_2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_n_out_2),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveMaxPool2d((img_width // 4, img_height // 4)),

            # Bloco convolutivo 3
            nn.Conv2d(cnn_n_out_2, cnn_n_out_3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_n_out_3),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveMaxPool2d((img_width // 8, img_height // 8)),

            # Bloco convolutivo 4
            nn.Conv2d(cnn_n_out_3, cnn_n_out_4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_n_out_4),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveMaxPool2d((img_width // 16, img_height // 16)),

        )

        # Camadas densas para pré-classificação, a serem usadas como dimensão de embedding para o Transformer
        dropout = 0.5
        expected_flattened_size = cnn_n_out_4 * img_out_width * img_out_height

        self.classifier = nn.Sequential(
            # Primeira camada densa
            nn.Linear(expected_flattened_size, dense_l_1),  # Conecta todas as características a uma camada densa
            nn.Dropout(p=dropout),  # Dropout para evitar overfitting
            nn.ReLU(),

            # Segunda camada densa
            nn.Linear(dense_l_1, dense_l_2),
            nn.Dropout(p=dropout),
            nn.ReLU(),

            # Terceiraa camada densa
            nn.Linear(dense_l_2, dense_l_3),
            nn.Dropout(p=dropout),
            nn.ReLU(),

            # Camada de saída da CNN usada como embedding dimension para o Transformer
            nn.Linear(dense_l_3, cnn_pre_classification),  # Dimensão de embedding para o Transformer
            nn.Dropout(p=dropout),  # Dropout para regularização,
            nn.ReLU(),
        )

    def forward(self, x):
        # Propagação das imagens através das camadas convolucionais
        x = self.features(x)

        # Redimensionamento das saídas para serem compatíveis com as camadas densas
        x = x.view(x.size(0), -1)

        # Propagação através das camadas densas para a pré-classificação
        x = self.classifier(x)

        return x

# Instanciar a CNN + Transformer
# model = CNNTransformer(cnn_model, transformer_layers, num_classes=len(train_dataset.classes))
num_classes = len(class_labels)
cnn_model = CNN()
model = CNNTransformer(cnn_model,
                       num_heads = 16,
                       transformer_layers = 2,
                       num_dense_layers = 1)
