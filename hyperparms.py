import torch
import torch.nn as nn

nmaxpool = 3
img_width = 512
img_height = 512
img_out_width = img_width // 2**nmaxpool
img_out_height = img_height // 2**nmaxpool

# Parâmetros de treinamento
num_epochs = 200
batch_size = 32
learning_rate = 0.0001

num_classes = 4

# Definir a arquitetura da CNN
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(

            # bloco convolutivo 1
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # bloco convolutivo 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # bloco convolutivo 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # bloco convolutivo 4
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # bloco convolutivo 5
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # bloco convolutivo 6
            # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # camadas densas
        expected_flattened_size = 64 * img_out_width * img_out_height
        self.classifier = nn.Sequential(

            nn.Linear(expected_flattened_size, 512),
            nn.Dropout(),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.Dropout(),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.Dropout(),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.Dropout(),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.Dropout(),
            nn.ReLU(),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

