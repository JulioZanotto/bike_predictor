import torch
import albumentations as A
from albumentations.pytorch import ToTensor
import torchvision.models as models
import torch.nn as nn
import os
import pandas as pd

test_transforms = A.Compose([
    A.Resize(640, 640),
    ToTensor()
    ])


def load_model(checkpoint_path):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(nn.Dropout(0.2),
                             nn.Linear(512, 7))

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model


def generate_csv(path):
    data = []
    for folder in sorted(os.listdir(path)):
        for file in sorted(os.listdir(path + folder)):
            data.append((file, folder))

    df = pd.DataFrame(data, columns=['imagem', 'classe'])

    df.to_csv('dataset.csv', index=False)

    return df
