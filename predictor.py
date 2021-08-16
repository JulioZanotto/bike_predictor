import torch
import shutil
from glob import glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torchvision.models as models

import torch.nn as nn

from utils import configuration
from utils.general import test_transforms

# Pipeline
# Cria os folders baseado nas classes
labs = configuration.label_dict
Path("/predictions/").mkdir(exist_ok=True)

for key, value in labs:
    Path("/predictions/" + value).mkdir(parents=True, exist_ok=True)

# Le o diretorio com as imagens
image_folder = glob(configuration.image_path)

# Carrega o modelo
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    device = 'cpu'
    print('CUDA is not available.  Training on CPU ...')
else:
    device = 'cuda'
    print('CUDA is available!  Training on GPU ...')

model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(nn.Linear(512, 6))

# Load model
checkpoint = torch.load(
        'model_best.pt', map_location='cpu'
                            )
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)

# Pega as imagens e passa pelo modelo
for image in tqdm(image_folder):
    img_read = Image.open(image)
    # Transform Image
    img_transf = test_transforms(img_read)
    # Predict Image
    ia_result = model(img_transf.unsqueeze(0))
    # baseado no output
    _, pred = torch.max(ia_result, 1)
    # Move a imagem para o folder
    folder = labs[pred.detach().cpu()]
    shutil.move(image, './predictions/' + folder)
