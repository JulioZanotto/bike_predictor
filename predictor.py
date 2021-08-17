import torch
import shutil
from glob import glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from utils import configuration
from utils.general import test_transforms, load_model

# Pipeline
# Cria os folders baseado nas classes
labs = configuration.label_dict
Path("./predictions/").mkdir(exist_ok=True)

for key, value in labs.items():
    Path("./predictions/" + value).mkdir(parents=True, exist_ok=True)

# Le o diretorio com as imagens
image_folder = glob(configuration.image_path + '*.*')

# Carrega o modelo
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    device = 'cpu'
    print('CUDA is not available.  Training on CPU ...')
else:
    device = 'cuda'
    print('CUDA is available!  Training on GPU ...')

model = load_model('modelo_adam.pt')
model.to(device)

# Pega as imagens e passa pelo modelo
for image in tqdm(image_folder):
    img_read = Image.open(image)
    # Transform Image
    img_transf = test_transforms(img_read)
    # Predict Image
    ia_result = model(img_transf.unsqueeze(0).to(device))
    # baseado no output
    _, pred = torch.max(ia_result, 1)
    # Move a imagem para o folder
    folder = labs[pred.detach().cpu().item()]
    shutil.move(image, './predictions/' + folder)
