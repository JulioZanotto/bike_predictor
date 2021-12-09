import torch
import shutil
from glob import glob
from pathlib import Path
from tqdm import tqdm
import cv2

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

model.eval()
with torch.no_grad():

    # Pega as imagens e passa pelo modelo
    for image in tqdm(image_folder):
        img_read = cv2.imread(image)
        img_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
        # Transform Image
        img_transf = test_transforms(image=img_read)['image']
        # Predict Image
        ia_result = model(img_transf.unsqueeze(0).to(device))
        # baseado no output
        _, pred = torch.max(ia_result, 1)
        # Move a imagem para o folder
        folder = labs[pred.detach().cpu().item()]
        print(pred, folder)
        try:
            shutil.move(image, './predictions/' + folder)
        except Exception as e:
            print(e)

'''
# Obtendo um batch de imagens de teste
dataiter = iter(valid_loader)
samples = dataiter.next()
images = samples['image']
labels = samples['label']

# Pegando as saidas
output = model(images.cuda())
# Converte as probabilidades de saida para a classe predita
_, pred = torch.max(output, 1)
# prep as imagens para serem exibidas
images = images.cpu().numpy()

# Plota as imagens do batch, juntamente com as classes preditas e as classes corretas
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(4):
    ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]).transpose(1,2,0))
    ax.set_title("{} ({})".format(le.inverse_transform([pred[idx].detach().cpu()]), le.inverse_transform([labels[idx]])),
                 color=("green" if pred[idx]==labels[idx] else "red"))
'''