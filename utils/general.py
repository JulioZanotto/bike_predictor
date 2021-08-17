import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

test_transforms = transforms.Compose([transforms.Resize((640, 640)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


def load_model(checkpoint_path):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(512, 6))

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model
