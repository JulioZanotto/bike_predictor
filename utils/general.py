import torchvision.transforms as transforms

test_transforms = transforms.Compose([transforms.Resize((640, 640)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
