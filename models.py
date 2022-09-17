import timm
from collections import OrderedDict
from config import CFG
import torch
import torch.nn as nn

def efficientnet_b2(num_classes):
    model = timm.create_model("efficientnet_b2", pretrained=False)
    classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1408, 512)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(512, 256)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(256, 5))
    ]))

    model.classifier = classifier
    return model

def resnet50(num_classes):
    model = timm.create_model("resnet50", pretrained=False)
    fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(2048, 1024)),
    ('relu1', nn.ReLU(inplace=True)),
    ('fc2', nn.Linear(1024, 256)),
    ('relu2', nn.ReLU(inplace=True)),
    ('fc3', nn.Linear(256, 5))
    ]))

    model.fc = fc
    return model

def resnext101(num_classes):
    model = timm.create_model('resnext101_32x8d', pretrained=False)
    fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(2048, 1024)),
    ('relu1', nn.ReLU(inplace=True)),
    ('fc2', nn.Linear(1024, 256)),
    ('relu2', nn.ReLU(inplace=True)),
    ('fc3', nn.Linear(256, 5))
    ]))

    model.fc = fc
    return model

def swin_transformer(num_classes):
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=False)
    head = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 512)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(512, 256)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(256, 5))
    ]))

    model.head = head
    return model

def create_model(model_name, num_classes=CFG.num_classes):
    assert model_name in ["efficientnet_b2", "resnet50", "resnext101", "swin_transformer"], "Invalid model name"
    if model_name == "efficientnet_b2":
        model = efficientnet_b2(num_classes)
    elif model_name == "resnet50":
        model = resnet50(num_classes)
    elif model_name == "resnext101":
        model = resnext101(num_classes)
    else:
        model = swin_transformer(num_classes)
    model = model.to(CFG.device)
    model.load_state_dict(torch.load(CFG.model_path[model_name]))
    return model

create_model("resnet50", 5) # Test OK