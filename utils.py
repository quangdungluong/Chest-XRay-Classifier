from torchvision import transforms
from PIL import Image
import numpy as np
from config import CFG
from models import create_model
import torch
import matplotlib.pyplot as plt

def process_image(image_path, display=False):
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
        ])

    image = Image.open(image_path).convert('RGB')
    if display:
        image.show()
    image = data_transforms(image)
    image = image[None, :]
    image = image.to(CFG.device)
    return image
    
def predict(model, image):
    model.eval()
    output = model(image)
    probs = torch.nn.Softmax(dim=1)(output).detach().numpy()[0]
    _, prediction = output.max(1)
    return prediction, probs

def show_image(inp):
    print(inp.size())
    inp = inp[0].numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

## Test OK
# model = create_model("efficientnet_b2")
# image = process_image("./sample_data/Lung_Opacity/Lung_Opacity-169.png")
# prediction, probs = predict(model, image)
# print(prediction)
# print(probs)
# process_image("./sample_data/COVID/COVID-9.png", display=True) # Test OK