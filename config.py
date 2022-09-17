import torch

class CFG:
    num_classes = 5
    device = torch.device("cpu")
    model_path = {
        "efficientnet_b2" : "./weights/effnet_b2.pth",
        "resnet50" : "./weights/resnet_50.pth",
        "resnext101" : "./weights/resnext_101.pth",
        "swin_transformer" : "./weights/swin_transformer.pth"
    }
    labels_map = ['COVID', 'Lung_Opacity', 'Normal', 'Pneunomia', 'Tuberculosis']