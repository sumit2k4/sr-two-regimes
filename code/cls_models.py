"""Scene-classification backbones. Three architecturally distinct ImageNet
pre-trained CNNs are used as ensemble members, matching proposal objective 4."""
import torch, torch.nn as nn
import torchvision.models as tvm

BACKBONES = ["convnext_tiny", "resnet18", "efficientnet_b0", "densenet121"]

# smallest square input each backbone can ingest
MIN_INPUT = {"convnext_tiny": 32, "convnext_small": 32, "resnet18": 8,
             "efficientnet_b0": 16, "densenet121": 32}


def build_cls(name, n_classes, pretrained=True):
    if name == "resnet18":
        m = tvm.resnet18(pretrained=pretrained)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif name == "resnet50":
        m = tvm.resnet50(pretrained=pretrained)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif name.startswith("convnext"):
        m = getattr(tvm, name)(pretrained=pretrained)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_classes)
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(pretrained=pretrained)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif name == "densenet121":
        m = tvm.densenet121(pretrained=pretrained)
        m.classifier = nn.Linear(m.classifier.in_features, n_classes)
    else:
        raise ValueError(name)
    return m


IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
