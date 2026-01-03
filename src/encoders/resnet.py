import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

from src.encoders.strategy import EncoderStrategy


class ResNetStrategy(EncoderStrategy):
    def build(self, **config):
        model_name = config.get("model_name", "resnet18")
        pretrained = config.get("pretrained", True)

        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")

        # Remove the final classification layer
        model.fc = nn.Identity()

        # Define preprocessing transformations
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        return model, transform

    def encode(self, model, data):
        return model(data)
