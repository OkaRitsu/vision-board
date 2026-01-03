import torch
from torchvision.transforms import InterpolationMode, transforms

from src.encoders.strategy import EncoderStrategy


class DinoV2Strategy(EncoderStrategy):
    def build(self, **config):
        model = torch.hub.load(
            "facebookresearch/dinov2",
            config.get("dinov2_variant", "dinov2_vits14_reg"),
        )
        dinov2_size = config.get("dinov2_image_size", 224)
        transform = transforms.Compose(
            [
                transforms.Resize(
                    dinov2_size, interpolation=InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(dinov2_size),
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
