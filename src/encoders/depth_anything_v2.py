from torchvision.transforms import InterpolationMode, transforms
from transformers import pipeline

from src.encoders.strategy import EncoderStrategy


class DepthAnythingV2Strategy(EncoderStrategy):
    def build(self, **config):
        pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device="cpu",
        )
        model = pipe.model.backbone
        # transform = pipe.transform
        transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=InterpolationMode.BILINEAR),
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
        outputs = model(data)
        # Extract the CLS token features from the last feature map
        return outputs["feature_maps"][-1][:, 0, :]
