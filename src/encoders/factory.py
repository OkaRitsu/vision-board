from src.encoders.clip import CLIPStrategy
from src.encoders.depth_anything_v2 import DepthAnythingV2Strategy
from src.encoders.dinov2 import DinoV2Strategy
from src.encoders.resnet import ResNetStrategy


class EncoderStrategyFactory:
    _strategy = {
        "resnet": ResNetStrategy,
        "clip": CLIPStrategy,
        "dinov2": DinoV2Strategy,
        "depth_anything_v2": DepthAnythingV2Strategy,
    }

    @classmethod
    def get_strategy(cls, encoder_type: str):
        if encoder_type not in cls._strategy:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        return cls._strategy[encoder_type]()

    @classmethod
    def list_strategies(cls):
        return list(cls._strategy.keys())
