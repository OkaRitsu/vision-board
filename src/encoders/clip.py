import open_clip

from src.encoders.strategy import EncoderStrategy


class CLIPStrategy(EncoderStrategy):
    def build(self, **config):
        model_name = config.get("clip_model", "ViT-B-32")
        pretrained = config.get("pretrained", "laion2b_s34b_b79k")
        model, _, transform = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        return model, transform

    def encode(self, model, data):
        return model.encode_image(data)
