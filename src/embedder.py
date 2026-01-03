from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from src.datasets import build_dataloader
from src.encoders.strategy import EncoderStrategy
from src.reducers.strategy import ReducerStrategy


class Embedder:
    def __init__(
        self,
        encoder_strategy: EncoderStrategy,
        reducer_strategy: ReducerStrategy,
    ):
        self.encoder_strategy = encoder_strategy
        self.reducer_strategy = reducer_strategy

    def embed(self, data_dir: str, encoder_config, reducer_config):
        model, transform = self.encoder_strategy.build(**encoder_config)
        reducer = self.reducer_strategy.build(**reducer_config)
        # Build dataloader
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = build_dataloader(
            data_dir=Path(data_dir),
            batch_size=32,
            num_workers=2,
            transform=transform,
        )
        model.to(device)
        model.eval()
        feature_batches: List[np.ndarray] = []
        class_labels: List[str] = []
        image_paths: List[str] = []
        collected = 0

        # Encode images
        with torch.no_grad():
            for images, targets, paths in tqdm(dataloader, desc="Encoding images"):
                images = images.to(device)
                features = self.encoder_strategy.encode(model, images)
                features = features.cpu().numpy()
                feature_batches.append(features)
                class_labels.extend([dataloader.dataset.classes[t] for t in targets])
                image_paths.extend(paths)
                collected += len(images)
                print(f"Collected {collected} samples...", end="\r")
        vectors = np.vstack(feature_batches)

        # Reduce dimensions
        coords = self.reducer_strategy.reduce(vectors, reducer)
        return coords, class_labels, image_paths
