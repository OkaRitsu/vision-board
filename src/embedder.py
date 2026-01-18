from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

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

    def embed(self, dataset_df: pd.DataFrame, encoder_config, reducer_config):
        model, transform = self.encoder_strategy.build(**encoder_config)
        reducer = self.reducer_strategy.build(**reducer_config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        feature_batches: List[np.ndarray] = []

        # Encode images
        with torch.no_grad():
            for row in tqdm(
                dataset_df.itertuples(),
                total=len(dataset_df),
                desc="Encoding images",
            ):
                image = Image.open(f"static/{row.filename}").convert("RGB")
                images = transform(image).unsqueeze(0)
                images = images.to(device)

                features = self.encoder_strategy.encode(model, images)
                features = features.cpu().numpy()
                feature_batches.append(features)
        vectors = np.vstack(feature_batches)

        # Reduce dimensions
        coords = self.reducer_strategy.reduce(vectors, reducer)
        dataset_df["x"] = coords[:, 0]
        dataset_df["y"] = coords[:, 1]
        return dataset_df
