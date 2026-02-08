import hashlib
import json
from pathlib import Path
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
        cache_dir: str = "cache/features",
    ):
        self.encoder_strategy = encoder_strategy
        self.reducer_strategy = reducer_strategy
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, dataset_df, encoder_config):
        payload = {
            "files": dataset_df["filename"].tolist(),
            "encoder_config": encoder_config,
        }
        s = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(s).hexdigest()

    def embed(self, dataset_df, encoder_config, reducer_config):
        vectors = self.encode(dataset_df, encoder_config)
        reducer = self.reducer_strategy.build(**reducer_config)
        coords = self.reducer_strategy.reduce(vectors, reducer)
        dataset_df["x"] = coords[:, 0]
        dataset_df["y"] = coords[:, 1]
        return dataset_df

    def encode(self, dataset_df: pd.DataFrame, encoder_config):
        key = self._cache_key(dataset_df, encoder_config)
        cache_path = self.cache_dir / f"{key}.npz"
        if cache_path.exists():
            data = np.load(cache_path)
            return data["vectors"]
        model, transform = self.encoder_strategy.build(**encoder_config)
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
        np.savez_compressed(cache_path, vectors=vectors)
        return vectors
