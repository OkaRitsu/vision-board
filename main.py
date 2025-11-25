from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm


class ImageFolderWithPaths(datasets.ImageFolder):
    """ImageFolder variant that also returns each sample's absolute path."""

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode an ImageFolder dataset with ResNet18 and visualise a 2D PCA projection."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the dataset root arranged like torchvision.datasets.ImageFolder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the PCA scatter plot and coordinates will be saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size used while encoding images.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (e.g. cuda, mps, cpu).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optionally limit how many images are used for PCA (default uses all).",
    )
    parser.add_argument(
        "--no-imagenet-weights",
        action="store_true",
        help="Skip loading ImageNet weights and keep the randomly initialised encoder.",
    )
    return parser.parse_args()


def build_transforms(use_imagenet_weights: bool) -> transforms.Compose:
    if use_imagenet_weights:
        return ResNet18_Weights.DEFAULT.transforms()

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_dataloader(
    data_root: Path, batch_size: int, num_workers: int, use_imagenet_weights: bool
) -> DataLoader:
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset directory '{data_root}' does not exist.")

    dataset = ImageFolderWithPaths(
        root=str(data_root),
        transform=build_transforms(use_imagenet_weights),
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in '{data_root}'.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )


def build_model(model_name: str, use_imagenet_weights: bool) -> nn.Module:
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if use_imagenet_weights else None
        model = resnet18(weights=weights)
        model.fc = nn.Identity()
    else:
        raise ValueError(f"Unsupported model name '{model_name}'.")
    return model


def encode_images(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: Optional[int],
) -> Tuple[np.ndarray, List[int], List[str]]:
    if max_samples is not None and max_samples <= 0:
        raise ValueError("--max-samples must be greater than zero.")

    model.to(device)
    model.eval()
    feature_batches: List[np.ndarray] = []
    class_indices: List[int] = []
    image_paths: List[str] = []
    collected = 0

    with torch.no_grad():
        for images, targets, paths in tqdm(dataloader, desc="Encoding images"):
            images = images.to(device)
            outputs = model(images)
            batch_embeddings = outputs.cpu().numpy()
            batch_targets = targets.tolist()
            batch_paths = list(paths)

            if max_samples is not None:
                remaining = max_samples - collected
                if remaining <= 0:
                    break
                batch_embeddings = batch_embeddings[:remaining]
                batch_targets = batch_targets[:remaining]
                batch_paths = batch_paths[:remaining]

            feature_batches.append(batch_embeddings)
            class_indices.extend(batch_targets)
            image_paths.extend(batch_paths)
            collected += len(batch_embeddings)

            if max_samples is not None and collected >= max_samples:
                break

    if not feature_batches:
        raise RuntimeError("No embeddings were extracted from the dataset.")

    vectors = np.vstack(feature_batches)
    return vectors, class_indices, image_paths


def project_embeddings_with_pca(
    vectors: np.ndarray, n_components: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    if vectors.shape[0] < n_components:
        raise ValueError(
            f"PCA requires at least {n_components} samples, got {vectors.shape[0]}."
        )

    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(vectors)
    return coords, pca.explained_variance_ratio_


def save_pca_visualisation(
    coords: np.ndarray,
    labels: Sequence[int],
    paths: Sequence[str],
    class_names: Sequence[str],
    output_dir: Path,
) -> Tuple[Path, Path]:
    df = pd.DataFrame(
        {
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "label": [class_names[idx] for idx in labels],
            "filename": [Path(p).name for p in paths],
            "path": paths,
        }
    )
    fig = px.scatter(
        df,
        x="pc1",
        y="pc2",
        color="label",
        hover_data=["filename", "path"],
        title="PCA projection of ResNet18 embeddings",
    )
    fig.update_layout(xaxis_title="PC1", yaxis_title="PC2", legend_title="Class")

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "pca_scatter.html"
    csv_path = output_dir / "pca_projection.csv"
    fig.write_html(str(html_path))
    df.to_csv(csv_path, index=False)
    return html_path, csv_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    args = parse_args()

    use_imagenet_weights = not args.no_imagenet_weights
    dataloader = build_dataloader(
        data_root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_imagenet_weights=use_imagenet_weights,
    )
    model = build_model(
        model_name="resnet18",
        use_imagenet_weights=use_imagenet_weights,
    )
    vectors, labels, paths = encode_images(
        model=model,
        dataloader=dataloader,
        device=torch.device(args.device),
        max_samples=args.max_samples,
    )
    coords, variance_ratio = project_embeddings_with_pca(vectors, n_components=2)
    logging.info(
        "Explained variance | PC1: %.2f%%, PC2: %.2f%%",
        variance_ratio[0] * 100,
        variance_ratio[1] * 100,
    )
    html_path, csv_path = save_pca_visualisation(
        coords=coords,
        labels=labels,
        paths=paths,
        class_names=dataloader.dataset.classes,
        output_dir=args.output_dir,
    )
    logging.info("Saved interactive scatter plot to %s", html_path)
    logging.info("Saved PCA coordinates to %s", csv_path)


if __name__ == "__main__":
    main()
