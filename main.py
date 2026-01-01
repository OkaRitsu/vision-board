from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import open_clip
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


class ImageFolderWithPaths(datasets.ImageFolder):
    """ImageFolder variant that also returns each sample's absolute path."""

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode an ImageFolder dataset with ResNet18, CLIP, or DINOv3 and visualise a 2D projection."
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
        help="Directory where scatter plots and coordinates will be saved.",
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
        help="Optionally limit how many images are embedded (default uses all).",
    )
    parser.add_argument(
        "--no-imagenet-weights",
        action="store_true",
        help="Skip loading ImageNet weights and keep the randomly initialised encoder.",
    )
    parser.add_argument(
        "--encoder",
        choices=("resnet18", "clip", "dinov2"),
        default="resnet18",
        help="Backbone encoder used to extract features before dimensionality reduction.",
    )
    parser.add_argument(
        "--clip-model",
        default="ViT-B-32",
        help="Model architecture string passed to open_clip when --encoder=clip.",
    )
    parser.add_argument(
        "--clip-pretrained",
        default="laion2b_s34b_b79k",
        help="Pretrained weights identifier for open_clip when --encoder=clip.",
    )
    parser.add_argument(
        "--dinov2-variant",
        default="dinov2_vits14_reg",
        help="Torch Hub model name from facebookresearch/dinov2 when --encoder=dinov2.",
    )
    parser.add_argument(
        "--dinov2-image-size",
        type=int,
        default=224,
        help="Input resolution for DINOv2 preprocessing pipeline.",
    )
    parser.add_argument(
        "--method",
        choices=("pca", "tsne"),
        default="pca",
        help="Dimensionality reduction method for the 2D visualisation.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="Perplexity value for t-SNE (ignored when --method=pca).",
    )
    parser.add_argument(
        "--tsne-learning-rate",
        type=float,
        default=200.0,
        help="Learning rate for t-SNE optimisation (ignored when --method=pca).",
    )
    parser.add_argument(
        "--tsne-iterations",
        type=int,
        default=1000,
        help="Number of optimisation iterations for t-SNE (ignored when --method=pca).",
    )
    parser.add_argument(
        "--tsne-random-state",
        type=int,
        default=42,
        help="Random seed fed to t-SNE for reproducibility (ignored when --method=pca).",
    )
    return parser.parse_args()


def build_resnet_transforms(use_imagenet_weights: bool) -> transforms.Compose:
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


def build_dino_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_dataloader(
    data_root: Path, batch_size: int, num_workers: int, transform: transforms.Compose
) -> DataLoader:
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset directory '{data_root}' does not exist.")

    dataset = ImageFolderWithPaths(
        root=str(data_root),
        transform=transform,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in '{data_root}'.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )


def build_resnet_model(use_imagenet_weights: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if use_imagenet_weights else None
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    return model


def prepare_encoder_components(
    args: argparse.Namespace,
) -> Tuple[
    nn.Module,
    transforms.Compose,
    Callable[[nn.Module, torch.Tensor], torch.Tensor],
    torch.dtype,
]:
    encoder_choice = args.encoder

    if encoder_choice == "resnet18":
        use_imagenet_weights = not args.no_imagenet_weights
        transform = build_resnet_transforms(use_imagenet_weights)
        model = build_resnet_model(use_imagenet_weights)

        def forward_fn(m: nn.Module, batch: torch.Tensor) -> torch.Tensor:
            return m(batch)

        input_dtype = torch.float32
    elif encoder_choice == "clip":
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.clip_model,
            pretrained=args.clip_pretrained,
        )

        def forward_fn(m: nn.Module, batch: torch.Tensor) -> torch.Tensor:
            return m.encode_image(batch)

        transform = preprocess
        input_dtype = next(model.parameters()).dtype
    elif encoder_choice == "dinov2":
        model = torch.hub.load(
            "facebookresearch/dinov2",
            args.dinov2_variant,
            pretrained=True,
        )
        transform = build_dino_transforms(args.dinov2_image_size)

        def forward_fn(m: nn.Module, batch: torch.Tensor) -> torch.Tensor:
            outputs = m(batch, is_training=False)
            return standardise_model_output(outputs)

        input_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported encoder '{encoder_choice}'.")

    return model, transform, forward_fn, input_dtype


def standardise_model_output(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, (list, tuple)):
        for item in outputs:
            if isinstance(item, torch.Tensor):
                return item
        raise TypeError("Model returned a sequence without tensors.")
    if isinstance(outputs, dict):
        preferred_keys = [
            "x_norm_clstoken",
            "cls_token",
            "pooler_output",
            "embeddings",
            "features",
        ]
        for key in preferred_keys:
            value = outputs.get(key)
            if isinstance(value, torch.Tensor):
                return value
        for value in outputs.values():
            if isinstance(value, torch.Tensor):
                return value
        raise TypeError("Model returned a dict without tensor values.")
    raise TypeError(f"Unsupported output type {type(outputs)} from encoder.")


def encode_images(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: Optional[int],
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    input_dtype: torch.dtype,
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
            if images.dtype != input_dtype:
                images = images.to(dtype=input_dtype)
            outputs = forward_fn(model, images)
            batch_embeddings = outputs.float().cpu().numpy()
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


def project_embeddings_with_tsne(
    vectors: np.ndarray,
    perplexity: float,
    learning_rate: float,
    n_iter: int,
    random_state: int,
) -> np.ndarray:
    if vectors.shape[0] <= perplexity:
        raise ValueError(
            f"t-SNE perplexity ({perplexity}) must be smaller than the number of samples ({vectors.shape[0]})."
        )
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        init="pca",
        random_state=random_state,
        metric="euclidean",
    )
    coords = tsne.fit_transform(vectors)
    return coords


def save_projection_visualisation(
    method: str,
    encoder_label: str,
    coords: np.ndarray,
    labels: Sequence[int],
    paths: Sequence[str],
    class_names: Sequence[str],
    output_dir: Path,
) -> Tuple[Path, Path]:
    df = pd.DataFrame(
        {
            "dim1": coords[:, 0],
            "dim2": coords[:, 1],
            "label": [class_names[idx] for idx in labels],
            "filename": [Path(p).name for p in paths],
            "path": paths,
        }
    )
    axis_titles = {
        "pca": ("PC1", "PC2"),
        "tsne": ("Dim 1", "Dim 2"),
    }
    xaxis, yaxis = axis_titles.get(method, ("Dim 1", "Dim 2"))
    title = f"{method.upper()} projection of {encoder_label.upper()} embeddings"

    fig = px.scatter(
        df,
        x="dim1",
        y="dim2",
        color="label",
        hover_data=["filename", "path"],
        title=title,
    )
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis, legend_title="Class")

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{method}_scatter.html"
    csv_path = output_dir / f"{method}_projection.csv"
    fig.write_html(str(html_path))
    df.to_csv(csv_path, index=False)
    return html_path, csv_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    args = parse_args()

    model, transform, forward_fn, input_dtype = prepare_encoder_components(args)
    device = torch.device(args.device)
    dataloader = build_dataloader(
        data_root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=transform,
    )
    vectors, labels, paths = encode_images(
        model=model,
        dataloader=dataloader,
        device=device,
        max_samples=args.max_samples,
        forward_fn=forward_fn,
        input_dtype=input_dtype,
    )
    method = args.method
    if method == "pca":
        coords, variance_ratio = project_embeddings_with_pca(vectors, n_components=2)
        logging.info(
            "Explained variance | PC1: %.2f%%, PC2: %.2f%%",
            variance_ratio[0] * 100,
            variance_ratio[1] * 100,
        )
    else:
        coords = project_embeddings_with_tsne(
            vectors=vectors,
            perplexity=args.tsne_perplexity,
            learning_rate=args.tsne_learning_rate,
            n_iter=args.tsne_iterations,
            random_state=args.tsne_random_state,
        )
        logging.info(
            "t-SNE parameters | perplexity %.1f | learning rate %.1f | iterations %d",
            args.tsne_perplexity,
            args.tsne_learning_rate,
            args.tsne_iterations,
        )
    html_path, csv_path = save_projection_visualisation(
        method=method,
        encoder_label=args.encoder,
        coords=coords,
        labels=labels,
        paths=paths,
        class_names=dataloader.dataset.classes,
        output_dir=args.output_dir,
    )
    logging.info("Saved interactive %s scatter plot to %s", method.upper(), html_path)
    logging.info("Saved %s coordinates to %s", method.upper(), csv_path)


def app():
    st.sidebar.title("Configuration")

    st.sidebar.subheader("Data")
    data_dir = st.sidebar.text_input("Dataset directory", "data/images")

    st.sidebar.subheader("Encoder")
    # st.sidebar.write("Select the encoder model for feature extraction:")
    encoder = st.sidebar.selectbox(
        "Encoder type",
        ("resnet18", "clip", "dinov2"),
        label_visibility="collapsed",
    )
    encoder_config = {}
    if encoder == "clip":
        clip_model = st.sidebar.text_input("CLIP model", "ViT-B-32")
        clip_pretrained = st.sidebar.text_input(
            "CLIP pretrained weights", "laion2b_s34b_b79k"
        )
        encoder_config["clip_model"] = clip_model
        encoder_config["clip_pretrained"] = clip_pretrained
    elif encoder == "dinov2":
        dinov2_variant = st.sidebar.text_input("DINOv2 variant", "dinov2_vits14_reg")
        dinov2_image_size = st.sidebar.number_input(
            "DINOv2 image size", min_value=64, max_value=1024, value=224, step=1
        )
        encoder_config["dinov2_variant"] = dinov2_variant
        encoder_config["dinov2_image_size"] = dinov2_image_size

    st.sidebar.subheader("Dimensionality Reduction")
    st.sidebar.write("Choose a method to reduce embedding dimensions:")
    dim_reduction = st.sidebar.selectbox(
        "Reduction method",
        ("pca", "tsne"),
        label_visibility="collapsed",
    )
    dim_reduction_config = {}
    if dim_reduction == "tsne":
        tsne_perplexity = st.sidebar.number_input(
            "t-SNE perplexity", min_value=5.0, max_value=100.0, value=30.0, step=1.0
        )
        tsne_learning_rate = st.sidebar.number_input(
            "t-SNE learning rate",
            min_value=10.0,
            max_value=1000.0,
            value=200.0,
            step=10.0,
        )
        tsne_iterations = st.sidebar.number_input(
            "t-SNE iterations", min_value=250, max_value=5000, value=1000, step=250
        )
        dim_reduction_config["tsne_perplexity"] = tsne_perplexity
        dim_reduction_config["tsne_learning_rate"] = tsne_learning_rate
        dim_reduction_config["tsne_iterations"] = tsne_iterations

    st.title("Vision Board")
    if st.sidebar.button("Run"):
        # Validate data directory
        data_path = Path(data_dir)
        if not data_path.exists():
            st.error(f"Dataset directory not found: {data_dir}")
            return

        # Prepare encoder
        if encoder == "resnet18":
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Identity()
            transform = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            forward_fn = lambda m, x: m(x)
            input_dtype = torch.float32
        elif encoder == "clip":
            model, _, transform = open_clip.create_model_and_transforms(
                encoder_config.get("clip_model", "ViT-B-32"),
                pretrained=encoder_config.get("clip_pretrained", "laion2b_s34b_b79k"),
            )
            forward_fn = lambda m, x: m.encode_image(x)
            input_dtype = torch.float32
        elif encoder == "dinov2":
            model = torch.hub.load(
                "facebookresearch/dinov2",
                encoder_config.get("dinov2_variant", "dinov2_vits14_reg"),
            )
            dinov2_size = encoder_config.get("dinov2_image_size", 224)
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
            forward_fn = lambda m, x: m(x)
            input_dtype = torch.float32

        # Build dataloader
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = build_dataloader(
            data_root=data_path,
            batch_size=32,
            num_workers=2,
            transform=transform,
        )

        # Encode images
        with st.spinner("Encoding images..."):
            vectors, labels, paths = encode_images(
                model=model,
                dataloader=dataloader,
                device=device,
                max_samples=None,
                forward_fn=forward_fn,
                input_dtype=input_dtype,
            )

        # Dimensionality reduction
        if dim_reduction == "pca":
            with st.spinner("Applying PCA..."):
                coords, variance_ratio = project_embeddings_with_pca(
                    vectors, n_components=2
                )
            st.success(
                f"PCA: Explained variance | PC1: {variance_ratio[0]*100:.2f}%, PC2: {variance_ratio[1]*100:.2f}%"
            )
        else:
            with st.spinner("Applying t-SNE..."):
                coords = project_embeddings_with_tsne(
                    vectors=vectors,
                    perplexity=dim_reduction_config.get("tsne_perplexity", 30.0),
                    learning_rate=dim_reduction_config.get("tsne_learning_rate", 200.0),
                    n_iter=dim_reduction_config.get("tsne_iterations", 1000),
                    random_state=42,
                )
            st.success("t-SNE completed")

        # Create visualization
        df = pd.DataFrame(
            {
                "dim1": coords[:, 0],
                "dim2": coords[:, 1],
                "label": [dataloader.dataset.classes[idx] for idx in labels],
                "filename": [Path(p).name for p in paths],
                "path": paths,
            }
        )

        axis_titles = {
            "pca": ("PC1", "PC2"),
            "tsne": ("Dim 1", "Dim 2"),
        }
        xaxis, yaxis = axis_titles.get(dim_reduction, ("Dim 1", "Dim 2"))
        title = f"{dim_reduction.upper()} projection of {encoder.upper()} embeddings"

        fig = px.scatter(
            df,
            x="dim1",
            y="dim2",
            color="label",
            hover_data=["filename", "path"],
            title=title,
        )
        fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis, legend_title="Class")

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    # main()
    app()
