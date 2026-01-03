from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ImageFolderWithPaths(datasets.ImageFolder):
    """ImageFolder variant that also returns each sample's absolute path."""

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path


def build_dataloader(
    data_dir: Path, batch_size: int, num_workers: int, transform: transforms.Compose
) -> DataLoader:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory '{data_dir}' does not exist.")

    dataset = ImageFolderWithPaths(
        root=str(data_dir),
        transform=transform,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in '{data_dir}'.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
