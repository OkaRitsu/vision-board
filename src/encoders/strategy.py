from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.transforms as transforms


class EncoderStrategy(ABC):
    @abstractmethod
    def build(self, **config) -> Tuple[nn.Module, transforms.Compose]:
        pass

    @abstractmethod
    def encode(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        pass
