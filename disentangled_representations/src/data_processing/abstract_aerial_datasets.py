import pathlib
from abc import ABC, abstractmethod
from typing import override
import cv2
import numpy as np
import torch
import albumentations as A
from loguru import logger
import copy
import random


def _read_image(path: pathlib.Path, cv2_flag: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2_flag)
    if cv2_flag == cv2.IMREAD_COLOR:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def _get_image_paths_in_dir(dir_path: pathlib.Path) -> list[pathlib.Path]:
    image_paths = []
    for ext in ("png", "jpg"):
        image_paths.extend(dir_path.glob(f"*.{ext}"))
    image_paths = sorted(image_paths, key=lambda p: p.name.lower())

    if not image_paths:
        logger.warning(f"No images found in {dir_path}.")
    return image_paths


class SingleImagesDataset(torch.utils.data.Dataset, ABC):
    @abstractmethod
    def __getitem__(self, idx: int) -> torch.Tensor:
        pass


class SimpleSingleImagesDataset(SingleImagesDataset):
    def __init__(self, images_dir: pathlib.Path, cv2_read_flag: int, transform: A.Compose):
        super().__init__()

        self.image_paths = _get_image_paths_in_dir(images_dir)
        self.cv2_read_flag = cv2_read_flag
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    @override
    def __getitem__(self, idx: int) -> torch.Tensor:
        image = _read_image(self.image_paths[idx], self.cv2_read_flag)
        image = self.transform(image=image)["image"]
        return image


class PairedImagesDataset(torch.utils.data.Dataset, ABC):
    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class FilenameIDPairedImagesDataset(PairedImagesDataset):
    def __init__(self, images_A_dir: pathlib.Path, images_B_dir: pathlib.Path, cv2_read_flag: int,
                 shared_transform: A.Compose, unique_transform: A.Compose):
        super().__init__()

        self.image_paths_A = _get_image_paths_in_dir(images_A_dir)
        self.image_paths_B = _get_image_paths_in_dir(images_B_dir)

        assert len(self.image_paths_A) == len(self.image_paths_B), f"{len(self.image_paths_A)=}, {len(self.image_paths_B)=}."
        assert all(a.name == b.name for a, b in
                   zip(self.image_paths_A, self.image_paths_B)), "Image filenames in A and B paths must be the same."

        self.cv2_flag = cv2_read_flag

        assert shared_transform.additional_targets == {}
        shared_transform.add_targets({"image_B": "image"})

        self.shared_transform = shared_transform
        self.unique_transform = unique_transform

    def __len__(self):
        return len(self.image_paths_A)

    @override
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_A = _read_image(self.image_paths_A[idx], self.cv2_flag)
        image_B = _read_image(self.image_paths_B[idx], self.cv2_flag)

        if self.shared_transform:
            aug = self.shared_transform(image=image_A, image_B=image_B)
            image_A, image_B = aug["image"], aug["image_B"]

        image_A = self.unique_transform(image=image_A)["image"]
        image_B = self.unique_transform(image=image_B)["image"]

        return image_A, image_B


class RandomDomainFilenameIDDataset(FilenameIDPairedImagesDataset):
    def __init__(
        self,
        image_dirs: list[pathlib.Path],
        cv2_read_flag: int,
        shared_transform: A.Compose,
        unique_transform: A.Compose,
    ):
        assert len(image_dirs) >= 2, "Need at least two domains to sample from."
        shared = copy.deepcopy(shared_transform)
        unique = unique_transform

        super().__init__(
            images_A_dir=image_dirs[0],
            images_B_dir=image_dirs[1],
            cv2_read_flag=cv2_read_flag,
            shared_transform=shared,
            unique_transform=unique,
        )

        self.paths_per_domain = [
            _get_image_paths_in_dir(d) for d in image_dirs
        ]
        self.num_domains = len(self.paths_per_domain)
        self.num_images = len(self.paths_per_domain[0])

        for paths in self.paths_per_domain[1:]:
            assert len(paths) == self.num_images, "All dirs must have equal size"
            for a, b in zip(self.paths_per_domain[0], paths):
                assert a.name == b.name, "Filenames must match across domains"

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        dom_a, dom_b = random.sample(range(self.num_domains), 2)

        self.image_paths_A = self.paths_per_domain[dom_a]
        self.image_paths_B = self.paths_per_domain[dom_b]

        return super().__getitem__(idx)
