from typing import Literal

import albumentations as A
import cv2

from disentangled_representations import ORIGINAL_DATASETS_DIR, PREPROCESSED_DATASETS_DIR
from .abstract_aerial_datasets import SimpleSingleImagesDataset, FilenameIDPairedImagesDataset, RandomDomainFilenameIDDataset


class VPairDistractorsDataset(SimpleSingleImagesDataset):
    def __init__(self, read_color: bool, transform: A.Compose):
        distractor_images_dir = ORIGINAL_DATASETS_DIR / "vpair" / "distractors"
        super().__init__(images_dir=distractor_images_dir,
                         cv2_read_flag=cv2.IMREAD_COLOR if read_color else cv2.IMREAD_GRAYSCALE, transform=transform)


class LEVIR_CDPlus_Dataset(FilenameIDPairedImagesDataset):
    def __init__(self, split: Literal["train", "test"], read_color: bool, shared_transform: A.Compose, unique_transform: A.Compose):
        split_path = ORIGINAL_DATASETS_DIR / "LEVIR-CD+" / split
        A_path = split_path / "A"
        B_path = split_path / "B"

        super().__init__(images_A_dir=A_path, images_B_dir=B_path,
                         cv2_read_flag=cv2.IMREAD_COLOR if read_color else cv2.IMREAD_GRAYSCALE, shared_transform=shared_transform, unique_transform=unique_transform)


class SYSU_CD_Dataset(FilenameIDPairedImagesDataset):
    def __init__(self, split: Literal["train", "val", "test"], read_color: bool, shared_transform: A.Compose, unique_transform: A.Compose):
        split_path = ORIGINAL_DATASETS_DIR / "SYSU-CD" / split
        A_path = split_path / "time1"
        B_path = split_path / "time2"

        super().__init__(images_A_dir=A_path, images_B_dir=B_path,
                         cv2_read_flag=cv2.IMREAD_COLOR if read_color else cv2.IMREAD_GRAYSCALE, shared_transform=shared_transform, unique_transform=unique_transform)


class S2LookingDataset(FilenameIDPairedImagesDataset):
    def __init__(self, split: Literal["train", "val", "test"], read_color: bool, shared_transform: A.Compose, unique_transform: A.Compose):
        split_path = ORIGINAL_DATASETS_DIR / "S2Looking" / split
        A_path = split_path / "Image1"
        B_path = split_path / "Image2"

        super().__init__(images_A_dir=A_path, images_B_dir=B_path,
                         cv2_read_flag=cv2.IMREAD_COLOR if read_color else cv2.IMREAD_GRAYSCALE, shared_transform=shared_transform, unique_transform=unique_transform)


class Hi_UCD_Dataset(FilenameIDPairedImagesDataset):
    def __init__(self, split: Literal["train", "val", "test"], read_color: bool, shared_transform: A.Compose, unique_transform: A.Compose):
        split_path = ORIGINAL_DATASETS_DIR / "Hi-UCD" / split / "image"
        A_path = split_path / "2018"
        B_path = split_path / "2019"

        super().__init__(images_A_dir=A_path, images_B_dir=B_path,
                         cv2_read_flag=cv2.IMREAD_COLOR if read_color else cv2.IMREAD_GRAYSCALE, shared_transform=shared_transform, unique_transform=unique_transform)


class GVLM_CD_Dataset(FilenameIDPairedImagesDataset):
    def __init__(self, read_color: bool, shared_transform: A.Compose, unique_transform: A.Compose):
        split_path = PREPROCESSED_DATASETS_DIR / "GVLM_CD"
        A_path = split_path / "A"
        B_path = split_path / "B"

        super().__init__(images_A_dir=A_path, images_B_dir=B_path,
                         cv2_read_flag=cv2.IMREAD_COLOR if read_color else cv2.IMREAD_GRAYSCALE, shared_transform=shared_transform, unique_transform=unique_transform)



class BANDONDataset(RandomDomainFilenameIDDataset):
    def __init__(self, split: Literal["train", "val", "test"], read_color: bool, shared_transform: A.Compose, unique_transform: A.Compose):
        # NOTE: split `test_ood` is not supported because it has a large image imbalance.
        split_path = PREPROCESSED_DATASETS_DIR / "BANDON" / split

        A_path = split_path / "t1"
        B_path = split_path / "t2"
        C_path = split_path / "t3"

        super().__init__(image_dirs=[A_path, B_path, C_path], cv2_read_flag=cv2.IMREAD_COLOR if read_color else cv2.IMREAD_GRAYSCALE, shared_transform=shared_transform, unique_transform=unique_transform)
