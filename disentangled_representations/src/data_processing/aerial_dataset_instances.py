import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

from .aerial_datasets import LEVIR_CDPlus_Dataset, SYSU_CD_Dataset, S2LookingDataset, Hi_UCD_Dataset, GVLM_CD_Dataset, BANDONDataset

_read_color = False

_test_resize_transform = A.Compose([
    A.Resize(height=256, width=256, p=1.0),
], additional_targets={})



_shared_shared_transform = A.Compose(
    [A.HorizontalFlip(p=0.4), A.VerticalFlip(p=0.2), A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
     A.Perspective(scale=(0.02, 0.05), p=0.3), ])

_noise_transform = A.Compose([A.ShiftScaleRotate(shift_limit=0.20, scale_limit=0.15, rotate_limit=30, border_mode=0, p=0.9),
                              A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
                              A.Blur(blur_limit=3, p=0.5),
                              A.GaussNoise(
                                  var_limit=(10.0, 50.0),
                                  p=0.3
                              ),
                            A.MotionBlur(blur_limit=5, p=0.3),
                            A.CoarseDropout(max_holes=4, max_height=32, max_width=32, fill_value=0, p=0.4),
                              ])



_final_transform = A.Compose([A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.)), ToTensorV2(), ])

_noise_and_final_transform = A.Compose([_noise_transform, _final_transform])


def make_transforms(crop_scale, use_noise: bool):
    shared = A.Compose([A.RandomResizedCrop(height=256, width=256, scale=crop_scale, ratio=(0.95, 1 / 0.95), p=1.0), _shared_shared_transform, ],
                       additional_targets={})
    unique = _noise_and_final_transform if use_noise else _final_transform
    return shared, unique


DATASETS = {LEVIR_CDPlus_Dataset: [("train", (0.6, 0.9), True), ("test", (1.0, 1.0), False), ],
            SYSU_CD_Dataset: [("train", (0.9, 1.0), True), ("val", (0.9, 1.0), False), ("test", (1.0, 1.0), False), ],
            S2LookingDataset: [("train", (0.4, 0.75), True), ("val", (0.7, 0.95), False), ("test", (1.0, 1.0), False), ],
            Hi_UCD_Dataset: [("train", (0.75, 1.0), True), ("val", (0.9, 1.0), False), ("test", (1.0, 1.0), False), ],
            GVLM_CD_Dataset: [(None, (0.1, 0.4), True), ],
            BANDONDataset: [("train", (0.3, 0.9), True), ("val", (0.65, 0.95), False), ("test", (1.0, 1.0), False), ], }

aerial_datasets_mapping: dict[str, torch.utils.data.Dataset] = {}
for ds_class, specs in DATASETS.items():
    for split, scale, noise_flag in specs:
        if split == "test":
            shared_tf = _test_resize_transform
            unique_tf = _final_transform
        else:
            shared_tf, unique_tf = make_transforms(scale, noise_flag)

        kwargs = {"read_color": _read_color, "shared_transform": shared_tf, "unique_transform": unique_tf}
        if split is not None:
            kwargs["split"] = split

        key = f"{ds_class.__name__}" + (f"_{split}" if split else "")
        ds = ds_class(**kwargs)
        aerial_datasets_mapping[key] = ds

LEVIR_dataset_train, SYSU_dataset_train, S2Looking_dataset_train, Hi_UCD_dataset_train, BANDON_dataset_train = (
    aerial_datasets_mapping[f"{ds_class.__name__}_train"] for ds_class, splits in DATASETS.items() if splits[0][0] == "train")
GVLM_dataset = aerial_datasets_mapping["GVLM_CD_Dataset"]

SYSU_dataset_val, S2Looking_dataset_val, Hi_UCD_dataset_val, BANDON_dataset_val = (aerial_datasets_mapping[f"{ds_class.__name__}_val"] for ds_class, splits in
                                                                                   DATASETS.items() if len(splits) > 1 and splits[1][0] == "val")
