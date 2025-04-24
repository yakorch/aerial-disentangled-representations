from .aerial_datasets import LEVIR_CDPlus_Dataset, SYSU_CD_Dataset, S2LookingDataset, Hi_UCD_Dataset, GVLM_CD_Dataset, BANDONDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


_shared_shared_transform = A.Compose(
    [A.HorizontalFlip(p=0.4), A.VerticalFlip(p=0.2), A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
     A.Perspective(scale=(0.02, 0.05), p=0.3), ])

_noise_transform = A.Compose([A.OneOf(
    [A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20), A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2), ], p=0.8),

    A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=(3, 7)), ], p=0.5), A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
    # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(), ])

_read_color = True

LEVIR_dataset_train = LEVIR_CDPlus_Dataset(split="train", read_color=_read_color, shared_transform=A.Compose(
    [A.RandomResizedCrop(height=256, width=256, scale=(0.4, 0.75), ratio=(0.95, 1 / 0.95), p=1.0), _shared_shared_transform, ]),
                                           unique_transform=_noise_transform)

SYSU_dataset_train = SYSU_CD_Dataset(split="train", read_color=_read_color, shared_transform=A.Compose(
    [A.RandomResizedCrop(height=256, width=256, scale=(0.9, 1), ratio=(0.95, 1 / 0.95), p=1.0), _shared_shared_transform, ]), unique_transform=_noise_transform)

S2Looking_dataset_train = S2LookingDataset(split="train", read_color=_read_color, shared_transform=A.Compose(
    [A.RandomResizedCrop(height=256, width=256, scale=(0.4, 0.75), ratio=(0.95, 1 / 0.95), p=1.0), _shared_shared_transform, ]),
                                           unique_transform=_noise_transform)

Hi_UCD_dataset_train = Hi_UCD_Dataset(split="train", read_color=_read_color, shared_transform=A.Compose(
    [A.RandomResizedCrop(height=256, width=256, scale=(0.75, 1.0), ratio=(0.95, 1 / 0.95), p=1.0), _shared_shared_transform, ]),
                                      unique_transform=_noise_transform)

GVLM_dataset = GVLM_CD_Dataset(read_color=_read_color, shared_transform=A.Compose(
    [A.RandomResizedCrop(height=256, width=256, scale=(0.1, 0.5), ratio=(0.95, 1 / 0.95), p=1.0), _shared_shared_transform, ]), unique_transform=_noise_transform)

BANDON_dataset_train = BANDONDataset(split="train", read_color=_read_color, shared_transform=A.Compose(
    [A.RandomResizedCrop(height=256, width=256, scale=(0.3, 0.9), ratio=(0.95, 1 / 0.95), p=1.0), _shared_shared_transform, ]), unique_transform=_noise_transform)


