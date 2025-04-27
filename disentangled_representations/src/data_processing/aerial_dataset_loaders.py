import torch
from loguru import logger
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from .aerial_dataset_instances import (BANDON_dataset_train, BANDON_dataset_val, GVLM_dataset, Hi_UCD_dataset_train, Hi_UCD_dataset_val, LEVIR_dataset_train,
                                       S2Looking_dataset_train, S2Looking_dataset_val, SYSU_dataset_train, SYSU_dataset_val, )


def create_train_data_loader_for_image_pairs(batch_size: int, num_workers: int):
    # dataset_weights_mapping_train: dict[torch.utils.data.Dataset, float] = {LEVIR_dataset_train: 1.5, SYSU_dataset_train: 0.1, S2Looking_dataset_train: 0.25,
    #                                                                         Hi_UCD_dataset_train: 0.5, GVLM_dataset: 25.0, BANDON_dataset_train: 4.0, }

    dataset_weights_mapping_train: dict[torch.utils.data.Dataset, float] = {BANDON_dataset_train: 1.5, GVLM_dataset: 6.0, Hi_UCD_dataset_train: 0.75, LEVIR_dataset_train: 0.4}

    train_datasets = list(dataset_weights_mapping_train.keys())

    logger.info(f"{({ds.__class__.__name__: len(ds) for ds in train_datasets})=}")

    weights = []
    for ds in train_datasets:
        w = dataset_weights_mapping_train[ds]
        weights += [w] * len(ds)
    weights = torch.DoubleTensor(weights)

    num_samples = int(weights.sum().item())
    logger.info(f"{num_samples=} per epoch.")

    sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)

    complete_train_ds = ConcatDataset(train_datasets)
    train_data_loader = torch.utils.data.DataLoader(complete_train_ds, batch_size=batch_size, num_workers=num_workers, sampler=sampler, prefetch_factor=3,
                                                    persistent_workers=True, pin_memory=True)

    return train_data_loader


def create_val_data_loader_for_image_pairs(batch_size: int, num_workers: int):
    # dataset_weights_mapping_val: dict[torch.utils.data.Dataset, float] = {SYSU_dataset_val: 0.02, S2Looking_dataset_val: 0.1, Hi_UCD_dataset_val: 0.15,
    #                                                                       BANDON_dataset_val: 0.5}

    dataset_weights_mapping_val: dict[torch.utils.data.Dataset, float] = {BANDON_dataset_val: 0.5, Hi_UCD_dataset_val: 0.2}

    val_datasets = list(dataset_weights_mapping_val.keys())

    logger.info(f"{({ds.__class__.__name__: len(ds) for ds in val_datasets})=}")

    weights = []
    for ds in val_datasets:
        w = dataset_weights_mapping_val[ds]
        weights += [w] * len(ds)
    weights = torch.DoubleTensor(weights)

    num_samples = int(weights.sum().item())
    logger.info(f"{num_samples=} per epoch.")

    sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=False)

    complete_train_ds = ConcatDataset(val_datasets)
    val_data_loader = torch.utils.data.DataLoader(complete_train_ds, batch_size=batch_size, num_workers=num_workers, sampler=sampler, prefetch_factor=1,
                                                    persistent_workers=True, shuffle=False)

    return val_data_loader
