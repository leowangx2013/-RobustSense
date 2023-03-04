import os
import torch
import logging
import numpy as np

from torch.utils.data import DataLoader
from input_utils.acids_dataset import ACIDS_Dataset


def create_dataloader(index_file, vehiclel_type_mask, terrain_type_mask, is_train=True, batch_size=64, workers=5):
    # init the dataset
    dataset = ACIDS_Dataset(index_file, vehiclel_type_mask, terrain_type_mask)
    batch_size = min(batch_size, len(dataset))

    # define the dataloader with weighted sampler for training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=workers)

    return dataloader

if __name__ == "__main__":
    dataloader = create_dataloader("/home/sl29/data/ACIDS/random_partition_index_vehicle_classification/train_index.txt", [1], [1])
    data, label = next(iter(dataloader))
    print("label: ", label)
    print("data audio: ", data["shake"]["audio"].shape)

    