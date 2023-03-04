import os
import torch
import numpy as np
from torch.utils.data import Dataset
from random import shuffle
import os


class ACIDS_Dataset(Dataset):

    def load_meta(self):
        meta_info = {}
        with open("/home/tianshi/GAN_Vehicle/acids_dataset_utils/ACIDS_meta.csv", "r") as file:
            lines = file.readlines()
            for line in lines:
                filename, vehicle_id, speed, terrain_id, distance = line.split(",")
                vehicle_id = int(vehicle_id.split(" ")[1])
                meta_info[os.path.basename(filename).split(".")[0]] = {"vehicle_id": vehicle_id, "terrain_id": terrain_id}
        return meta_info 

    def __init__(self, index_file, vehicle_type_mask, terrain_type_mask):
        sample_files = list(np.loadtxt(index_file, dtype=str))
        meta_info = self.load_meta()
        filtered_sample_files = [] 
        terrain_name_to_id = {"Desert": 0, "Arctic": 1, "Normal": 2}
        for fn in sample_files:
            vehicle_id = meta_info[os.path.basename(fn).split("_")[0]]["vehicle_id"]
            terrain_name = meta_info[os.path.basename(fn).split("_")[0]]["terrain_id"]
            terrain_id = terrain_name_to_id[terrain_name]

            if vehicle_id not in vehicle_type_mask or terrain_id not in terrain_type_mask:
                filtered_sample_files.append(fn)
        self.sample_files = filtered_sample_files

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        sample = torch.load(self.sample_files[idx])
        data = sample["data"]
        label = sample["label"]
        
        return data, label
