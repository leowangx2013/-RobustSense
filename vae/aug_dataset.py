import numpy as np
import os, time, tqdm
from pathlib import Path
import yaml
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--checkpoint", type=str, default="test", help="name of the checkpoint")
parser.add_argument("--run", type=str, default="test", help="name of the run")
parser.add_argument("--gen_n", type=int, default=0, help="Generate n samples")
parser.add_argument("--model", type=str, default="cVAE_2d", help="Model: cVAE_1d or cVAE_2d")
parser.add_argument("--gpu", type=str, default="0", help="Visible GPU")
parser.add_argument("--signal_len", type=int, default=1024, help="length of the time series data")
parser.add_argument("--gen_masked", type=bool, default=False, help="generate masked data or not")
parser.add_argument("--visualize", type=bool, default=False, help="Visualize data or not")

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

import torch
import torch.nn.functional as F
import torchvision
sys.path.append("../")
sys.path.append("../acids_dataset_utils")
sys.path.append("input_utils")
from acids_dataset_utils.data_loader import *
from vae_utils import *
from input_utils.acids_dataloader import *
from input_utils.preprocess import *
from visualize_utils import *
from model_1d import loss_1d, cVAE_1d
from model_2d import loss_2d, cVAE_2d

VEHICLE_TYPES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TERRAIN_TYPES = [0, 1, 2]
SPEED_TYPES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0]
DISTANCE_TYPES = [0.0, 1.0, 2.0, 3.0, 4.0, -1.0]

ORIGINAL_PT_FILE_PATH = f"/home/tianshi/GAN_Vehicle/pt_files/original"
train_generated = True
if not os.path.exists(ORIGINAL_PT_FILE_PATH):
    Path(ORIGINAL_PT_FILE_PATH).mkdir(parents=True, exist_ok=True)
    train_generated = False

OUTPUT_PT_FILE_PATH = f"/home/tianshi/GAN_Vehicle/pt_files/{opt.run}"
if not os.path.exists(OUTPUT_PT_FILE_PATH):
    Path(OUTPUT_PT_FILE_PATH).mkdir(parents=True, exist_ok=True)
else:
    for f in os.listdir(OUTPUT_PT_FILE_PATH):
        os.remove(os.path.join(OUTPUT_PT_FILE_PATH, f))

OUTPUT_PT_INDEX_PATH = f"/home/tianshi/GAN_Vehicle/pt_indexes/{opt.run}"
if not os.path.exists(OUTPUT_PT_INDEX_PATH):
    Path(OUTPUT_PT_INDEX_PATH).mkdir(parents=True, exist_ok=True)
else:
    for f in os.listdir(OUTPUT_PT_INDEX_PATH):
        os.remove(os.path.join(OUTPUT_PT_INDEX_PATH, f))

train_file_paths = []
test_file_paths = []

train_dataloader = create_dataloader("/home/tianshi/data/ACIDS/random_partition_index_vehicle_classification/train_index.txt", is_train=False)
test_dataloader = create_dataloader("/home/tianshi/data/ACIDS/random_partition_index_vehicle_classification/test_index.txt", is_train=False)

train_counter = 0
for n, (Xs, Ys) in enumerate(train_dataloader):
    Xs, Ys = preprocess(Xs, Ys)
    for x, y in zip(Xs, Ys):
        vehicle_type, speed, terrain_type, distance = label_to_attributes(y)
        # if vehicle_type.detach().item() == 0:
        #     print(f"sample {train_counter}, vehicle type is 0, speed: {speed}, terrain_type: {terrain_type}, distance: {distance}")
        save_as_pt(x, y, os.path.join(ORIGINAL_PT_FILE_PATH, f"train_{train_counter}.pt"))
        train_file_paths.append(os.path.join(ORIGINAL_PT_FILE_PATH, f"train_{train_counter}.pt"))
        train_counter += 1

test_counter = 0
for n, (Xs, Ys) in enumerate(test_dataloader):
    Xs, Ys = preprocess(Xs, Ys)
    for x, y in zip(Xs, Ys):
        vehicle_type, speed, terrain_type, distance = label_to_attributes(y)
        save_as_pt(x, y, os.path.join(ORIGINAL_PT_FILE_PATH, f"test_{test_counter}.pt"))
        test_file_paths.append(os.path.join(ORIGINAL_PT_FILE_PATH, f"test_{test_counter}.pt"))
        test_counter += 1


############## loading models ###################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if opt.model == "cVAE_1d":
    net = cVAE_1d(512, 14, nhid = 8, ncond = 64)
elif opt.model == "cVAE_2d":
    net = cVAE_2d((7, 128), 15, nhid = 128, ncond = 32)

net.to(device)
net.eval()
print(net)
save_name = f"cVAE_{opt.checkpoint}.pt"

if os.path.exists(save_name):
    checkpoint = torch.load(save_name, map_location = device)
    net.load_state_dict(checkpoint["net"])
else:
    print("No checkpoint found.")

counter = 0
for vehicle_type in VEHICLE_TYPES:
    for terrain_type in TERRAIN_TYPES:
        for speed_type in SPEED_TYPES:
            for distance_type in DISTANCE_TYPES:
                for n in range(opt.gen_n):
                    label = attributes_to_label(vehicle_type, speed_type, terrain_type, distance_type)

                    gen_signal = net.generate(label)[0].cpu().detach()
                    gen_signal_np = gen_signal.numpy() # Only one item in a batch
                    if opt.visualize:
                        if opt.model == "cVAE_1d":
                            visualize_single_signal(f"{vehicle_type}_{speed_type}_{terrain_type}_{distance_type}_{n}.png", gen_signal_np, label, f"./visualization/{opt.run}_gen")
                        elif opt.model == "cVAE_2d":
                            visualize_single_spect(f"{vehicle_type}_{speed_type}_{terrain_type}_{distance_type}_{n}.png", gen_signal_np, label, f"./visualization/{opt.run}_gen")

                    output_path = os.path.join(OUTPUT_PT_FILE_PATH, f"gen_{counter}.pt")
                    train_file_paths.append(output_path)
                    save_as_pt(gen_signal, torch.from_numpy(np.array(label)), output_path)
                    counter += 1

print("train_file_paths: ", len(train_file_paths))
print("test_file_paths: ", len(test_file_paths))

with open(os.path.join(OUTPUT_PT_INDEX_PATH, "train_index.txt"), "w") as f:
    for file_path in train_file_paths:
        f.write(file_path + "\n")

with open(os.path.join(OUTPUT_PT_INDEX_PATH, "val_index.txt"), "w") as f:
    for file_path in test_file_paths:
        f.write(file_path + "\n")

with open(os.path.join(OUTPUT_PT_INDEX_PATH, "test_index.txt"), "w") as f:
    for file_path in test_file_paths:
        f.write(file_path + "\n")