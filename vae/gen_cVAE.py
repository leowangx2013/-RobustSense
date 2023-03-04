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
parser.add_argument("--output_folder", type=str, default="", help="name of the run")
parser.add_argument("--gen_n", type=int, default=100, help="Generate n samples")
parser.add_argument("--model", type=str, default="cVAE_2d", help="Model: cVAE_1d or cVAE_2d")
parser.add_argument("--gpu", type=str, default="0", help="Visible GPU")
parser.add_argument("--signal_len", type=int, default=1024, help="length of the time series data")
parser.add_argument("--gen_masked", type=bool, default=False, help="generate masked data or not")
parser.add_argument("--visualize", type=bool, default=False, help="Visualize data or not")
parser.add_argument("--cvae_config_file", type=str, default="./cVAE_config.yaml", help="config file for cVAE")

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

if not os.path.exists(f"./visualization/{opt.run}_gen"):
    Path(f"./visualization/{opt.run}_gen").mkdir(parents=True, exist_ok=True)
else:
    for f in os.listdir(f"./visualization/{opt.run}_gen"):
        os.remove(os.path.join(f"./visualization/{opt.run}_gen", f))

if not os.path.exists(f"./visualization/{opt.run}_filtered"):
    Path(f"./visualization/{opt.run}_filtered").mkdir(parents=True, exist_ok=True)
else:
    for f in os.listdir(f"./visualization/{opt.run}_filtered"):
        os.remove(os.path.join(f"./visualization/{opt.run}_filtered", f))

cvae_config = load_yaml(opt.cvae_config_file)

ORIGINAL_PT_FILE_PATH = "/home/tianshi/GAN_Vehicle/pt_files/original" # the original dataset pt files without any filtering
ORIGINAL_PT_INDEX_PATH = "/home/tianshi/GAN_Vehicle/pt_indexes/original" # the original dataset index files without any filtering

if len(opt.output_folder) > 0:
    TRAIN_PT_FILE_PATH = f"/home/tianshi/GAN_Vehicle/pt_files/{opt.output_folder}/{(opt.run).split('_')[0]}"
else:
    TRAIN_PT_FILE_PATH = f"/home/tianshi/GAN_Vehicle/pt_files/{(opt.run).split('_')[0]}"

if len(opt.output_folder) > 0:
    OUTPUT_PT_FILE_PATH = f"/home/tianshi/GAN_Vehicle/pt_files/{opt.output_folder}/{opt.run}"
else:
    OUTPUT_PT_FILE_PATH = f"/home/tianshi/GAN_Vehicle/pt_files/{opt.run}"

if len(opt.output_folder) > 0:
    OUTPUT_PT_INDEX_PATH = f"/home/tianshi/GAN_Vehicle/pt_indexes/{opt.output_folder}/{opt.run}"
else:
    OUTPUT_PT_FILE_PATH = f"/home/tianshi/GAN_Vehicle/pt_files/{opt.run}"

if not os.path.exists(OUTPUT_PT_FILE_PATH):
    Path(OUTPUT_PT_FILE_PATH).mkdir(parents=True, exist_ok=True)
else:
    for f in os.listdir(OUTPUT_PT_FILE_PATH):
        os.remove(os.path.join(OUTPUT_PT_FILE_PATH, f))

if len(opt.output_folder) > 0:
    OUTPUT_PT_INDEX_PATH = f"/home/tianshi/GAN_Vehicle/pt_indexes/{opt.output_folder}/{opt.run}"
else:
    OUTPUT_PT_INDEX_PATH = f"/home/tianshi/GAN_Vehicle/pt_indexes/{opt.run}"
if not os.path.exists(OUTPUT_PT_INDEX_PATH):
    Path(OUTPUT_PT_INDEX_PATH).mkdir(parents=True, exist_ok=True)
else:
    for f in os.listdir(OUTPUT_PT_INDEX_PATH):
        os.remove(os.path.join(OUTPUT_PT_INDEX_PATH, f))

train_file_paths = []
test_file_paths = []

with open(os.path.join(ORIGINAL_PT_INDEX_PATH, "train_index.txt"), "r") as f:
    for line in f:
        file_path = line[:-1]
        label = torch.load(file_path)["label"]
        if label["vehicle_type"].detach().item() not in cvae_config["masked_vehicle_types"] or label["terrain"].detach().item() not in cvae_config["masked_terrain_types"]:
            train_file_paths.append(file_path)
        else:
            test_file_paths.append(file_path)

############## loading models ###################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if opt.model == "cVAE_1d":
    net = cVAE_1d(512, 14, nhid = 8, ncond = 64)
elif opt.model == "cVAE_2d":
    net = cVAE_2d((7, 128), 15, nhid = 128, ncond = 32)

net.to(device)
net.eval()
# print(net)
save_name = f"./weights/{opt.checkpoint}"
print("save_name: ", save_name)
if os.path.exists(save_name):
    checkpoint = torch.load(save_name, map_location = device)
    net.load_state_dict(checkpoint["net"])
else:
    print("No checkpoint found.")

counter = 0
for vehicle_type in cvae_config["masked_vehicle_types"]:
    for terrain_type in cvae_config["masked_terrain_types"]:
        for speed_type in cvae_config["speed_types"]:
            for distance_type in cvae_config["distance_types"]:
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

with open(os.path.join(OUTPUT_PT_INDEX_PATH, "train_index_aug.txt"), "w") as f:
    for file_path in train_file_paths:
        f.write(file_path + "\n")

with open(os.path.join(OUTPUT_PT_INDEX_PATH, "train_index_no-aug.txt"), "w") as f:
    for file_path in train_file_paths:
        if "gen" not in os.path.basename(file_path):
            f.write(file_path + "\n")


with open(os.path.join(OUTPUT_PT_INDEX_PATH, "val_index.txt"), "w") as f:
    for file_path in test_file_paths:
        f.write(file_path + "\n")

with open(os.path.join(OUTPUT_PT_INDEX_PATH, "test_index.txt"), "w") as f:
    for file_path in test_file_paths:
        f.write(file_path + "\n")