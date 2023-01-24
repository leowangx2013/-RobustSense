import torch
import numpy as np
import torch.nn.functional as F
import torchvision
import os, time, tqdm
from model import loss, cVAE
from pathlib import Path
import yaml

import os
import sys
sys.path.append("../")
sys.path.append("../acid_dataset_utils")
from acid_dataset_utils.data_loader import *
from vae_utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--run", type=str, default="test", help="name of the run")
parser.add_argument("--gen_n", type=int, default=5, help="Generate n samples")
parser.add_argument("--gpu", type=str, default="0", help="Visible GPU")

opt = parser.parse_args()

if not os.path.exists(f"./visualization/{opt.run}_gen"):
    Path(f"./visualization/{opt.run}_gen").mkdir(parents=True, exist_ok=True)

cvae_config = load_yaml("./cVAE_config.yaml")

############## loading data ###################
train_X, train_Y, test_X, test_Y, train_sample_count, test_sample_count, train_labels, test_labels = load_data()
print("train_X.shape: ", train_X.shape)
print("train_Y.shape: ", train_Y.shape)
############## masking data ###################
train_X, train_Y = mask_training_data(train_X, train_Y, cvae_config["masked_vehicle_types"], cvae_config["masked_terrain_types"])
test_X, test_Y = mask_training_data(test_X, test_Y, cvae_config["masked_vehicle_types"], cvae_config["masked_terrain_types"])
print("masked_train_X.shape: ", train_X.shape)
print("masked_train_Y.shape: ", train_Y.shape)
vehicle_type_set = set()
for y in train_Y:
    # if np.argmax(Ys[:9]) in masked_vehicle_types and np.argmax(Ys[10:13]) in masked_terrain_types:
    vehicle_type_set.add(f"{np.argmax(y[:9])} - {np.argmax(y[10:13])}")

# print("vehicle_type_set: ", vehicle_type_set)
############## loading models ###################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def __init__(self, signal_len, label_len, nhid = 16, ncond = 64):
net = cVAE(512, 14, nhid = 8, ncond = 64)
net.to(device)
net.eval()
print(net)
save_name = f"cVAE_{opt.run}.pt"

lr = 0.01
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay = 0.0001)

def adjust_lr(optimizer, decay_rate=0.95):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate

def one_hot_encode(n, N):
    enc = np.zeros(N)
    enc[n] = 1
    return enc.tolist()

if os.path.exists(save_name):
    checkpoint = torch.load(save_name, map_location = device)
    net.load_state_dict(checkpoint["net"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for g in optimizer.param_groups:
        g['lr'] = lr
else:
    print("No checkpoint found.")

for vehicle_type in cvae_config["masked_vehicle_types"]:
    for terrain_type in cvae_config["masked_terrain_types"]:
        for speed_type in cvae_config["speed_types"]:
            for distance_type in cvae_config["distance_types"]:
                for n in range(opt.gen_n):
                    label = one_hot_encode(vehicle_type, 9) + [float(speed_type)] + one_hot_encode(terrain_type, 3) + [float(distance_type)]
                    # label = [float(i) for i in label]
                    # list(vehicle_type) + [float(speed)] + list(terrain_type) + [float(distance)]
                    gen_signal = net.generate(label)
                    gen_signal = gen_signal.cpu().detach().numpy()
                    visualize_single_signal(f"{vehicle_type}_{speed_type}_{terrain_type}_{distance_type}_{n}.png", gen_signal, label, f"./visualization/{opt.run}_gen")
                    