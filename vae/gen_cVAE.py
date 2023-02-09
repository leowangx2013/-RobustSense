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
parser.add_argument("--run", type=str, default="test", help="name of the run")
parser.add_argument("--gen_n", type=int, default=100, help="Generate n samples")
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
sys.path.append("../acid_dataset_utils")
from acid_dataset_utils.data_loader import *
from vae_utils import *
from visualize_utils import *
from model_1d import loss_1d, cVAE_1d
from model_2d import loss_2d, cVAE_2d

if not os.path.exists(f"./visualization/{opt.run}_gen"):
    Path(f"./visualization/{opt.run}_gen").mkdir(parents=True, exist_ok=True)
else:
    for f in os.listdir(f"./visualization/{opt.run}_gen"):
        os.remove(os.path.join(f"./visualization/{opt.run}_gen", f))

if not os.path.exists(f"./visualization/{opt.run}_masked"):
    Path(f"./visualization/{opt.run}_masked").mkdir(parents=True, exist_ok=True)
else:
    for f in os.listdir(f"./visualization/{opt.run}_masked"):
        os.remove(os.path.join(f"./visualization/{opt.run}_masked", f))

cvae_config = load_yaml("./cVAE_config.yaml")

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

############## loading data ###################
if opt.model == "cVAE_1d":
    train_X, train_Y, test_X, test_Y, train_sample_count, test_sample_count, train_labels, test_labels = load_data(mode="fft")
elif opt.model == "cVAE_2d":
    train_X, train_Y, test_X, test_Y, train_sample_count, test_sample_count, train_labels, test_labels = load_data(mode="stft", sample_len=opt.signal_len)

# Xs = np.concatenate([train_X, test_X], axis=0)
# Ys = np.concatenate([train_Y, test_Y], axis=0)
# Xs = train_X
# Ys = train_Y
# print("Ys.shape: ", Ys.shape)


# Masked training data for cVAE. Will be used as the testing set in the classifier.
masked_train_X, masked_train_Y = get_masked_data(train_X, train_Y, cvae_config["masked_vehicle_types"], cvae_config["masked_terrain_types"])
# Training data for cVAE. Add up the generated data, together it becomes the training set for the classifier.
train_X, train_Y = mask_training_data(train_X, train_Y, cvae_config["masked_vehicle_types"], cvae_config["masked_terrain_types"])

# print("masked_train_X: ", len(masked_train_X))
# print("train_x: ", len(train_X))

# print("masked_train_X.shape: ", train_X.shape)
# print("masked_train_Y.shape: ", train_Y.shape)

if opt.gen_masked:
    for n, (x, y) in enumerate(zip(masked_train_X, masked_train_Y)):
        vehicle_type = np.argmax(y[:9])
        speed_type = y[9]
        terrain_type = np.argmax(y[10:13])
        distance_type = y[13]
        visualize_single_spect(f"{vehicle_type}_{int(speed_type)}_{terrain_type}_{int(distance_type)}_{n}.png", x, y, f"./visualization/{opt.run}_masked")

############## loading models ###################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if opt.model == "cVAE_1d":

    net = cVAE_1d(512, 14, nhid = 8, ncond = 64)
elif opt.model == "cVAE_2d":
    _, _, Zxx = signal.stft(np.random.rand(opt.signal_len), nperseg=128, noverlap=64)
    net = cVAE_2d(Zxx.shape, 14, nhid = 128, ncond = 16)

net.to(device)
net.eval()
print(net)
save_name = f"cVAE_{opt.run}.pt"

# def one_hot_encode(n, N):
#     enc = np.zeros(N)
#     enc[n] = 1
#     return enc.tolist()

if os.path.exists(save_name):
    checkpoint = torch.load(save_name, map_location = device)
    net.load_state_dict(checkpoint["net"])
else:
    print("No checkpoint found.")

train_file_paths = []
counter = 0
for vehicle_type in cvae_config["masked_vehicle_types"]:
    for terrain_type in cvae_config["masked_terrain_types"]:
        for speed_type in cvae_config["speed_types"]:
            for distance_type in cvae_config["distance_types"]:
                for n in range(opt.gen_n):
                    label = attributes_to_label(vehicle_type, speed_type, terrain_type, distance_type)
                    gen_signal = net.generate(label)
                    gen_signal = gen_signal.cpu().detach().numpy()[0] # Only one item in a batch
                    if opt.visualize:
                        if opt.model == "cVAE_1d":
                            visualize_single_signal(f"{vehicle_type}_{speed_type}_{terrain_type}_{distance_type}_{n}.png", gen_signal, label, f"./visualization/{opt.run}_gen")
                        elif opt.model == "cVAE_2d":
                            visualize_single_spect(f"{vehicle_type}_{speed_type}_{terrain_type}_{distance_type}_{n}.png", gen_signal, label, f"./visualization/{opt.run}_gen")

                    output_path = os.path.join(OUTPUT_PT_FILE_PATH, f"gen_{counter}.pt")
                    train_file_paths.append(output_path)
                    save_as_pt(gen_signal, label, output_path)
                    counter += 1

for n, (x, y) in enumerate(zip(train_X, train_Y)):
    output_path = os.path.join(OUTPUT_PT_FILE_PATH, f"train_{n}.pt")
    save_as_pt(x, y, output_path)
    train_file_paths.append(output_path)

test_file_paths = []
for n, (x, y) in enumerate(zip(masked_train_X, masked_train_Y)):
    output_path = os.path.join(OUTPUT_PT_FILE_PATH, f"test_{n}.pt")
    save_as_pt(x, y, output_path)
    test_file_paths.append(output_path)


with open(os.path.join(OUTPUT_PT_INDEX_PATH, "train_index.txt"), "w") as f:
    for file_path in train_file_paths:
        f.write(file_path + "\n")

with open(os.path.join(OUTPUT_PT_INDEX_PATH, "val_index.txt"), "w") as f:
    for file_path in test_file_paths:
        f.write(file_path + "\n")

with open(os.path.join(OUTPUT_PT_INDEX_PATH, "test_index.txt"), "w") as f:
    for file_path in test_file_paths:
        f.write(file_path + "\n")