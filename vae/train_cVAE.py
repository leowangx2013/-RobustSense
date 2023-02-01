import torch
import numpy as np
import torch.nn.functional as F
import torchvision
import os, time, tqdm
from model_1d import loss_1d, cVAE_1d
from model_2d import loss_2d, cVAE_2d

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
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--n_classes", type=int, default=9, help="number of classes for dataset")
parser.add_argument("--label_len", type=int, default=14, help="number of classes for dataset")
# parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--signal_len", type=int, default=1024, help="length of the time series data")
parser.add_argument("--channels", type=int, default=2, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--run", type=str, default="test", help="name of the run")
parser.add_argument("--gpu", type=str, default="0", help="Visible GPU")
parser.add_argument("--mode", type=str, default="train", help="Mode: train or test")
parser.add_argument("--model", type=str, default="cVAE_2d", help="Model: cVAE_1d or cVAE_2d")
parser.add_argument("--beta", type=float, default=1, help="weight for KL divergence")

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
cuda = True if torch.cuda.is_available() else False

if not os.path.exists(f"./visualization/{opt.run}"):
    Path(f"./visualization/{opt.run}").mkdir(parents=True, exist_ok=True)

cvae_config = load_yaml("./cVAE_config.yaml")

############## loading data ###################
if opt.model == "cVAE_1d":
    train_X, train_Y, test_X, test_Y, train_sample_count, test_sample_count, train_labels, test_labels = load_data(mode="fft")
elif opt.model == "cVAE_2d":
    train_X, train_Y, test_X, test_Y, train_sample_count, test_sample_count, train_labels, test_labels = load_data(mode="stft", sample_len=opt.signal_len)
# train_X, train_Y, test_X, test_Y, train_sample_count, test_sample_count, train_labels, test_labels = load_data(mode="stft")
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

############## loading models ###################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if opt.model == "cVAE_1d":
    net = cVAE_1d(1024, 14, nhid = 8, ncond = 64)
elif opt.model == "cVAE_2d":
    _, _, Zxx = signal.stft(np.random.rand(opt.signal_len), nperseg=128, noverlap=64)
    net = cVAE_2d(Zxx.shape, 14, nhid = 8, ncond = 64)

net.to(device)
print(net)
save_name = f"cVAE_{opt.run}.pt"

lr = 0.01
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay = 0.0001)

def adjust_lr(optimizer, decay_rate=0.95):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate

if opt.mode == "train":
    retrain = False
    # retrain = True
    # if os.path.exists(save_name):
        # print("Model parameters have already been trained before. Retrain ? (y/n)")
        # ans = input()
        # if not (ans == 'y'):
        #     checkpoint = torch.load(save_name, map_location = device)
        #     net.load_state_dict(checkpoint["net"])
        #     optimizer.load_state_dict(checkpoint["optimizer"])
        #     for g in optimizer.param_groups:
        #         g['lr'] = lr

############### training #########################
max_epochs = 1000
early_stop = EarlyStop(patience = 20, save_name = save_name)
net = net.to(device)

print("training on ", device)
for epoch in range(max_epochs):

    batch_samples = []
    batch_labels = []

    # Randomly shuffle the data
    temp = list(zip(train_X, train_Y))
    np.random.shuffle(temp)
    train_X, train_Y = zip(*temp)
    train_X, train_Y = list(train_X), list(train_Y)

    train_loss, n, start = 0.0, 0, time.time()
    
    accumulate_rec_loss = 0.0
    accumulate_kl_loss = 0.0

    # for X, y in tqdm.tqdm(train_iter, ncols = 50):
    for n, (X, y) in enumerate(zip(train_X, train_Y)):
        # for X, y in zip(train_X, train_Y):
        batch_samples.append(X)
        batch_labels.append(y)
        if len(batch_samples) < opt.batch_size:
            continue

        batch_samples = torch.Tensor(batch_samples).to(device)
        batch_labels = torch.Tensor(batch_labels).to(device)
        X_hat, mean, logvar = net(batch_samples, batch_labels)

        if epoch > 0 and epoch % 10 == 0:
        # if epoch % 1 == 0:

            # Plot the first sample for each batch
            if opt.model == "cVAE_1d":
                visualize_reconstruct_signals(n-opt.batch_size+1, np.expand_dims(batch_samples.detach().cpu().numpy()[0], 0), 
                    batch_labels.detach().cpu().numpy(), np.expand_dims(X_hat.detach().cpu().numpy()[0], 0), f"./visualization/{opt.run}",
                    skip_n=1)
            elif opt.model == "cVAE_2d":
                visualize_reconstruct_spect(n-opt.batch_size+1, np.expand_dims(batch_samples.detach().cpu().numpy()[0], 0), 
                    batch_labels.detach().cpu().numpy(), np.expand_dims(X_hat.detach().cpu().numpy()[0], 0), f"./visualization/{opt.run}",
                    skip_n=1)                

        # print("batch_samples: ", batch_samples.shape)
        # print("X_hat: ", X_hat.shape)
        # exit()
        if opt.model == "cVAE_1d":
            reconstruction_loss, KL_divergence = loss_1d(batch_samples, X_hat, mean, logvar)
        elif opt.model == "cVAE_2d":
            reconstruction_loss, KL_divergence = loss_2d(batch_samples, X_hat, mean, logvar)

        reconstruction_loss, KL_divergence = loss_1d(batch_samples, X_hat, mean, logvar)
        accumulate_rec_loss += reconstruction_loss.cpu().item()
        accumulate_kl_loss += KL_divergence.cpu().item()
        l = (reconstruction_loss + opt.beta * KL_divergence).to(device)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_loss += l.cpu().item()
        n += X.shape[0]
        
        batch_samples = []
        batch_labels = []


    train_loss /= n
    print('epoch %d, train loss %.4f , rec loss %.4f, kl loss %.4f, weighted kl loss %.4f, time %.1f sec'
          % (epoch, train_loss, accumulate_rec_loss/n, accumulate_kl_loss/n,
          accumulate_kl_loss/n * opt.beta, time.time() - start))
    
    adjust_lr(optimizer)
    
    if (early_stop(train_loss, net, optimizer)):
        break

checkpoint = torch.load(early_stop.save_name)
net.load_state_dict(checkpoint["net"])