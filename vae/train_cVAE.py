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
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--label_len", type=int, default=14, help="number of classes for dataset")
parser.add_argument("--signal_len", type=int, default=1024, help="length of the time series data")
parser.add_argument("--channels", type=int, default=2, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--run", type=str, default="test", help="name of the run")
parser.add_argument("--cvae_config_file", type=str, default="./cVAE_config.yaml", help="config file for cVAE")
parser.add_argument("--weight_output_path", type=str, default="./weights", help="weight file output path")
parser.add_argument("--gpu", type=str, default="0", help="Visible GPU")
parser.add_argument("--mode", type=str, default="train", help="Mode: train or test")
parser.add_argument("--model", type=str, default="cVAE_2d", help="Model: cVAE_1d or cVAE_2d")
parser.add_argument("--beta", type=float, default=1, help="weight for KL divergence")
parser.add_argument("--visualize", type=bool, default=False, help="whether to visualize the results")

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

import torch
import torch.nn.functional as F
import torchvision
from model_1d import loss_1d, cVAE_1d
from model_2d import loss_2d, cVAE_2d
sys.path.append("../")
sys.path.append("../acids_dataset_utils")
sys.path.append("input_utils")
from input_utils.acids_dataloader import *
from input_utils.preprocess import *
from vae_utils import *
from visualize_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(f"./visualization/{opt.run}"):
    Path(f"./visualization/{opt.run}").mkdir(parents=True, exist_ok=True)

cvae_config = load_yaml(opt.cvae_config_file)
# print("opt.cvae_config_file: ", opt.cvae_config_file)

train_dataloader = create_dataloader("/home/tianshi/data/ACIDS/random_partition_index_vehicle_classification/train_index.txt", cvae_config["masked_vehicle_types"], cvae_config["masked_terrain_types"])
test_dataloader = create_dataloader("/home/tianshi/data/ACIDS/random_partition_index_vehicle_classification/test_index.txt", cvae_config["masked_vehicle_types"], cvae_config["masked_terrain_types"])
# print("created dataloader")
############## loading models ###################

if opt.model == "cVAE_1d":
    net = cVAE_1d(1024, 14, nhid = 128, ncond = 32)
elif opt.model == "cVAE_2d":
    net = cVAE_2d((7, 128), 15, nhid = 128, ncond = 32)
# print("create model")

net.to(device)
# print("model to device")
# print(net)
save_name = os.path.join(opt.weight_output_path, f"cVAE_{opt.run}.pt")

lr = 0.1
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay = 0.0001)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay = 0)

def adjust_lr(optimizer, decay_rate=0.95):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate

if opt.mode == "train":
    retrain = False

############### training #########################
max_epochs = opt.n_epochs
early_stop = EarlyStop(patience = 20, save_name = save_name)
# net = net.to(device)

print("training on ", device)
for epoch in range(max_epochs):

    Xs_means = []
    Xs_hat_means = []
    means_means = []
    logvar_means = []

    accumulate_train_loss  = 0
    accumulate_rec_loss = 0 
    accumulate_kl_loss = 0

    start_time = time.time()

    for n, (Xs, Ys) in enumerate(train_dataloader):
        start_time = time.time()
        Xs, Ys = preprocess(Xs, Ys, device)
        start_time = time.time()

        # print("Ys len: ", len(Ys))
        # start_time = time.time()
        # Xs, Ys = filter_train_data(Xs, Ys, cvae_config["masked_vehicle_types"], cvae_config["masked_terrain_types"])
        # print("filter time: ", time.time() - start_time)
        # print("filtered Ys len: ", len(Ys))

        # start_time = time.time()
        # Xs = Xs.to(device)
        # Ys = Ys.to(device)

        Xs_hat, mean, logvar = net(Xs, Ys)
        # print("forward time: ", time.time() - start_time)
        # exit()
        Xs_means.append(torch.mean(Xs, axis=(0,2,3)).detach().cpu().numpy())
        Xs_hat_means.append(torch.mean(Xs_hat, axis=(0,2,3)).detach().cpu().numpy())
        means_means.append(torch.mean(mean, axis=0).detach().cpu().numpy())
        logvar_means.append(torch.mean(logvar, axis=0).detach().cpu().numpy())

        if epoch > 0 and epoch % 20 == 0:
            if n % 10 == 0 and opt.visualize:
                if opt.model == "cVAE_1d":
                    visualize_reconstruct_signals(n-opt.batch_size+1, np.expand_dims(Xs.detach().cpu().numpy()[0], 0), 
                        Ys.detach().cpu().numpy(), np.expand_dims(Xs_hat.detach().cpu().numpy()[0], 0), f"./visualization/{opt.run}")
                elif opt.model == "cVAE_2d":
                    visualize_reconstruct_spect(n, np.expand_dims(Xs.detach().cpu().numpy()[0], 0), 
                        Ys.detach().cpu().numpy(), np.expand_dims(Xs_hat.detach().cpu().numpy()[0], 0), f"./visualization/{opt.run}")

        if opt.model == "cVAE_1d":
            reconstruction_loss, KL_divergence = loss_1d(Xs, Xs_hat, mean, logvar)
        elif opt.model == "cVAE_2d":
            reconstruction_loss, KL_divergence = loss_2d(Xs, Xs_hat, mean, logvar)

        accumulate_rec_loss += reconstruction_loss.cpu().item()
        accumulate_kl_loss += KL_divergence.cpu().item()
        l = (reconstruction_loss + opt.beta * KL_divergence).to(device)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        accumulate_train_loss += l.cpu().item()
        
        batch_samples = []
        batch_labels = []

    print('epoch %d, mean %.4f, logvar %.4f, train loss %.4f , rec loss %.4f, kl loss %.4f, weighted kl loss %.4f, time %.1f sec'
          % (epoch, np.mean(means_means), np.mean(logvar_means), accumulate_train_loss/n/opt.batch_size, accumulate_rec_loss/n/opt.batch_size, accumulate_kl_loss/n/opt.batch_size,
          accumulate_kl_loss/n/opt.batch_size * opt.beta, time.time() - start_time))
    
    # print("Xs_means: ", np.mean(Xs_means, axis = 0), ", Xs_hat_means: ", np.mean(Xs_hat_means, axis = 0))

    # adjust_lr(optimizer)
    
    if (early_stop(accumulate_train_loss, net, optimizer)):
        break

checkpoint = torch.load(early_stop.save_name)
net.load_state_dict(checkpoint["net"])