import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from pathlib import Path

import sys
sys.path.append("../")
sys.path.append("../acid_dataset_utils")

from acid_dataset_utils.data_loader import *
from gan_utils import visualize_signals

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--n_classes", type=int, default=9, help="number of classes for dataset")
parser.add_argument("--label_len", type=int, default=14, help="number of classes for dataset")
# parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--signal_len", type=int, default=512, help="length of the time series data")
parser.add_argument("--channels", type=int, default=2, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--run", type=str, default="test", help="name of the run")
parser.add_argument("--gpu", type=str, default="0", help="Visible GPU")
parser.add_argument("--mode", type=str, default="train", help="Mode: train or test")

opt = parser.parse_args()
print(opt)

# img_shape = (opt.channels, opt.img_size, opt.img_size)
time_series_shape = (opt.channels, opt.signal_len)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
cuda = True if torch.cuda.is_available() else False

if not os.path.exists(f"./weights/{opt.run}"):
    Path(f"./weights/{opt.run}").mkdir(parents=True, exist_ok=True)

if not os.path.exists(f"./visualization/{opt.run}"):
    Path(f"./visualization/{opt.run}").mkdir(parents=True, exist_ok=True)

if not os.path.exists(f"./visualization/{opt.run}/test"):
    Path(f"./visualization/{opt.run}/test").mkdir(parents=True, exist_ok=True)
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def linear_block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            # layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Sigmoid())
            # layers.append(nn.ReLU())
            return layers
        
        def conv_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, (3,1), stride=1, padding="same")]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.Sigmoid())
            return layers

        def rnn_block(in_feat, out_feat, num_layers=3, normalize=True, dropout_ratio=0.8):
            layers = [nn.GRU(in_feat, out_feat, num_layers, bias=True, batch_first=True, dropout=dropout_ratio, bidirectional=True)]
            if normalize:
                layer.append(nn.BachNorm1d(out_feat))
            layers.append(nn.Sigmoid())

        self.meta_mapping = nn.Sequential(
            *linear_block(opt.label_len, 128, normalize=False),
        )

        self.model = nn.Sequential(
            *linear_block(opt.signal_len + 128, 256, normalize=False),
            *linear_block(256, 512),
            *linear_block(512, 1024),
            nn.Linear(1024, int(np.prod(time_series_shape))),
            # nn.ReLU()
            nn.Sigmoid(),
        )


    def forward(self, noise, labels):
        # print("noise.shape: ", noise.shape)
        # print("self.label_emb(labels).shape: ", self.label_emb(labels).shape)
        # Concatenate label embedding and image to produce input
        # print("labels: ", labels.shape)
        mapped_labels = self.meta_mapping(labels)
        # print("mapped_labels: ", mapped_labels.shape)
        gen_input = torch.cat((mapped_labels, noise), -1)
        # print("gen_input: ", gen_input.shape)
        sample = self.model(gen_input)
        # print("samples.shape: ", sample.shape)
        sample = sample.view(sample.size(0), *time_series_shape)
        return sample


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.label_len + int(np.prod(time_series_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, opt.label_len),
        )

    def forward(self, sample, labels):
        # Concatenate label embedding and image to produce input
        # print("labels.shape: ", labels.shape)
        # print("sample.view(sample.size(0), -1): ", sample.view(sample.size(0), -1).shape)
        # print("labels: ", labels.shape)
        d_in = torch.cat((sample.view(sample.size(0), -1), labels), -1)
        validity = self.model(d_in)
        return validity

# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

train_X, train_Y, test_X, test_Y, train_sample_count, test_sample_count, train_labels, test_labels = load_data()

print("train_X.shape", train_X.shape)
print("train_Y.shape", train_Y.shape)

print("test_X.shape", test_X.shape)
print("test_Y.shape", test_Y.shape)
 
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

if opt.mode == "test":
    generator.load_state_dict(torch.load(f"./weights/{opt.run}/generator_latest.pt"))
    discriminator.load_state_dict(torch.load(f"./weights/{opt.run}/discriminator_latest.pt"))

    gen_num = 5

    for vehicle_type_id in range(0,9):
        vehicle_type = np.zeros(9)
        vehicle_type[vehicle_type_id] = 1
        for speed in [1, 10, 15, 20, 30, 50]:
            for terrain_id in range(3):
                terrain = np.zeros(3)
                terrain[terrain_id] = 1
                for distance in [5, 10, 25, 50]:
                    z = Variable(FloatTensor(np.random.normal(0, 1, (gen_num, opt.signal_len))))
                    # print("vehicle_type: ", vehicle_type, ", speed: ", speed, ", terrain: ", terrain, ", distance: ", distance)
                    gen_labels = np.concatenate((vehicle_type, [speed], terrain, [distance]), axis=-1)
                    gen_labels = np.tile(gen_labels, (gen_num, 1))
                    # print("z.shape: ", z.shape)
                    gen_labels = Variable(FloatTensor(gen_labels))
                    # print("gen_labels.shape: ", gen_labels.shape)
                    # gen_labels = gen_labels.repeat(gen_num, 1)
                    gen_samples = generator(z, gen_labels)
                    gen_samples = gen_samples.detach().cpu().numpy()
                    gen_labels = gen_labels.detach().cpu().numpy()
                    visualize_signals(gen_samples, gen_labels, f"{vehicle_type_id}_{speed}_{terrain_id}_{distance}", f"./visualization/{opt.run}/test", skip_n=1)
else:
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # ----------
    #  Training
    # ----------

    for epoch in range(opt.n_epochs):
        # for i, (imgs, labels) in enumerate(dataloader):

        all_dis_loss = []
        all_gen_loss = []

        all_real_samples = []
        all_real_labels = []

        all_fake_samples = []
        all_fake_labels = []

        batch_samples = []
        batch_labels = []

        # Randomly shuffle the data
        temp = list(zip(train_X, train_Y))
        np.random.shuffle(temp)
        train_X, train_Y = zip(*temp)
        train_X, train_Y = list(train_X), list(train_Y)

        for i, (sample, label) in enumerate(zip(train_X, train_Y)):
            batch_samples.append(sample)
            batch_labels.append(label)
            if len(batch_samples) < opt.batch_size:
                continue

            # batch_samples = np.array(batch_samples)
            batch_samples = torch.tensor(batch_samples)
            batch_labels = torch.tensor(batch_labels)
            # print("batch_samples.shape", batch_samples.shape)
            # batch_size = imgs.shape[0]
            batch_size = batch_samples.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            # real_imgs = Variable(imgs.type(FloatTensor))
            # labels = Variable(labels.type(LongTensor))

            real_samples = Variable(batch_samples.type(FloatTensor))
            labels = Variable(batch_labels.type(LongTensor))
            # labels = torch.argmax(labels, dim=-1)
            all_real_samples.append(real_samples.cpu().detach().numpy())
            all_real_labels.append(labels.cpu().detach().numpy())

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.signal_len))))

            # gen_vehicle_type = np.eye(9)[np.random.choice(9, batch_size)]
            # gen_speed = np.random.uniform(0, 50, (batch_size, 1))
            # gen_terrain = np.eye(3)[np.random.choice(3, batch_size)]
            # gen_distance = np.random.uniform(0, 50, (batch_size, 1))

            gen_vehicle_type = np.eye(9)[np.random.choice(9, batch_size)]
            gen_speed = [[1, 10, 15, 20, 30, 50][np.random.choice(6)] for _ in range(batch_size)]
            gen_speed = np.array(gen_speed).reshape(-1, 1)
            gen_terrain = np.eye(3)[np.random.choice(3, batch_size)]
            gen_distance = [[5, 10, 25, 50][np.random.choice(4)] for _ in range(batch_size)]
            gen_distance = np.array(gen_distance).reshape(-1, 1)
            
            gen_labels = np.concatenate((gen_vehicle_type, gen_speed, gen_terrain, gen_distance), axis=-1)
            gen_labels = Variable(FloatTensor(gen_labels))
            # gen_labels = Variable(batch_labels.type(FloatTensor))

            # print("gen_labels: ", gen_labels.shape)
            gen_samples = generator(z, gen_labels)

            all_fake_samples.append(gen_samples.detach().cpu().numpy())
            all_fake_labels.append(gen_labels.detach().cpu().numpy())

            # print("gen_samples.shape: ", gen_samples.shape)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_samples, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_samples, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_samples.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i % 25 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(train_X), d_loss.item(), g_loss.item())
                )
            batches_done = epoch * len(train_X) + i
            # if batches_done % opt.sample_interval == 0:
            #     sample_image(n_row=10, batches_done=batches_done)

            all_dis_loss.append(d_loss.item())
            all_gen_loss.append(g_loss.item())

            batch_samples = []
            batch_labels = []

        torch.save(discriminator.state_dict(), f"./weights/{opt.run}/discriminator_latest.pt")
        torch.save(generator.state_dict(), f"./weights/{opt.run}/generator_latest.pt")

        all_real_samples = np.concatenate(all_real_samples, axis=0)
        all_real_labels = np.concatenate(all_real_labels, axis=0)
        visualize_signals(all_real_samples, all_real_labels, "real", f"./visualization/{opt.run}")

        all_fake_samples = np.concatenate(all_fake_samples, axis=0)
        all_fake_labels = np.concatenate(all_fake_labels, axis=0)
        visualize_signals(all_fake_samples, all_fake_labels, "fake", f"./visualization/{opt.run}")