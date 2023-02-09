import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        # print("x.shape: ", x.shape)
        batch_size = x.shape[0]
        # print("x.view(batch_size, -1): ", x.view(batch_size, -1).shape)
        return x.view(batch_size, -1)

class Mean(nn.Module):
    def __init__(self):
        super(Mean, self).__init__()
    def forward(self, x):
        return torch.mean(x, axis=1)

class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation = True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size)-1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size)-2) or ((i == len(hidden_size) - 2) and (last_activation)):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
                # q.append(("Sigmoid_%d" % i, nn.Sigmoid()))
                q.append(("Dropout_%d" % i, nn.Dropout(p=0.3)))

        self.mlp = nn.Sequential(OrderedDict(q))
    def forward(self, x):
        return self.mlp(x)
    
class Encoder(nn.Module):
    def __init__(self, spect_shape, nhid = 128, ncond = 32):
        super(Encoder, self).__init__()
        
        def conv_block(in_channels, out_channels, kernel_size, normalize=True, stride=1, padding="same"):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=not normalize)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3))
            return layers

        self.aud_encode = nn.Sequential(
            *conv_block(1, 32, 5, stride=2, padding=2),
            # nn.MaxPool2d((2, 2), stride=2),
            *conv_block(32, 64, 5, stride=2, padding=2),
            # nn.MaxPool2d((2, 2), stride=2),
            # *conv_block(128, 256, 5, normalize=True, stride=2, padding=2),
        )

        self.sei_encode = nn.Sequential(
            *conv_block(1, 32, 5, stride=2, padding=2),
            # nn.MaxPool2d((2, 2), stride=2),
            *conv_block(32, 64, 5, stride=2, padding=2),
            # nn.MaxPool2d((2, 2), stride=2),
            # *conv_block(128, 256, 5, normalize=True, stride=2, padding=2),
        )  

        self.encode = nn.Sequential(
            *conv_block(128, 256, 5, stride=2, padding=1),
            # nn.MaxPool2d((2, 2), stride=2),
        )
        print("spect_shape: ", spect_shape)
        # print("256*spect_shape[0]//8*spect_shape[1]//8 = ", 256*spect_shape[0]//8*spect_shape[1]//8)

        self.fc = nn.Sequential(nn.Linear(256*(spect_shape[0]//8)*(spect_shape[1]//8), 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

        self.calc_mean = MLP([1024+ncond, nhid])
        self.calc_logvar = MLP([1024+ncond, nhid])


    def forward(self, x, y = None):
        # print("x.shape: ", x.shape)
        # x_aud = x[:,0:2,:,:]
        # x_sei = x[:,2:4,:,:]

        x_aud = x[:,0:1,:]
        x_sei = x[:,1:2,:]

        # print("x_sei_real.shape: ", x_sei_real.shape)
        # x_aud = torch.spose(x_aud, 1, 2)
        # x_sei = torch.transpose(x_sei, 1, 2)
        x_aud = self.aud_encode(x_aud)
        x_sei = self.sei_encode(x_sei)

        # print("after mod encode: ", x_aud.shape)
        
        # x_aud = torch.unsqueeze(self.aud_encode(x_aud), axis=1)
        # x_sei = torch.unsqueeze(self.sei_encode(x_sei), axis=1)

        x = torch.cat((x_aud, x_sei), axis=1)
        # print("after concat, x.shape: ", x.shape)
        x = self.encode(x)
        x = x.view(x.shape[0], -1) # flatten
        x = self.fc(x)

        mean = self.calc_mean(torch.cat((x, y), dim=1))
        logvar = self.calc_logvar(torch.cat((x, y), dim=1))

        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, spect_shape, nhid = 128, ncond=32):
        super(Decoder, self).__init__()
        
        self.spect_shape = spect_shape

        def conv_trans_block(in_channels, out_channels, kernel_size, stride=2, dilation=1, padding=2, output_padding=1, activation=True, normalize=True):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                stride=stride, dilation=dilation, padding=padding, output_padding=output_padding, bias=not normalize)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            if activation:
                layers.append(nn.ReLU())
            return layers
       
        self.fc = nn.Sequential(MLP([ncond+nhid, 256*(spect_shape[0]//8)*(spect_shape[1]//8)]))
        
        self.deconv = nn.Sequential(
            *conv_trans_block(256, 256, 5),
            # *conv_trans_block(256, 128, 5),
            # *conv_trans_block(128, 32, 5, padding=1, output_padding=0, activation=False),
            # nn.Conv2d(32, 4, 5, padding="same"),
            # nn.Tanh()
        )
        
        self.aud_deconv = nn.Sequential(
            *conv_trans_block(128, 64, 5),
            *conv_trans_block(64, 32, 5, padding=1, output_padding=0, activation=False),
            nn.Conv2d(32, 1, 5, padding="same"),        
        )

        self.sei_deconv = nn.Sequential(
            *conv_trans_block(128, 64, 5),
            *conv_trans_block(64, 32, 5, padding=1, output_padding=0, activation=False),
            nn.Conv2d(32, 1, 5, padding="same"),    
        )

    def forward(self, z, y):
        # print("z.shape: ", z.shape)
        # print("y.shape: ", y.shape)
        # print("torch.cat((z, y), dim=1): ", torch.cat((z, y), dim=1).shape)

        z = self.fc(torch.cat((z, y), dim=1)).view(-1, 256, self.spect_shape[0]//8, self.spect_shape[1]//8)
        # print("after linear decoding, z.shape: ", z.shape)
        z = self.deconv(z)
        # print("after deconv: ", z.shape)
        (z_aud, z_sei) = torch.split(z, [128, 128], dim=1)
        # print("z_aud.shape: ", z_aud.shape)
        z_aud = self.aud_deconv(z_aud)
        z_sei = self.sei_deconv(z_sei)
        # print("after aud_deconv, z_aud.shape: ", z_aud.shape)
        res = torch.cat((z_aud, z_sei), axis=1)
        # exit()
        # print("after tranpose conv, res.shape: ", res.shape)
        return res

class cVAE_2d(nn.Module):
    def __init__(self, spect_shape, label_len, nhid = 128, ncond = 32):

        # Encoder: def __init__(self, signal_len, nhid = 16):
        # Decoder: def __init__(self, signal_len, nhid = 16):
        super(cVAE_2d, self).__init__()
        self.dim = nhid
        self.encoder = Encoder(spect_shape, nhid=nhid, ncond=ncond)
        self.decoder = Decoder(spect_shape, nhid=nhid, ncond=ncond)
        self.label_embedding = nn.Linear(label_len, ncond)
        
    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x, y):
        y = self.label_embedding(y)
        mean, logvar = self.encoder(x, y)
        z = self.sampling(mean, logvar)
        return self.decoder(z, y), mean, logvar
    
    def generate(self, label):
        label = np.expand_dims(label, axis=0)
        label = torch.tensor(label)
        label = label.to(device).float()
        if (len(label.shape) == 0):
            batch_size = None
            label = label.unsqueeze(0)
            z = torch.randn((1, self.dim)).to(device)
        else:
            batch_size = label.shape[0]
            z = torch.randn((1, self.dim)).to(device)
        # print("label: ", label.shape)
        y = self.label_embedding(label)
        # print("y.shape: ", y.shape)
        # print("z.shape: ", z.shape)
        res = self.decoder(z, y)
        if not batch_size:
            res = res.squeeze(0)
        return res
    
def loss_2d(X, X_hat, mean, logvar):
    # print("X.shape: ", X.shape, ", X mean: ", torch.mean(X))
    # print("X_hat.shape: ", X_hat.shape, ", X_hat mean: ", torch.mean(X_hat))
    # exit()

    # print("X mean: ", torch.mean(X, axis=(0, 2, 3)), ", X_hat mean: ", torch.mean(X_hat, axis=(0, 2, 3)))
    # reconstruction_loss = BCE_loss(X_hat, X) # better than MSE
    # audio_abs_loss = torch.mean(torch.square(torch.sqrt(torch.square(X_hat[0]) + torch.square(X_hat[1])) - 
    #     torch.sqrt(torch.square(X[0]) + torch.square(X[1]))))
    # seismic_abs_loss = torch.mean(torch.square(torch.sqrt(torch.square(X_hat[2]) + torch.square(X_hat[3])) - 
    #     torch.sqrt(torch.square(X[2]) + torch.square(X[3]))))    
    
    # reconstruction_loss = audio_abs_loss + seismic_abs_loss

    # X_audio = X[0:2]
    # norm_X_audio = (X_audio - torch.mean(X_audio)) / torch.std(X_audio)
    # X_seismic = X[2:4]
    # norm_X_seismic = (X_seismic - torch.mean(X_seismic)) / torch.std(X_seismic)
    # X_hat_audio = X_hat[0:2]
    # norm_X_hat_audio = (X_hat_audio - torch.mean(X_hat_audio)) / torch.std(X_hat_audio)
    # X_hat_seismic = X_hat[2:4]
    # norm_X_hat_seismic = (X_hat_seismic - torch.mean(X_hat_seismic)) / torch.std(X_hat_seismic)
    # reconstruction_loss = torch.mean(torch.square(norm_X_audio - norm_X_hat_audio)) + torch.mean(torch.square(norm_X_seismic - norm_X_hat_seismic))

    # reconstruction_loss = []
    # for i in range(4):
        # x = (X[:,i,:,:] - torch.mean(X[:,i,:,:])) / torch.std(X[:,i,:,:])
        # x_hat = (X_hat[:,i,:,:] - torch.mean(X_hat[:,i,:,:])) / torch.std(X_hat[:,i,:,:]) 
        # reconstruction_loss += torch.mean(torch.square(x - x_hat))
        # reconstruction_loss.append(torch.mean(torch.square(x - x_hat)))
    reconstruction_loss = torch.mean(torch.square(X - X_hat))

    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
    # print("reconstruction_loss: ", reconstruction_loss, ", KL_divergence * beta: ", KL_divergence*beta)
    # return reconstruction_loss + beta * KL_divergence
    return reconstruction_loss, KL_divergence