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
                # q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
                q.append(("Sigmoid_%d" % i, nn.Sigmoid()))

        self.mlp = nn.Sequential(OrderedDict(q))
    def forward(self, x):
        return self.mlp(x)
    
class Encoder(nn.Module):
    def __init__(self, signal_len, nhid = 16, ncond = 0):
        super(Encoder, self).__init__()
        
        def linear_block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            # layers.append(nn.Sigmoid())
            layers.append(nn.ReLU())
            return layers

        self.aud_encode_real = nn.Sequential(
            *linear_block(signal_len, 256, normalize=False),
            *linear_block(256, 128),
        )
        self.aud_encode_imag = nn.Sequential(
            *linear_block(signal_len, 256, normalize=False),
            *linear_block(256, 128),
        )

        self.sei_encode_real = nn.Sequential(
            *linear_block(signal_len, 256, normalize=False),
            *linear_block(256, 128),
        )  
        self.sei_encode_imag = nn.Sequential(
            *linear_block(signal_len, 256, normalize=False),
            *linear_block(256, 128),
        )       

        self.encode = nn.Sequential(
            *linear_block(128, 64, normalize=True),
        )

        self.calc_mean = MLP([64+ncond, 64, nhid], last_activation = False)
        self.calc_logvar = MLP([64+ncond, 64, nhid], last_activation = False)

    def forward(self, x, y = None):
        # print("x.shape: ", x.shape)
        x_aud_real = x[:,0,:]
        x_aud_imag = x[:,1,:]
        x_sei_real = x[:,2,:]
        x_sei_imag = x[:,3,:]

        # print("x_aud_real.shape: ", x_aud_real.shape)
        # print("x_sei_real.shape: ", x_sei_real.shape)
        # x_aud = torch.transpose(x_aud, 1, 2)
        # x_sei = torch.transpose(x_sei, 1, 2)
        x_aud_real = torch.unsqueeze(self.aud_encode_real(x_aud_real), axis=1)
        x_aud_imag = torch.unsqueeze(self.aud_encode_imag(x_aud_imag), axis=1)
        x_sei_real = torch.unsqueeze(self.sei_encode_real(x_sei_real), axis=1)
        x_sei_imag = torch.unsqueeze(self.sei_encode_imag(x_sei_imag), axis=1)
        # x_aud = torch.unsqueeze(self.aud_encode(x_aud), axis=1)
        # x_sei = torch.unsqueeze(self.sei_encode(x_sei), axis=1)
    
        x = torch.cat((x_aud_real, x_aud_imag, x_sei_real, x_sei_imag), axis=1)
        # print("after concat, x.shape: ", x.shape)
        x = torch.mean(x, axis=1)
        x = self.encode(x)
        # y = self.meta_mapping(y)
        if (y is None):
            return self.calc_mean(x), self.calc_logvar(x)
        else:
            return self.calc_mean(torch.cat((x, y), dim=1)), self.calc_logvar(torch.cat((x, y), dim=1))


class Decoder(nn.Module):
    def __init__(self, signal_len, nhid = 16, ncond=0):
        super(Decoder, self).__init__()
        self.signal_len = signal_len
        # self.decode = nn.Sequential(MLP([nhid+ncond, 128, 256, 512, 4*signal_len], last_activation = False), nn.Sigmoid())
        self.decode = nn.Sequential(
            MLP([nhid+ncond, 128, 256, 512]),
            MLP([512, 512, 4*signal_len], last_activation=False), 
            nn.ReLU())
    def forward(self, z, y = None):
        if (y is None):
            return self.decode(z).view(-1, 4, self.signal_len)
        else:
            # print("z.shape: ", z.shape)
            # print("y.shape: ", y.shape)
            # print("torch.cat((z, y), dim=1): ", torch.cat((z, y), dim=1).shape)
            return self.decode(torch.cat((z, y), dim=1)).view(-1, 4, self.signal_len)

class cVAE_1d(nn.Module):
    def __init__(self, signal_len, label_len, nhid = 16, ncond = 64):

        # Encoder: def __init__(self, signal_len, nhid = 16):
        # Decoder: def __init__(self, signal_len, nhid = 16):
        super(cVAE_1d, self).__init__()
        self.dim = nhid
        self.encoder = Encoder(signal_len, nhid, ncond=ncond)
        self.decoder = Decoder(signal_len, nhid, ncond=ncond)
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
    
# BCE_loss = nn.BCELoss(reduction = "sum")
def loss_1d(X, X_hat, mean, logvar):
    # reconstruction_loss = BCE_loss(X_hat, X) # better than MSE
    reconstruction_loss = torch.mean(torch.pow(X_hat - X, 2))
    # print("logvar: ", torch.mean(logvar), ", mean: ", torch.mean(mean))
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
    # print("reconstruction_loss: ", reconstruction_loss, ", KL_divergence * beta: ", KL_divergence*beta)
    # return reconstruction_loss + beta * KL_divergence
    return reconstruction_loss, KL_divergence