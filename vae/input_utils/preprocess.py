import torch
import numpy as np

def fft_preprocess(xs):
    xs = torch.concat([xs['shake']['audio'], xs['shake']['seismic']], axis=1)
    xs_fft = torch.fft.fft(xs, dim=-1)
    # print("xs_fft.shape: ", xs_fft.shape)
    xs_fft = xs_fft[:,:,:,:128] # Throw out the duplicate spectrum
    xs_fft_audio = xs_fft[:, 0:3, :, :]
    xs_fft_seismic = xs_fft[:, 3:5, :, :]

    xs_fft_audio = torch.view_as_real(xs_fft_audio)
    xs_fft_audio = torch.permute(xs_fft_audio, (0, 1, 4, 2, 3))
    # print("xs_fft_audio.shape: ", xs_fft_audio.shape)
    b, c1, c2, i, s = xs_fft_audio.shape
    xs_fft_audio = xs_fft_audio.reshape(b, c1 * c2, i, s) # [64, 10, 7, 256]
    # print("after reshape xs_fft_audio.shape: ", xs_fft_audio.shape)

    xs_fft_seismic = torch.view_as_real(xs_fft_seismic)
    # print("xs_fft_seismic as real: ", xs_fft_seismic.shape)
    xs_fft_seismic = torch.permute(xs_fft_seismic, (0, 1, 4, 2, 3))
    b, c1, c2, i, s = xs_fft_seismic.shape
    xs_fft_seismic = xs_fft_seismic.reshape(b, c1 * c2, i, s) # [64, 10, 7, 256]
    # print("after reshape xs_fft_seismic.shape: ", xs_fft_seismic.shape)

    xs_fft = torch.concat([xs_fft_audio, xs_fft_seismic], axis=1) # [64, 10 (6 + 4), 7, 256]
    # print("xs.shape: ", xs_fft.shape)

    return xs_fft

def fft_abs_preprocess(xs):
    xs = torch.concat([xs['shake']['audio'], xs['shake']['seismic']], axis=1)
    xs_fft = torch.fft.fft(xs, dim=-1)
    # print("xs_fft.shape: ", xs_fft.shape)
    xs_fft = xs_fft[:,:,:,:128] # Throw out the duplicate spectrum
    xs_fft = torch.abs(xs_fft)

    xs_fft_audio = xs_fft[:, 0:1, :, :]
    xs_fft_seismic = xs_fft[:, 3:4, :, :]

    # print("xs_fft_audio.shape: ", xs_fft_audio.shape)

    xs_fft = torch.concat([xs_fft_audio, xs_fft_seismic], axis=1) # [64, 2, 7, 256]
    # print("xs.shape: ", xs_fft.shape)

    return xs_fft

def one_hot_embed(ns, N):
    batch_size = ns.shape[0]
    emb = torch.zeros((batch_size, N))
    for i, n in enumerate(ns):
        emb[i][n] = 1
    return emb

def reformat_labels(labels):
    vehicle_type = labels['vehicle_type'] # 0 ~ 10, 10 is no vehicle
    vehicle_type_emb = one_hot_embed(vehicle_type, 10)
    terrain = labels['terrain'] 
    terrain_emb = one_hot_embed(terrain, 3)
    speed = torch.unsqueeze(labels['speed'], dim=-1)
    distance = torch.unsqueeze(labels['distance'], dim=-1)

    return torch.cat((vehicle_type_emb, speed, terrain_emb, distance), dim=1)

def preprocess(batch_data, batch_label):
    batch_label = reformat_labels(batch_label)
    # batch_data = fft_preprocess(batch_data)
    batch_data = fft_abs_preprocess(batch_data)

    # print("batch_label.shape: ", batch_label.shape)
    # print("batch_data: ", batch_data.shape)

    return batch_data, batch_label
