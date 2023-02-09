import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
import yaml
from scipy import signal
from scipy.fft import fft

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_yaml(file_path):
    """Load the YAML config file

    Args:
        file_path (_type_): _description_
    """
    with open(file_path, "r", errors="ignore") as stream:
        yaml_data = yaml.safe_load(stream)

    return yaml_data

class EarlyStop:
    """Used to early stop the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=0, 
                 save_name="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_name (string): The filename with which the model and the optimizer is saved when improved.
                            Default: "checkpoint.pt"
        """
        self.patience = patience
        self.verbose = verbose
        self.save_name = save_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0
            
        return self.early_stop

    def save_checkpoint(self, val_loss, model, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        state = {"net":model.state_dict(), "optimizer":optimizer.state_dict()}
        torch.save(state, self.save_name)
        self.val_loss_min = val_loss

def mask_training_data(Xs, Ys, masked_vehicle_types, masked_terrain_types):
    """Used to mask the training data for the specified vehicle types and terrain types.
    Args:
        Xs (np.array): The training data.
        Ys (np.array): The labels for the training data.
        masked_vehicle_types (list): The vehicle types to be masked.
        masked_terrain_types (list): The terrain types to be masked.
    Returns:
        Xs (np.array): The masked training data.
        Ys (np.array): The masked labels for the training data.
    """
    masked_Xs = []
    masked_Ys = []
    for x, y in zip(Xs, Ys):
        if np.argmax(y[:9]) not in masked_vehicle_types or np.argmax(y[10:13]) not in masked_terrain_types:
            masked_Xs.append(x)
            masked_Ys.append(y)
    
    return np.array(masked_Xs), np.array(masked_Ys)

def get_masked_data(Xs, Ys, masked_vehicle_types, masked_terrain_types):
    """
    Used to mask the training data for the specified vehicle types and terrain types.
    Args:
        Xs (np.array): The training data.
        Ys (np.array): The labels for the training data.
        masked_vehicle_types (list): The vehicle types to be masked.
        masked_terrain_types (list): The terrain types to be masked.
    Returns:
        Xs (np.array): The masked training data.
        Ys (np.array): The masked labels for the training data.
    """
    masked_Xs = []
    masked_Ys = []
    for x, y in zip(Xs, Ys):
        if np.argmax(y[:9]) in masked_vehicle_types and np.argmax(y[10:13]) in masked_terrain_types:
            masked_Xs.append(x)
            masked_Ys.append(y)
    
    return np.array(masked_Xs), np.array(masked_Ys)

def one_hot_encode(n, N):
    enc = np.zeros(N)
    enc[n] = 1
    return enc.tolist()

def label_to_attributes(label):
    vehicle_type = np.argmax(label[:9])
    speed_type = label[9]
    terrain_type = np.argmax(label[10:13])
    distance_type = label[13]
    return vehicle_type, speed_type, terrain_type, distance_type

def attributes_to_label(vehicle_type, speed_type, terrain_type, distance_type):
    label = one_hot_encode(vehicle_type, 9) + [float(speed_type)] + one_hot_encode(terrain_type, 3) + [float(distance_type)]
    return label

def save_as_pt(x, y, output_path):
    x = np.transpose(x, (0, 2, 1))
    vehicle_type, speed_type, terrain_type, distance_type = label_to_attributes(y)

    speed_enc = np.zeros(3)
    if speed_type <= 10:
        speed_enc[0] = 1
    elif speed_type <= 30:
        speed_enc[1] = 1
    else:
        speed_enc[2] = 1

    distance_enc = np.zeros(2)
    if distance_type <= 25:
        distance_enc[0] = 1
    else:
        distance_enc[1] = 1

    sample = {"data": 
    {"shake": 
        {"audio": np.expand_dims(x[0], 0), 
        "seismic": np.expand_dims(x[1], 0)}}, 
        "label": {"vehicle_type": np.array(one_hot_encode(vehicle_type, 9)), "speed": speed_enc, 
            "terrain_type": np.array(one_hot_encode(terrain_type, 3)), "distance": distance_enc}}
    torch.save(sample, output_path)

def stft_to_time_seq(array_real, array_imag, fs=1024):
    _, time_seq = signal.istft(array_real+1j*array_imag, fs=fs)
    return time_seq

def time_seq_to_fft(time_seq, fs=1024):
    return np.abs(fft(time_seq)[1: len(time_seq) // 2])

