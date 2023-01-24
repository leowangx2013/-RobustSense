import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
import yaml

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

def visualize_reconstruct_signals(n_start, batch_signal, batch_label, batch_gen, output_path, skip_n=1000):
    batch_signal = batch_signal[::skip_n]
    batch_label = batch_label[::skip_n]
    
    for i, (signal, gen, label) in enumerate(zip(batch_signal, batch_gen, batch_label)):
        fig = plt.figure(figsize=(8, 12), dpi=120)
        plt.subplot(411)

        ori_aud = signal[0,:]
        # audio1 = np.abs(np.fft.fft(audio1))[1: len(audio1) // 2]
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot(ori_aud, color='red')
        ax1.set_title('Original Audio')

        ori_sei = signal[1,:]
        # seismic1 = np.abs(np.fft.fft(seismic1))[1: len(seismic1) // 2]
        ax2 = plt.subplot(4, 1, 2)
        ax2.plot(ori_sei, color='red')
        ax2.set_title('Original Seismic')

        gen_aud = gen[0,:]
        ax3 = plt.subplot(4, 1, 3)
        ax3.plot(gen_aud, color='blue')
        ax3.set_title('Gen Audio')

        gen_sei = gen[1,:]
        ax4 = plt.subplot(4, 1, 4)
        ax4.plot(gen_sei, color='blue')
        ax4.set_title('Gen Seismic')      

        fig.suptitle("Vehicle Type: {}, Speed: {}, Terrain: {}, Distance: {}".format(
            np.argmax(label[:9]), label[9], np.argmax(label[10:13]), label[13]))

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "{}.png".format(n_start + i)))
        plt.clf()
        plt.cla()
    plt.close()

def visualize_single_signal(name, signal, label, output_path):
    signal = np.squeeze(signal)

    fig = plt.figure(figsize=(8, 6), dpi=120)
    plt.subplot(211)

    aud = signal[0,:]
    # audio1 = np.abs(np.fft.fft(audio1))[1: len(audio1) // 2]
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(aud, color='blue')
    ax1.set_title('Audio')

    sei = signal[1,:]
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(sei, color='red')
    ax2.set_title('Seismic')

    fig.suptitle("Vehicle Type: {}, Speed: {}, Terrain: {}, Distance: {}".format(
        np.argmax(label[:9]), label[9], np.argmax(label[10:13]), label[13]))

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, name))
    plt.clf()
    plt.cla()
    # plt.close()
