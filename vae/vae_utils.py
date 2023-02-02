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

def visualize_reconstruct_signals(n_start, batch_signal, batch_label, batch_gen, output_path, skip_n=1000):
    # batch_signal = batch_signal[::skip_n]
    # batch_label = batch_label[::skip_n]
    
    for i, (signal, gen, label) in enumerate(zip(batch_signal, batch_gen, batch_label)):
        fig = plt.figure(figsize=(8, 12), dpi=120)
        plt.subplot(411)

        ori_aud = np.sqrt(np.square(signal[0,:]) + np.square(signal[1,:]))
        # audio1 = np.abs(np.fft.fft(audio1))[1: len(audio1) // 2]
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot(ori_aud, color='red')
        ax1.set_title('Original Audio')

        ori_sei = np.sqrt(np.square(signal[2,:]) + np.square(signal[3,:]))
        # seismic1 = np.abs(np.fft.fft(seismic1))[1: len(seismic1) // 2]
        ax2 = plt.subplot(4, 1, 2)
        ax2.plot(ori_sei, color='red')
        ax2.set_title('Original Seismic')

        gen_aud = np.sqrt(np.square(gen[0,:]) + np.square(gen[1,:]))
        ax3 = plt.subplot(4, 1, 3)
        ax3.plot(gen_aud, color='blue')
        ax3.set_title('Gen Audio')

        gen_sei = np.sqrt(np.square(gen[2,:]) + np.square(gen[3,:]))
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

    aud = np.sqrt(np.square(signal[0,:]) + np.square(signal[1,:]))
    # audio1 = np.abs(np.fft.fft(audio1))[1: len(audio1) // 2]
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(aud, color='blue')
    ax1.set_title('Audio')

    sei = np.sqrt(np.square(signal[2,:]) + np.square(signal[3,:]))
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

def stft_to_time_seq(array_real, array_imag, fs=1024):
    _, time_seq = signal.istft(array_real+1j*array_imag, fs=fs)
    return time_seq

def time_seq_to_fft(time_seq, fs=1024):
    return np.abs(fft(time_seq)[1: len(time_seq) // 2])

def visualize_reconstruct_spect(n_start, batch_signal, batch_label, batch_gen, output_path, skip_n=1000):
    # print("batch_signal.shape: ", batch_signal.shape)
  
    for i, (signal, gen, label) in enumerate(zip(batch_signal, batch_gen, batch_label)):
        if i % skip_n != 0:
            continue
        # print("ori means: ", np.mean(signal, axis=(1,2)))
        # print("gen means: ", np.mean(gen, axis=(1,2)))

        f_len = signal.shape[1]
        t_len = signal.shape[2]

        ori_audio_real = signal[0]
        ori_audio_imag = signal[1]
        ori_audio_abs = np.sqrt(np.square(ori_audio_real) + np.square(ori_audio_imag))
        # print("ori_audio_abs: ", ori_audio_abs.shape)
        ori_seismic_real = signal[2]
        ori_seismic_imag = signal[3]
        ori_seismic_abs = np.sqrt(np.square(ori_seismic_real) + np.square(ori_seismic_imag))

        gen_audio_real = gen[0]
        gen_audio_imag = gen[1]
        gen_audio_abs = np.sqrt(np.square(gen_audio_real) + np.square(gen_audio_imag))
        # print("gen_audio_abs: ", gen_audio_abs.shape)

        gen_seismic_real = gen[2]
        gen_seismic_imag = gen[3]
        gen_seismic_abs = np.sqrt(np.square(gen_seismic_real) + np.square(gen_seismic_imag))

        fig = plt.figure(figsize=(30, 12), dpi=120)
        # plt.subplot()

        audio_vmin = np.min([ori_audio_abs, gen_audio_abs])
        audio_vmax = np.max([ori_audio_abs, gen_audio_abs])

        seismic_vmin = np.min([ori_seismic_abs, gen_seismic_abs])
        seismic_vmax = np.max([ori_seismic_abs, gen_seismic_abs])

        ax1 = plt.subplot(3, 4, 1)
        plt.pcolormesh(range(t_len), range(f_len), ori_audio_abs, vmin=audio_vmin, vmax=audio_vmax, shading='gouraud', cmap="plasma")
        ax1.set_title('Original Audio')

        ax2 = plt.subplot(3, 4, 2)
        plt.pcolormesh(range(t_len), range(f_len), ori_seismic_abs, vmin=seismic_vmin, vmax=seismic_vmax, shading='gouraud', cmap="plasma")
        ax2.set_title('Original Seismic')

        ax3 = plt.subplot(3, 4, 3)
        plt.pcolormesh(range(t_len), range(f_len), gen_audio_abs, vmin=audio_vmin, vmax=audio_vmax, shading='gouraud', cmap="plasma")
        ax3.set_title('Gen Audio')

        ax4 = plt.subplot(3, 4, 4)
        plt.pcolormesh(range(t_len), range(f_len), gen_seismic_abs, vmin=seismic_vmin, vmax=seismic_vmax, shading='gouraud', cmap="plasma")
        ax4.set_title('Gen Seismic')      

        # Plot FFT
        ori_audio_time_seq = stft_to_time_seq(ori_audio_real, ori_audio_imag)
        ori_seismic_time_seq = stft_to_time_seq(ori_seismic_real, ori_seismic_imag)
        gen_audio_time_seq = stft_to_time_seq(gen_audio_real, gen_audio_imag)
        gen_seismic_time_seq = stft_to_time_seq(gen_seismic_real, gen_seismic_imag)

        ori_audio_fft = time_seq_to_fft(ori_audio_time_seq)
        ori_seismic_fft = time_seq_to_fft(ori_seismic_time_seq)
        gen_audio_fft = time_seq_to_fft(gen_audio_time_seq)
        gen_seismic_fft = time_seq_to_fft(gen_seismic_time_seq)

        ax5 = plt.subplot(3, 4, 5)
        ax5.plot(ori_audio_fft)
        ax5.set_title('Original Audio FFT')

        ax6 = plt.subplot(3, 4, 6)
        ax6.plot(ori_seismic_fft)
        ax6.set_title('Original Seismic FFT')

        ax7 = plt.subplot(3, 4, 7)
        ax7.plot(gen_audio_fft)
        ax7.set_title('Gen Audio FFT')

        ax8 = plt.subplot(3, 4, 8)
        ax8.plot(gen_seismic_fft)
        ax8.set_title('Gen Seismic FFT')      
 
        # Plot time sequence
        ax9 = plt.subplot(3, 4, 9)
        ax9.plot(ori_audio_time_seq)
        ax9.set_title('Original Audio Time Seq')

        ax10 = plt.subplot(3, 4, 10)
        ax10.plot(ori_seismic_time_seq)
        ax10.set_title('Original Seismic Time Seq')

        ax11 = plt.subplot(3, 4, 11)
        ax11.plot(gen_audio_time_seq)
        ax11.set_title('Gen Audio Time Seq')

        ax12 = plt.subplot(3, 4, 12)
        ax12.plot(gen_seismic_time_seq)
        ax12.set_title('Gen Seismic Time Seq')      

        fig.suptitle("Vehicle Type: {}, Speed: {}, Terrain: {}, Distance: {}".format(
            np.argmax(label[:9]), label[9], np.argmax(label[10:13]), label[13]))

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "{}.png".format(n_start + i)))
        plt.clf()
        plt.cla()
    plt.close()