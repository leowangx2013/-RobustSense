import torch
import matplotlib.pyplot as plt
import numpy as np
import os

'''
def visualize_signals(batch_signal, batch_label, prefix, output_path, skip_n=1000):
    batch_signal = batch_signal[::skip_n]
    batch_label = batch_label[::skip_n]
    
    for i, (signal, label) in enumerate(zip(batch_signal, batch_label)):
        fig = plt.figure(figsize=(8, 12), dpi=120)
        plt.subplot(511)

        audio1 = signal[0,:]
        # audio1 = np.abs(np.fft.fft(audio1))[1: len(audio1) // 2]
        ax1 = plt.subplot(5, 1, 1)
        ax1.plot(audio1, color='red')
        ax1.set_title('Audio 1')

        audio2 = signal[1,:]
        # audio2 = np.abs(np.fft.fft(audio2))[1: len(audio2) // 2]
        ax2 = plt.subplot(5, 1, 2)
        ax2.plot(audio2, color='red')
        ax2.set_title('Audio 2')

        audio3 = signal[2,:]
        # audio3 = np.abs(np.fft.fft(audio3))[1: len(audio3) // 2]
        ax3 = plt.subplot(5, 1, 3)
        ax3.plot(audio3, color='red')
        ax3.set_title('Audio 3')

        seismic1 = signal[3,:]
        # seismic1 = np.abs(np.fft.fft(seismic1))[1: len(seismic1) // 2]
        ax4 = plt.subplot(5, 1, 4)
        ax4.plot(seismic1, color='blue')
        ax4.set_title('Seismic 1')

        seismic2 = signal[4,:]
        # seismic2 = np.abs(np.fft.fft(seismic2))[1: len(seismic2) // 2]
        ax5 = plt.subplot(5, 1, 5)
        ax5.plot(seismic2, color='blue')
        ax5.set_title('Seismic 2')

        fig.suptitle("Vehicle Type: {}, Speed: {}, Terrain: {}, Distance: {}".format(
            np.argmax(label[:9]), label[9], np.argmax(label[10:13]), label[13]))

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "{}_signal_{}.png".format(prefix, i)))
        plt.clf()
        plt.cla()
    plt.close()
'''

def visualize_signals(batch_signal, batch_label, prefix, output_path, skip_n=1000):
    batch_signal = batch_signal[::skip_n]
    batch_label = batch_label[::skip_n]
    
    for i, (signal, label) in enumerate(zip(batch_signal, batch_label)):
        fig = plt.figure(figsize=(8, 12), dpi=120)
        plt.subplot(211)

        audio1 = signal[0,:]
        # audio1 = np.abs(np.fft.fft(audio1))[1: len(audio1) // 2]
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(audio1, color='red')
        ax1.set_title('Audio')

        seismic1 = signal[1,:]
        # seismic1 = np.abs(np.fft.fft(seismic1))[1: len(seismic1) // 2]
        ax4 = plt.subplot(2, 1, 2)
        ax4.plot(seismic1, color='blue')
        ax4.set_title('Seismic')

        fig.suptitle("Vehicle Type: {}, Speed: {}, Terrain: {}, Distance: {}".format(
            np.argmax(label[:9]), label[9], np.argmax(label[10:13]), label[13]))

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "{}_signal_{}.png".format(prefix, i)))
        plt.clf()
        plt.cla()
    plt.close()