import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from vae_utils import stft_to_time_seq, time_seq_to_fft


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

def visualize_reconstruct_spect(n_start, batch_signal, batch_label, batch_gen, output_path, skip_n=1000):
    # print("batch_signal.shape: ", batch_signal.shape)
  
    for i, (signal, gen, label) in enumerate(zip(batch_signal, batch_gen, batch_label)):
        # print("ori means: ", np.mean(signal, axis=(1,2)))
        # print("gen means: ", np.mean(gen, axis=(1,2)))
        # print("signal: ", signal.shape)
        # print("gen: ", gen.shape)
        # print("label: ", label.shape)

        signal = np.transpose(signal, (0, 2, 1))
        gen = np.transpose(gen, (0, 2, 1))

        f_len = signal.shape[1]
        t_len = signal.shape[2]
        # 0    1     2    3    4    5    6    7    8    9 
        # a1r, a2r, a3r, s1r, s2r, a1i, a2i, a3i, s1i, s2i
        # ori_audio_abs = np.sqrt( np.square(signal[0]) + np.square(signal[5]) )
        # ori_seismic_abs = np.sqrt( np.square(signal[3]) + np.square(signal[8]) )

        # gen_audio_abs = np.sqrt( np.square(gen[0]) + np.square(gen[5]) )
        # gen_seismic_abs = np.sqrt( np.square(gen[3]) + np.square(gen[8]) )

        ori_audio_abs = signal[0]
        ori_seismic_abs = signal[1]

        gen_audio_abs = gen[0]
        gen_seismic_abs = gen[1]

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

        # Back to time domain
        ori_audio_time_seq = stft_to_time_seq(ori_audio_abs, ori_audio_abs)
        ori_seismic_time_seq = stft_to_time_seq(ori_seismic_abs, ori_seismic_abs)
        gen_audio_time_seq = stft_to_time_seq(gen_audio_abs, gen_audio_abs)
        gen_seismic_time_seq = stft_to_time_seq(gen_seismic_abs, gen_seismic_abs)

        # Do FFT
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


# n_start, spect, output_path
def visualize_single_spect(name, spect, label, output_path):
    f_len = spect.shape[1]
    t_len = spect.shape[2]

    spect_audio_abs = spect[0]
    spect_seismic_abs = spect[1]

    fig = plt.figure(figsize=(30, 12), dpi=120)
    # plt.subplot()

    audio_vmin = np.min(spect_audio_abs)
    audio_vmax = np.max(spect_audio_abs)

    seismic_vmin = np.min(spect_seismic_abs)
    seismic_vmax = np.max(spect_seismic_abs)

    ax1 = plt.subplot(3, 2, 1)
    plt.pcolormesh(range(t_len), range(f_len), spect_audio_abs, vmin=audio_vmin, vmax=audio_vmax, shading='gouraud', cmap="plasma")
    ax1.set_title('Audio STFT')

    ax2 = plt.subplot(3, 2, 2)
    plt.pcolormesh(range(t_len), range(f_len), spect_seismic_abs, vmin=seismic_vmin, vmax=seismic_vmax, shading='gouraud', cmap="plasma")
    ax2.set_title('Seismic STFT')

    # Plot FFT
    audio_time_seq = stft_to_time_seq(spect_audio_abs, spect_audio_abs)
    seismic_time_seq = stft_to_time_seq(spect_seismic_abs, spect_seismic_abs)

    audio_fft = time_seq_to_fft(audio_time_seq)
    seismic_fft = time_seq_to_fft(seismic_time_seq)
    
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(audio_fft)
    ax3.set_title('Audio FFT')

    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(seismic_fft)
    ax4.set_title('Seismic FFT')

    # Plot time sequence
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(audio_time_seq)
    ax5.set_title('Audio Time Seq')

    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(seismic_time_seq)
    ax6.set_title('Seismic Time Seq')

    fig.suptitle("Vehicle Type: {}, Speed: {}, Terrain: {}, Distance: {}".format(
        np.argmax(label[:9]), label[9], np.argmax(label[10:13]), label[13]))

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, name))
    plt.clf()
    plt.cla()
    plt.close()