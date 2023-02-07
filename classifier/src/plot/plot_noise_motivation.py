import numpy as np
import time, datetime, os, csv
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import math
import os

np.set_printoptions(suppress=True)
pd.set_option("display.float_format", lambda x: "%.8f" % x)
# pd.set_option('display.max_rows', None)

matplotlib.rcParams.update({"font.size": 18})

figure_path = "/home/sl29/AutoCuration/result/figures"
root_data_path = "/home/dongxin/data/Parkland/raw_data"


def plot_aud(vehicle, sensor):
    print("Plotting", vehicle, sensor)
    df = pd.read_csv(
        os.path.join(root_data_path, vehicle + "/" + sensor + "/ehz.csv"),
        sep=" ",
        header=None,
        dtype=np.float64,
    )

    ehz = np.squeeze(df.values[:, 0])

    df = pd.read_csv(
        os.path.join(root_data_path, vehicle + "/" + sensor + "/aud16000.csv"),
        sep=",",
        header=None,
        dtype=np.float64,
    )
    aud = df.values[:, 0]
    # # 16000Hz --> 1000Hz
    aud = aud[::16]

    t_start = 450
    t_window = 40
    ind_start = 1000 * t_start
    ind_end = ind_start + 1000 * t_window
    ind_start_sei = 100 * t_start
    ind_end_sei = ind_start_sei + 100 * t_window

    t_start_noise = 260
    t_window_noise = 40
    ind_start_noise = 1000 * t_start_noise
    ind_end_noise = ind_start_noise + 1000 * t_window_noise
    ind_start_sei_noise = 100 * t_start_noise
    ind_end_sei_noise = ind_start_sei_noise + 100 * t_window_noise

    _, axs = plt.subplots(2, figsize=(12, 6), sharey=True)
    plt.ylim(-10000, 10000)
    axs[0].set_title("Acoustic Signal, No wind")
    axs[0].plot(np.array(range(1000 * t_window)) / 1000.0, aud[ind_start:ind_end], lw=0.5)
    axs[1].set_title("Acoustic Signal, With Wind")
    axs[1].plot(np.array(range(1000 * t_window_noise)) / 1000.0, aud[ind_start_noise:ind_end_noise], lw=0.5)
    for ax in axs:
        ax.label_outer()
    axs[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, "figure_noise_aud.pdf"))
    plt.close()

    _, axs = plt.subplots(2, figsize=(12, 6), sharey=True)
    plt.ylim(14000, 18000)
    axs[0].set_title("Seismic Signal, No Wind")
    axs[0].plot(np.array(range(100 * t_window)) / 100.0, ehz[ind_start_sei:ind_end_sei], lw=0.5)
    axs[1].set_title("Seismic Signal, With Wind")
    axs[1].plot(np.array(range(100 * t_window_noise)) / 100.0, ehz[ind_start_sei_noise:ind_end_sei_noise], lw=0.5)
    for ax in axs:
        ax.label_outer()
    axs[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, "figure_noise_sei.pdf"))
    plt.close()


plot_aud("Polaris0215pm", "rs1")
