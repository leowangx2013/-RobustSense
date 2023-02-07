import json
import os
import matplotlib
import brewer2mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({"font.size": 24})
matplotlib.rcParams["hatch.linewidth"] = 1
markers = ["x", "o", "*", "^"]
colors = brewer2mpl.get_map("Set1", "qualitative", 8).mpl_colors

log_root_path = "/home/sl29/AutoCuration/result/log"
figure_path = "/home/sl29/AutoCuration/result/figures"

dataset_mods = {
    "Parkland": ["Aud", "Seis"],
    "RealWorld_HAR": ["Acc", "Gyr", "Mag", "Lig"],
    "WESAD": ["EMG", "EDA", "Resp"],
}


def plot_scatter_3d(dataset, model, detector, noise_std):
    """Plot the scatter for the given log file."""

    log_file = os.path.join(
        log_root_path,
        f"{dataset}_{model}_noisy",
        f"{detector}_noisy_NoiseStd-{noise_std}_embs.json",
    )

    if not os.path.exists(log_file):
        return

    with open(log_file, "r") as f:
        logs = json.load(f)

    # [b, s]
    miss_masks = np.array(logs["miss_masks"])
    s = miss_masks.shape[1]

    # [b, s, 2]
    embs = np.array(logs["embs"])
    mods = dataset_mods[dataset]

    if embs.shape[1] < 3:
        return

    # plot
    plt.figure(figsize=(11, 6))
    ax = plt.axes(projection="3d")
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    ax.set_ylabel("Dim 2")
    ax.set_xlabel("Dim 1")
    ax.set_zlabel("Dim 3")

    for i in range(s):
        mod_masks = miss_masks[:, i]
        mod_embs = embs[:, i]
        ax.scatter(
            mod_embs[:, 0],
            mod_embs[:, 1],
            mod_embs[:, 2],
            c=[colors[int(e)] for e in mod_masks],
            marker=markers[i],
            label=mods[i],
            alpha=0.3,
        )

    # set the legend for both (color, marker)
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = []
    labels = []
    for i, marker in enumerate(markers[: len(mods)]):
        handles += [f(marker, colors[i]) for i in range(2)]
        labels += [f"Noisy {mods[i]}", f"Clean {mods[i]}"]
    plt.legend(handles, labels, ncol=len(mods))

    # save
    figure_file = os.path.join(figure_path, f"{dataset}_{model}_{detector}_std-{noise_std}_embs_3d.pdf")
    plt.savefig(figure_file, bbox_inches="tight")


def plot_scatter_2d(dataset, model, detector, noise_std):
    """Plot the scatter for the given log file."""

    log_file = os.path.join(
        log_root_path,
        f"{dataset}_{model}_noisy",
        f"{detector}_noisy_NoiseStd-{noise_std}_embs.json",
    )

    if not os.path.exists(log_file):
        return
    else:
        print(log_file)

    with open(log_file, "r") as f:
        logs = json.load(f)

    # [b, s]
    miss_masks = np.array(logs["miss_masks"])
    s = miss_masks.shape[1]

    # [b, s, 2]
    embs = np.array(logs["embs"])
    mods = dataset_mods[dataset]
    if embs.shape[2] != 2:
        return

    # plot
    plt.figure(figsize=(11, 7))
    ax = plt.axes()
    sns.set(font_scale=2)
    sns.set_style("whitegrid")
    ax.set_ylabel("Dim 2")
    ax.set_xlabel("Dim 1")

    for i in range(s):
        mod_masks = miss_masks[:, i]
        mod_embs = embs[:, i]
        ax.scatter(
            mod_embs[:, 0],
            mod_embs[:, 1],
            c=[colors[int(e)] for e in mod_masks],
            marker=markers[i],
            label=mods[i],
            alpha=0.9,
        )

    # set the legend for both (color, marker)
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = []
    labels = []
    for i, marker in enumerate(markers[: len(mods)]):
        handles += [f(marker, colors[i]) for i in range(2)]
        labels += [f"Noisy {mods[i]}", f"Clean {mods[i]}"]
    plt.legend(handles, labels, ncol=2)

    # save
    figure_file = os.path.join(figure_path, f"{dataset}_{model}_{detector}_std-{noise_std}_embs.pdf")
    plt.savefig(figure_file, bbox_inches="tight")


# for dataset in ["WESAD", "Parkland", "RealWorld_HAR"]:
#     for model in ["DeepSense", "Transformer", "ResNet"]:
#         for detector in ["VAEPlusDetector"]:
#             for noise_std in [1.0, 2.0, 3.0, 5.0, 10.0]:
#                 plot_scatter_2d(dataset, model, detector, noise_std)


for dataset in ["WESAD", "Parkland", "RealWorld_HAR"]:
    for model in ["Transformer"]:
        for detector in ["ReconstructionDetector"]:
            for noise_std in [2.0]:
                plot_scatter_2d(dataset, model, detector, noise_std)
