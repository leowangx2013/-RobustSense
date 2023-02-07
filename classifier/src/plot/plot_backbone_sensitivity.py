import os
import numpy as np
from matplotlib import pyplot as plt
import brewer2mpl
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

bmap = brewer2mpl.get_map("Set1", "qualitative", 8)
colors = bmap.mpl_colors
matplotlib.rcParams.update({"font.size": 22})

figure_path = "/home/sl29/AutoCuration/result/figures"


def plotMultiLine(
    x_list,
    y_list,
    x_label,
    y_label,
    min_y,
    max_y,
    labels,
    fileName=None,
    ytick=20,
    color_list=colors,
    hline=None,
):
    """
    Plot several lines in one figure.
    :return:
    """
    f = plt.figure(figsize=(9, 6))
    ax = f.add_subplot(111)

    # Plot lines
    l1 = plt.plot(x_list, y_list[0], "o-", label=labels[0], linewidth=3, markersize=7, color=color_list[0])
    l2 = plt.plot(x_list, y_list[1], "s-", label=labels[1], linewidth=3, markersize=7, color=color_list[1])
    l3 = plt.plot(x_list, y_list[2], "*-", label=labels[2], linewidth=3, markersize=7, color=color_list[2])

    l4 = plt.plot(x_list, y_list[3], "o--", label=labels[3], linewidth=3, markersize=7, color=color_list[0])
    l5 = plt.plot(x_list, y_list[4], "s--", label=labels[4], linewidth=3, markersize=7, color=color_list[1])
    l6 = plt.plot(x_list, y_list[5], "*--", label=labels[5], linewidth=3, markersize=7, color=color_list[2])

    ax.legend(ncol=2, loc="best")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Set ticks
    ax.set_yticks(np.arange(min_y, max_y + 0.01, ytick))
    ax.set_xticks(np.arange(5, np.max(x_list) + 1, 5))

    # Set guidlines
    ax.grid()

    # adjust layout
    plt.tight_layout()

    # Show or save the figure
    plt.savefig(fileName)


x_list = [0, 1, 2, 3, 5, 7, 10, 15]
x_label = "Noise Std Multiplier"
y_label = "F1 Score"
labels = ["DS", "RN", "Trans", "DS-Avl", "RN-Avl", "Tran-Avl"]

# RealWorld-HAR
min_y = 0
max_y = 1
y_tick = 0.2
y_list = [
    [0.9055, 0.82120, 0.74606, 0.67292, 0.48203, 0.37925, 0.28434, 0.22358],
    [0.9125, 0.85050, 0.68227, 0.55115, 0.40057, 0.29522, 0.22728, 0.18100],
    [0.7168, 0.70501, 0.68752, 0.63687, 0.61935, 0.59421, 0.57743, 0.52964],
    [0.66388] * 8,
    [0.74759] * 8,
    [0.64241] * 8,
]
file_name = os.path.join(figure_path, "RealWorld_HAR_backbone_sentivity.pdf")
plotMultiLine(x_list, y_list, x_label, y_label, min_y, max_y, labels, fileName=file_name, ytick=y_tick)

# Parkland
min_y = 0
max_y = 1
y_tick = 0.2
y_list = [
    [0.87433, 0.79271, 0.65543, 0.53238, 0.37224, 0.25234, 0.19090, 0.15464],
    [0.87387, 0.84676, 0.79548, 0.73277, 0.58419, 0.49196, 0.37259, 0.27206],
    [0.56523, 0.32084, 0.25661, 0.22037, 0.19989, 0.19671, 0.18475, 0.18334],
    [0.54722] * 8,
    [0.52043] * 8,
    [0.31527] * 8,
]
file_name = os.path.join(figure_path, "Parkland_backbone_sentivity.pdf")
plotMultiLine(x_list, y_list, x_label, y_label, min_y, max_y, labels, fileName=file_name, ytick=y_tick)

# WESAD
min_y = 0.2
max_y = 0.8
y_tick = 0.15
y_list = [
    [0.73016, 0.70362, 0.62732, 0.54683, 0.41046, 0.39301, 0.39301, 0.39301],
    [0.73721, 0.63174, 0.52614, 0.49754, 0.47663, 0.46612, 0.47340, 0.47051],
    [0.77188, 0.68437, 0.64917, 0.64917, 0.64734, 0.62832, 0.60658, 0.60094],
    [0.61661] * 8,
    [0.63045] * 8,
    [0.69133] * 8,
]
file_name = os.path.join(figure_path, "WESAD_HAR_backbone_sentivity.pdf")
plotMultiLine(x_list, y_list, x_label, y_label, min_y, max_y, labels, fileName=file_name, ytick=y_tick)
