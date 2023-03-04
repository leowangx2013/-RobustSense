import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results(runs, results, title, save_path):
    runs = runs[::2]
    runs = [run.split("_")[0] for run in runs]

    results_aug = results[::2]
    results_noaug = results[1::2]

    barWidth = 0.25
    x_aug = np.arange(len(runs))
    x_noaug = [x + barWidth for x in x_aug]

    plt.figure(figsize=(10, 5))
    plt.bar(x_aug, results_aug, color ='darkorange', width = barWidth,
        edgecolor ='grey', label ='aug')
    plt.bar(x_noaug, results_noaug, color ='darkcyan', width = barWidth,
            edgecolor ='grey', label ='no_aug')
    
    plt.xlabel('Masked condition', fontweight ='bold', fontsize = 15)
    plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(x_aug))],
            runs)
    plt.title(title)
    plt.legend()

    plt.savefig(os.path.join(save_path, title+".png"))
