import json
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
from matplotlib.pyplot import figure

def plot_roc_curve(file_dir, out_dir, std):
    figure(figsize=(6, 3), dpi=80)
    files = os.listdir(file_dir)
    model_name = [name[item[:item.find('_')]] for item in files]

    plot_data = {"VAE": None, "Reconstruction": None, "Density": None}
    setting = "ROC Curve\n" + files[0][files[0].find('_') + 1:].split('.')[0]
    sns.set(font_scale = 1.5)
    sns.set_style("whitegrid")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    for i, file in enumerate(files):
        if std not in file:
            continue
        with open(os.path.join(file_dir,file), 'r') as f:
            data = json.load(f)
        if "VAE" in file:
            plot_data["VAE"] = data
        elif "Reconstruction" in file:
            plot_data["Reconstruction"] = data
        elif "Density" in file:
            plot_data["Density"] = data
    for key in plot_data:
        plt.plot(plot_data[key]['fpr'], plot_data[key]['tpr'], label = key, linewidth=weight)

    plt.legend(loc=4)
    plt.savefig(os.path.join(out_dir,"%s_std_%s_detection.pdf" %(model,std)), bbox_inches = 'tight')
    # plt.savefig("%s_std_%s_detection.pdf" %(model,std), bbox_inches = 'tight')
    plt.show()
    return data

weight = 2.5
model = "DeepSense"
file_dir = "/home/sl29/AutoCuration/result/log/RealWorld_HAR_DeepSense_noisy/"
out_dir = "/home/sl29/AutoCuration/result/figures/"
name = {'VAEPlusDetector': "VAE", "ReconstructionDetector": "Reconstruction", "DensityDetector": "Density"}

data = plot_roc_curve(file_dir, out_dir, std = "1.0")
data = plot_roc_curve(file_dir, out_dir, std = "3.0")
data = plot_roc_curve(file_dir, out_dir, std = "10.0")

model = "Transformer"
file_dir = "/home/sl29/AutoCuration/result/log/RealWorld_HAR_Transformer_noisy/"
out_dir = "/home/sl29/AutoCuration/result/figures/"
name = {'VAEPlusDetector': "VAE", "ReconstructionDetector": "Reconstruction", "DensityDetector": "Density"}

data = plot_roc_curve(file_dir, out_dir, std = "1.0")
data = plot_roc_curve(file_dir, out_dir, std = "3.0")
data = plot_roc_curve(file_dir, out_dir, std = "10.0")

model = "ResNet"
file_dir = "/home/sl29/AutoCuration/result/log/RealWorld_HAR_ResNet_noisy/"
out_dir = "/home/sl29/AutoCuration/result/figures/"
name = {'VAEPlusDetector': "VAE", "ReconstructionDetector": "Reconstruction", "DensityDetector": "Density"}

data = plot_roc_curve(file_dir, out_dir, std = "1.0")
data = plot_roc_curve(file_dir, out_dir, std = "3.0")
data = plot_roc_curve(file_dir, out_dir, std = "10.0")

