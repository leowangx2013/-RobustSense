import json
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
from matplotlib.pyplot import figure

sns.set(font_scale=1.2)
sns.set_style("whitegrid")
width = 0.35  # the width of the bars: can also be len(x) sequence
labels = ["DeepSense", "ResNet", "Transformer"]
out_dir = "/home/sl29/AutoCuration/result/figures/"
dataset = "RealWorld_HAR"
x = np.arange(len(labels))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 3), dpi=80, sharey=True)
density_means = np.array([0.80822, 0.99651, 0.52494])
recon_means = np.array([0.56618, 0.53823, 0.57028])
vae_means = np.array([0.96330, 0.96040, 0.98780])

ax1.bar(x - 1 / 2 * width, density_means, width / 2, label="Density")
ax1.bar(x, recon_means, width / 2, label="Reconstruction")
ax1.bar(x + 1 / 2 * width, vae_means, width / 2, label="VAE")

ax1.set_ylabel("AUC Score")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
# ax1.set_title('Base Noise')

density_means = np.array([0.87807, 0.99986, 0.54781])
recon_means = np.array([0.59097, 0.54010, 0.63300])
vae_means = np.array([0.99644, 0.99639, 0.99797])

ax2.bar(x - 1 / 2 * width, density_means, width / 2, label="Density")
ax2.bar(x, recon_means, width / 2, label="Reconstruction")
ax2.bar(x + 1 / 2 * width, vae_means, width / 2, label="VAE")

ax2.set_ylabel("AUC Score")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)

density_means = np.array([0.95068, 1.00000, 0.54055])
recon_means = np.array([0.63543, 0.52748, 0.71320])
vae_means = np.array([1, 1, 0.99978])

ax3.bar(x - 1 / 2 * width, density_means, width / 2, label="Density")
ax3.bar(x, recon_means, width / 2, label="Reconstruction")
ax3.bar(x + 1 / 2 * width, vae_means, width / 2, label="VAE")

ax3.set_ylabel("AUC Score")
ax3.set_xticks(x)
ax3.set_xticklabels(labels)

handles, labels = ax3.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
plt.savefig(os.path.join(out_dir, "%s_detection.pdf" % (dataset)), bbox_inches="tight")
# plt.savefig("%s_detection.pdf" %(dataset), bbox_inches = 'tight')
plt.show()

sns.set(font_scale=1.2)
sns.set_style("whitegrid")
width = 0.35  # the width of the bars: can also be len(x) sequence
labels = ["DeepSense", "ResNet", "Transformer"]
out_dir = "/home/sl29/AutoCuration/result/figures/"
dataset = "Parkland"

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 3), dpi=80, sharey=True)
density_means = np.array([0.78783, 1.00000, 0.57411])
recon_means = np.array([0.50335, 0.52069, 0.61498])
vae_means = np.array([0.89166, 0.95270, 0.99574])

ax1.bar(x - 1 / 2 * width, density_means, width / 2, label="Density")
ax1.bar(x, recon_means, width / 2, label="Reconstruction")
ax1.bar(x + 1 / 2 * width, vae_means, width / 2, label="VAE")

ax1.set_ylabel("AUC Score")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
# ax1.set_title('Base Noise')

density_means = np.array([0.94932, 1.00000, 0.67502])
recon_means = np.array([0.56574, 0.47673, 0.61813])
vae_means = np.array([0.98792, 0.99906, 1.00000])

ax2.bar(x - 1 / 2 * width, density_means, width / 2, label="Density")
ax2.bar(x, recon_means, width / 2, label="Reconstruction")
ax2.bar(x + 1 / 2 * width, vae_means, width / 2, label="VAE")

ax2.set_ylabel("AUC Score")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)

density_means = np.array([0.99818, 1, 0.71920])
recon_means = np.array([0.62445, 0.51399, 0.60198])
vae_means = np.array([1, 1, 1])

ax3.bar(x - 1 / 2 * width, density_means, width / 2, label="Density")
ax3.bar(x, recon_means, width / 2, label="Reconstruction")
ax3.bar(x + 1 / 2 * width, vae_means, width / 2, label="VAE")

ax3.set_ylabel("AUC Score")
ax3.set_xticks(x)
ax3.set_xticklabels(labels)

handles, labels = ax3.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
plt.savefig(os.path.join(out_dir, "%s_detection.pdf" % (dataset)), bbox_inches="tight")
# plt.savefig("%s_detection.pdf" %(dataset), bbox_inches = 'tight')
plt.show()

sns.set(font_scale=1.2)
sns.set_style("whitegrid")
width = 0.35  # the width of the bars: can also be len(x) sequence
labels = ["DeepSense", "ResNet", "Transformer"]
out_dir = "/home/sl29/AutoCuration/result/figures/"
dataset = "WESAD"

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 3), dpi=80, sharey=True)

density_means = np.array([0.63658, 0.80564, 0.52494])
recon_means = np.array([0.66157, 0.56520, 0.56205])
vae_means = np.array([0.98242, 0.96933, 0.95432])

ax1.bar(x - 1 / 2 * width, density_means, width / 2, label="Density")
ax1.bar(x, recon_means, width / 2, label="Reconstruction")
ax1.bar(x + 1 / 2 * width, vae_means, width / 2, label="VAE")

ax1.set_ylabel("AUC Score")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
# ax1.set_title('Base Noise')

density_means = np.array([0.62437, 0.87927, 0.51263])
recon_means = np.array([0.70504, 0.51162, 0.60247])
vae_means = np.array([0.99952, 0.99650, 1.00000])

ax2.bar(x - 1 / 2 * width, density_means, width / 2, label="Density")
ax2.bar(x, recon_means, width / 2, label="Reconstruction")
ax2.bar(x + 1 / 2 * width, vae_means, width / 2, label="VAE")

ax2.set_ylabel("AUC Score")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)

density_means = np.array([0.65008, 0.88537, 0.50780])
recon_means = np.array([0.69913, 0.47721, 0.67189])
vae_means = np.array([1, 1, 1])

ax3.bar(x - 1 / 2 * width, density_means, width / 2, label="Density")
ax3.bar(x, recon_means, width / 2, label="Reconstruction")
ax3.bar(x + 1 / 2 * width, vae_means, width / 2, label="VAE")

ax3.set_ylabel("AUC Score")
ax3.set_xticks(x)
ax3.set_xticklabels(labels)

handles, labels = ax3.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
plt.savefig(os.path.join(out_dir, "%s_detection.pdf" % (dataset)), bbox_inches="tight")
# plt.savefig("%s_detection.pdf" %(dataset), bbox_inches = 'tight')
plt.show()
