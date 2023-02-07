import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

mus = [0, 1]
vars = [1, 2]
plt.figure(figsize=(3, 1.5))
for mu, var in zip(mus, vars):
    sigma = math.sqrt(var)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), lw=5)
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig("/home/sl29/AutoCuration/result/figures/sample_gaussian_distribution.pdf")
# plt.show()
