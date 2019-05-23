import matplotlib.pyplot as plt
import numpy as np
import pickle

nets = ['NN: Naive', 'NN: Pairwise', 'PDDT: Pairwise', 'PNN: Pairwise']
x_pos = np.arange(len(nets))

top1_means = [0.11255920947261658, 0.10479651010472664, 0.17348568527406807,0.09146019186676302]
top1_stds = [0.003708032550995642/np.sqrt(15), 0.029112924804902227/np.sqrt(25), 0.01,0.024514554902807083/np.sqrt(15)]
# TODO: divide std by sqrt  (50

# Build the plot
fig, ax = plt.subplots(figsize=(10,7))
ax.bar(x_pos, top1_means, yerr=top1_stds, align='center', alpha=0.8, ecolor='black', capsize=10, color='#2AEDF2')
ax.set_ylabel('Loss')
ax.set_xticks(x_pos)
ax.set_xticklabels(nets)
# ax.set_title('Apprenticeship Learning Performance')
ax.yaxis.grid(True)

# plt.tight_layout()
plt.savefig('starcraft_accuracy.png')
plt.show()

