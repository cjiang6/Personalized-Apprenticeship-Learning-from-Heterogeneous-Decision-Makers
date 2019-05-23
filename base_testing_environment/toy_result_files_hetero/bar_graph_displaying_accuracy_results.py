import matplotlib.pyplot as plt
import numpy as np
import pickle

nets = ['Decision Tree', 'DT  w/ Bimodal Embedding', 'Neural Network', 'Differentiable Decision Tree', 'k-means -> NN', 'GMM -> NN', 'PDDT', 'PNN']
nets = ['DT', 'DT w/ \n Bimodal Embedding', 'NN', 'DDT', 'k-means -> NN', 'GMM -> NN', 'PDDT', 'PNN']
x_pos = np.arange(len(nets))

top1_means = [0.5696789333744428, 0.544, 0.5629047899481815,0.5605012467866215,0.5581463904841616,0.5695957177705059, 0.8913488728021303, 0.9564700179444091]
top1_stds = [0.029806861532677694, 0.13734627770711516, 0.025337123380513187,0.023605629775417268,0.024039313170542403, 0.02664792103459427, 0.016397252472774945, 0.011069895656443776]
# TODO: divide std by sqrt  (50

# Build the plot
fig, ax = plt.subplots(figsize=(10,10))
ax.bar(x_pos, top1_means, yerr=top1_stds, align='center', alpha=1.0, ecolor='black', capsize=10, color='#2AEDF2', edgecolor='black')
ax.set_ylabel('Prediction Accuracy')
ax.set_xticks(x_pos)
ax.set_xticklabels(nets)
# ax.set_title('Apprenticeship Learning Performance')
ax.yaxis.grid(True)

# plt.tight_layout()
plt.savefig('toy_env_results_acc.png')
plt.show()

