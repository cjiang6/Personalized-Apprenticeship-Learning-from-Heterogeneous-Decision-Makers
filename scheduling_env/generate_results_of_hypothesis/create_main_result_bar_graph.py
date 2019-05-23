# Credit: Ghost

import numpy as np
import matplotlib.pyplot as plt
import pickle


# load in data

DT_naive_mean = .6075
DT_naive_std = 0.021657658648734104

NN_naive = pickle.load(open('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/naive_saved_models/NN_naive', "rb"))
DDT_naive = pickle.load(open('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/naive_saved_models/DDT', "rb"))
k_means_NN_naive = pickle.load(open('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/naive_saved_models/kmeans_to_NN_naive', "rb"))
DDT_embedding_naive = pickle.load(open('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/naive_saved_models/DDT_w_embedding', "rb"))
PNN_naive = pickle.load(open('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/naive_saved_models/NN_w_embedding', "rb"))



DT_pointwise_mean = 0.124
DT_pointwise_std = 0.07465922581972037 / np.sqrt(50)
NN_pointwise= pickle.load(open('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/pointwise_NN.pkl', "rb"))
DDT_pointwise= pickle.load(open('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/DDT_pointwise150', "rb"))
k_means_NN_pointwise= pickle.load(open('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/pointwise_NN_kmeans.pkl', "rb"))
DDT_embedding_pointwise= pickle.load(open('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/PDDT_pointwise150', "rb"))
PNN_pointwise= pickle.load(open('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/pointwise_NN_unimodal.pkl', "rb"))






DT_pairwise_mean = 0.1105
DT_pairwise_std = 0.09906942010529789 / np.sqrt(50)
NN_pairwise= pickle.load(open('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/NN_pairwise.pkl', "rb"))
DDT_pairwise= pickle.load(open('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/DDT_pairwise150', "rb"))
k_means_NN_pairwise= pickle.load(open('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/pairwise_NN_kmeans.pkl', "rb"))
DDT_embedding_pairwise= pickle.load(open('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/PDDT_pairwise150', "rb"))
PNN_pairwise= pickle.load(open('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/NN_w_embedding_pairwise.pkl', "rb"))





n_groups = 6

means_naive = (100*DT_naive_mean, 100*NN_naive['top1_mean'], 100*DDT_naive['top1_mean'], 100*k_means_NN_naive['top1_mean'], 100*DDT_embedding_naive['top1_mean'],100*PNN_naive['top1_mean'])
std_naive = (100*DT_naive_std, 100*NN_naive['top1_stderr'], 100*DDT_naive['top1_stderr'], 100*k_means_NN_naive['top1_stderr'] ,100*DDT_embedding_naive['top1_stderr'],100*PNN_naive['top1_stderr'])


means_pointwise = (100*DT_pointwise_mean, 100*NN_pointwise['top1_mean'], 100*DDT_pointwise['top1_mean'], 100*k_means_NN_pointwise['top1_mean'], 100*DDT_embedding_pointwise['top1_mean'],100*PNN_pointwise['top1_mean'])
std_pointwise = (100*DT_pointwise_std, 100*NN_pointwise['top1_stderr'], 100*DDT_pointwise['top1_stderr'], 100*k_means_NN_pointwise['top1_stderr'] ,100*DDT_embedding_pointwise['top1_stderr'],100*PNN_pointwise['top1_stderr'])


means_pairwise = (100*DT_pairwise_mean, 100*NN_pairwise['top1_mean'], 100*DDT_pairwise['top1_mean'], 100*k_means_NN_pairwise['top1_mean'], 100*DDT_embedding_pairwise['top1_mean'],100*PNN_pairwise['top1_mean'])
std_pairwise = (100*DT_pairwise_std, 100*NN_pairwise['top1_stderr'], 100*DDT_pairwise['top1_stderr'], 100*k_means_NN_pairwise['top1_stderr'] ,100*DDT_embedding_pairwise['top1_stderr'],100*PNN_pairwise['top1_stderr'])





fig, ax = plt.subplots(figsize=(10,10)) # TODO: remove to get rid of the warping

index = np.arange(n_groups)
bar_width = 0.25

opacity = 0.8
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index , means_naive, bar_width,
                alpha=opacity, color='#FF6475',
                yerr=std_naive, error_kw=error_config,edgecolor='black',
                label='Naive')

rects2 = ax.bar(index + 1.1*bar_width, means_pointwise, bar_width,
                alpha=opacity, color='#2FF77F',
                yerr=std_pointwise, error_kw=error_config, edgecolor='black',
                label='Pointwise')

rects3 = ax.bar(index+ 2.2*bar_width, means_pairwise, bar_width,
                alpha=opacity, color='#2AEDF2',
                yerr=std_pairwise, error_kw=error_config, edgecolor='black',
                label='Pairwise')




ax.set_xlabel('Various Approaches')
ax.set_ylabel('Accuracy')
ax.set_title('Schedule Prediction Accuracy between Various Approaches')
ax.set_xticks(index + bar_width / 2)
ax.yaxis.grid(True)
nets = ['DT', 'NN', 'DDT', 'k-means -> NN', 'PDDT', 'PNN']

ax.set_xticklabels(nets)
ax.legend()

fig.tight_layout()
plt.savefig('scheduling_acc.png')

plt.show()
