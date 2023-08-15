import pickle
import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import time
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.nn import GCNConv

from model import Encoder, Model, drop_feature
from eval import label_classification
from data_utils import loadAllData
from model import Model
import numpy as np
import networkx as nx
from torch_sparse import coalesce
import torch
import matplotlib.pyplot as plt

from data_utils import loadAllData
from sklearn import manifold, datasets

from torch_geometric.nn import GCNConv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Polblogs')
parser.add_argument('--gpu_id', '-g', type=int, default=0)
parser.add_argument('--config', type=str, default='./config.yaml')
parser.add_argument('--is_node_weight', '-n', action='store_true')
parser.add_argument('--is_edge_weight', '-w', action='store_true')
parser.add_argument('--is_adaptive', '-a', action='store_true')
parser.add_argument('--is_motif', '-m', action='store_true')
args = parser.parse_args()
assert args.gpu_id in range(0, 8)
torch.cuda.set_device(args.gpu_id)
is_node_weight = args.is_node_weight
is_edge_weight = args.is_edge_weight
is_adaptive = args.is_adaptive
is_motif = args.is_motif
tau=0.4
# for datasets in [["acmv9", "Football"],["dblpv7", "Football"],["citationv1", "Football"],["amazon-photo", "Football"],["amazon-computers", "Football"]]:
for datasets in [["citationv1", "Football"]]:
    config = yaml.load(open(args.config), Loader=SafeLoader)[datasets[1]]
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    learning_rate = config['learning_rate']
    # learning_rate = lr
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']
    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    # tau = config['tau']
    weight_decay = config['weight_decay']
    lower = config['lower']
    upper = config['upper']
    node_lower = config['node_lower']
    node_upper = config['node_upper']
    str1 = "_weight" if is_edge_weight else ""
    str2 = "_adaptive" if is_adaptive else ""
    str3 = "_node" if is_node_weight else ""
    str4 = "_motif" if is_motif else ""
    res_path = "./results/" + datasets[0] + "_marcia_shuffle_tau_" + str(tau) + ".txt"
    with open(res_path, 'a+') as res_file:
        res_file.write(str(config) + "\n")
    for _ in range(20):
        allx, ally, edges, edges_weight, motifs_all, motifs_num = loadAllData(datasets[0])
        allylabel = []
        for item in ally:
            allylabel.append(np.argmax(item))
        edges_index = torch.tensor(edges.astype(np.int64)).T
        edges_weight = torch.tensor(edges_weight).T
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        allx = torch.tensor(allx.A, dtype=torch.float32).to(device)
        ally = torch.tensor(allylabel).to(device)
        edges_index = edges_index.to(device)
        edges_weight = edges_weight.to(device)
        motifs_all = motifs_all.to(device)
        motifs_num = motifs_num.to(device)
        if is_motif:

            normalized_motifs_num = F.normalize(motifs_num, p=2, dim=1)

            allx = torch.cat([allx, normalized_motifs_num.T], dim=1)
        # 计算节点特征mask概率
        node_weight = motifs_num.sum(dim=0)
        a = node_lower
        b = node_upper
        k = (b - a) / (torch.max(node_weight) - torch.min(node_weight))
        node_probability = a + k * (node_weight - torch.min(node_weight))

        encoder = Encoder(allx.shape[1], num_hidden, activation,
                          base_model=base_model, k=num_layers, device=device).to(device)


        model = Model(encoder, num_hidden, num_proj_hidden, tau)

        model.load_state_dict(torch.load('model_0.4_citationv1.pkl'))

        #allx, ally, edges, edges_weight, motifs_all, motifs_num = loadAllData('citationv1')
        z = model(allx, edges_index, edges_weight, motifs_all, motifs_num)
        tsne = manifold.TSNE(n_components=3,  random_state=200,perplexity=55,learning_rate=300,init='pca').fit_transform(z.cpu().detach().numpy())
        #z.cpu().detach().numpy() #将tensor.gpu -> tensor.cpu ->numpy
        X_tsne = tsne
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)
        plt.figure(figsize=(18, 18))
        for i in range(X_norm.shape[0]):
            plt.scatter(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set1(ally.cpu().detach().numpy()[i]))
        plt.show()
        #tsne = manifold.TSNE(n_components=2, random_state=33).fit_transform(z.cpu().detach().numpy())
        #z = model(x, edge_index, edge_weight, motifs_all, motifs_num)
        if _ == 0:
            break
# print(z)


import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import manifold, datasets
#
# '''X是特征，不包含target;X_tsne是已经降维之后的特征'''
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
# X_tsne = tsne.fit_transform(X)
# print("Org data dimension is {}.Embedded datadimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
#
# '''嵌入空间可视化'''
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
# plt.figure(figsize=(8, 8))
# for i in range(X_norm.shape[0]):
#     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
#              fontdict={'weight': 'bold', 'size': 9})
# plt.xticks([])
# plt.yticks([])
# plt.show()


#三维的代码
# fig = plt.figure(figsize=(18, 18))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_norm[:,0], X_norm[:,1],X_norm[:,2] ,color=plt.cm.Set1(ally.cpu().detach().numpy()))
# plt.show()