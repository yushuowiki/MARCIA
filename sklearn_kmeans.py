from sklearn.cluster import KMeans
#from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment

import numpy as np
from sklearn import metrics
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
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

from train import train_test

np.random.seed(19680801)

#model.load_state_dict(torch.load('model_0.4_citationv1.pkl'))

# allx, ally, edges, edges_weight, motifs_all, motifs_num = loadAllData('citationv1')

allx, ally, edges, edges_weight, motifs_all, motifs_num = loadAllData('amazon-computers')
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


model = torch.load('./model_0.4_amazon-computers.pkl')

embedding = model(allx, edges_index, edges_weight, motifs_all, motifs_num).detach().cpu().numpy()

labels=ally.cpu().numpy()
#tsne = manifold.TSNE(n_components=3, init='pca', random_state=501,n_iter=3000000,perplexity=50)


y_pred_labels = KMeans(n_clusters=10, random_state=23).fit_predict(embedding)
def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    total = 0
    for i in range(len(ind[0])):
        total += w[ind[0][i], ind[1][i]]
    return total * 1.0 / Y_pred.size, w



print(cluster_acc(labels,y_pred_labels))