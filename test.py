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
from eval import label_classification, prob_to_one_hot
from data_utils import loadAllData
from model import Model
import numpy as np
import networkx as nx
from torch_sparse import coalesce
# Fixing random state for reproducibility
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

from train import train_test

np.random.seed(19680801)

#model.load_state_dict(torch.load('model_0.4_citationv1.pkl'))

# allx, ally, edges, edges_weight, motifs_all, motifs_num = loadAllData('citationv1')

allx, ally, edges, edges_weight, motifs_all, motifs_num = loadAllData('citationv1')
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


model = torch.load('./model_0.4_citationv1.pkl')
#z = model(x, edge_index, edge_weight, motifs_all, motifs_num)

embedding = model(allx, edges_index, edges_weight, motifs_all, motifs_num).detach().cpu().numpy()
#res = train_test(model, allx, motifs_all, motifs_num, edges_index, edges_weight, ally, True, True,final=True)
#pe:200, learning_rate:10 early_exaggeration=12 ,angle=0.2,
#,perplexity=100,learning_rate=10, n_iter=100000000




# tsne = manifold.TSNE(n_components=2,random_state=501,init='pca',learning_rate=11,perplexity=500).fit_transform(z.cpu().detach().numpy())
# # z.cpu().detach().numpy() #将tensor.gpu -> tensor.cpu ->numpy
# X_tsne = tsne
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)
# plt.figure(figsize=(18, 18))
# for i in range(X_norm.shape[0]):
#     plt.scatter(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set1(ally.cpu().detach().numpy()[i]))
# plt.show()


#
# fig = plt.figure(figsize=(30, 30))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_norm[:,0], X_norm[:,1],X_norm[:,2] ,color=plt.cm.Set1(ally.cpu().detach().numpy()))
# plt.show()



# ratio=0.6
# embeddings = z
# y=ally
# X = embeddings.detach().cpu().numpy()
# Y = y.detach().cpu().numpy()
# Y = Y.reshape(-1, 1)
# onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
# Y = onehot_encoder.transform(Y).toarray().astype(np.bool)
#
# X = normalize(X, norm='l2')
#
# X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=1 - ratio)
#
# logreg = LogisticRegression(solver='liblinear')
# c = 2.0 ** np.arange(-10, 10)
#
# clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
#                    param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
#                    verbose=0)
# clf.fit(X_train, y_train)
#
# y_pred = clf.predict_proba(X_test)
# y_pred = prob_to_one_hot(y_pred)
#
# micro = f1_score(y_test, y_pred, average="micro")
# macro = f1_score(y_test, y_pred, average="macro")
# acc = accuracy_score(y_test, y_pred)
labels=ally.cpu().numpy()
tsne = manifold.TSNE(n_components=3, init='pca', random_state=501,n_iter=3000000,perplexity=50)
X_tsne = tsne.fit_transform(embedding)
'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
model = TSNE(n_components=2, init="pca",perplexity=50,n_iter=3000000)
# model = TSNE(n_components=2)
node_pos = model.fit_transform(X_norm)
color_idx = {}
for i in range(X_norm.shape[0]):
    color_idx.setdefault(labels[i], [])
    color_idx[labels[i]].append(i)

for c, idx in color_idx.items():
    plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, s=2)  # c=node_colors)
# plt.axis('off')
# plt.legend()
plt.xticks([])
plt.yticks([])
plt.tick_params(top=False, bottom=False, left=False, right=False)
plt.gca.legend_ = None
plt.savefig('MARCIA' + '.png')
plt.show()
print('done!')