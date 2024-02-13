
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt

import numpy as np

# import torch

from umap import UMAP

import time

import pandas as pd

import plotly.express as px # for data visualization


def chart(X, y):
    # --------------------------------------------------------------------------#
    # This section is not mandatory as its purpose is to sort the data by label
    # so, we can maintain consistent colors for digits across multiple graphs

    # Concatenate X and y arrays
    arr_concat = np.concatenate((X, y.reshape(y.shape[0], 1)), axis=1)
    # Create a Pandas dataframe using the above array
    df = pd.DataFrame(arr_concat, columns=['x', 'y', 'z', 'label'])
    # Convert label data type from float to integer
    df['label'] = df['label'].astype(int)
    # Finally, sort the dataframe by label
    df.sort_values(by='label', axis=0, ascending=True, inplace=True)
    # --------------------------------------------------------------------------#
    # Create a 3D graph
    fig = px.scatter_3d(df, x='x', y='y', z='z', color=df['label'].astype(str), height=900, width=950)

    # Update chart looks
    fig.update_layout(title_text='UMAP',
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                      scene_camera=dict(up=dict(x=0, y=0, z=1),
                                        center=dict(x=0, y=0, z=-0.1),
                                        eye=dict(x=1.5, y=-1.4, z=0.5)),
                      margin=dict(l=0, r=0, b=0, t=0),
                      scene=dict(xaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            ),
                                 yaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            ),
                                 zaxis=dict(backgroundcolor='lightgrey',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            )))
    # Update marker size
    fig.update_traces(marker=dict(size=3, line=dict(color='black', width=0.1)))

    fig.show()


def visual_feas(feas,targets=None,texts=None):
    """
    :param feas: num,dim
    :return:
    """

    tsne = TSNE(n_components=3, learning_rate=1000, perplexity=(len(feas) - 1 )// 3).fit_transform(feas)

    # 使用PCA 进行降维处理
    pca = PCA(n_components=3).fit_transform(feas)

    umapper = UMAP(n_components=3,n_neighbors=15).fit_transform(feas)


    # ploting
    fig = plt.figure(dpi=400)

    # plt.rcParams['font.sans-serif'] = ['Songti SC']

    ax = fig.add_subplot(221, projection='3d')

    ax.scatter(tsne[:,0],tsne[:,1],tsne[:,2],c=targets)

    if texts is not None:
        for i,t in enumerate(texts):
            ax.text(tsne[i,0],tsne[i,1],tsne[i,2],t,fontsize=5.0)

    ax = plt.subplot(222, projection='3d')
    # print(pca)
    plt.scatter(pca[:, 0], pca[:, 1], pca[:, 2],c=targets)

    if texts is not None:
        for j,t in enumerate(texts):
            ax.text(pca[j,0],pca[j,1],pca[j,2],t,fontsize=5.0)

    ax = plt.subplot(223, projection='3d')
    # print(pca)
    plt.scatter(umapper[:, 0], umapper[:, 1], umapper[:, 2],c=targets)

    if texts is not None:
        for k,t in enumerate(texts):
            ax.text(umapper[k,0],umapper[k,1],umapper[k,2],t,fontsize=5.0)

    plt.colorbar()

    plt.savefig('.test.jpg')

    plt.show()


class Cluster(object):

    def __init__(self,method,**kwargs):
        super(Cluster).__init__()

        self.method_dict = {
            'kmeans': KMeans,
            'minikmeans':MiniBatchKMeans,
        }
        self.solver = self.method_dict[method](**kwargs)


    def get_clust_centers(self,feas):

        # print(type(feas))
        # print(feas.shape)
        # print(torch.max(feas),torch.min(feas))

        self.solver.fit(feas)

        cluster_centers = self.solver.cluster_centers_

        return cluster_centers



def init_onelevel_feas_tree(sequence_feas,clust_method,K,lengths,**kwargs):


    current_level = []
    new_lengths = []
    pad_masks=[]
    for b in range(sequence_feas.shape[0]):
        # print('real length:', lengths[b])
        # print('padded length:', sequence_feas.shape[1])
        solver = Cluster(clust_method, n_clusters=lengths[b] // K, **kwargs)
        real_centers=solver.get_clust_centers(feas=sequence_feas[b][:lengths[b]])
        # print('real center shape:', real_centers.shape)
        current_level.append(
            np.concatenate([real_centers,
                            np.zeros((sequence_feas.shape[1]//K-real_centers.shape[0],sequence_feas.shape[-1]))],axis=0),
        )
        new_lengths.append(lengths[b] // K)

        pad_mask = np.concatenate([np.ones(real_centers.shape[0]),
                        np.zeros(sequence_feas.shape[1] // K - real_centers.shape[0])],
                       axis=0)
        pad_masks.append(pad_mask)

        assert pad_mask.sum()==new_lengths[-1]

    # visual_feas(feas=sequence_feas[b][:lengths[b]].detach().numpy(),targets=solver.solver.labels_)

    current_level=np.stack(current_level,axis=0)
    pad_masks=np.stack(pad_masks,axis=0)

    # current_level = np.random.randn(sequence_feas.shape[0],sequence_feas.shape[1] // K,sequence_feas.shape[2])

    return current_level,new_lengths,pad_masks



def get_tree(sequence_feas,lengths, k=4):


    tree = []

    K = k

    while all([leng//K>0 for leng in lengths]):
        # print(sequence_feas.shape)
        sequence_feas,lengths,pad_masks=init_onelevel_feas_tree(sequence_feas=sequence_feas,lengths=lengths,clust_method='minikmeans',K=K)
        # print(sequence_feas.shape)
        tree.append([torch.tensor(sequence_feas),torch.tensor(pad_masks)])
        # print(lengths)

    tree[-1][0] = (torch.sum(tree[-1][0],dim=1)/torch.tensor(lengths).reshape(len(lengths),1)).unsqueeze(dim=1)
    tree[-1][1] = torch.ones(tree[-1][0].shape[:-1])

    return tree


