import torch
from torch.utils.data import DataLoader

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.cluster import KMeans

from deeptime.util.torch import MLP
from deeptime.util.data import TrajectoryDataset
from deeptime.decomposition.deep import TAE




def tae_discretizer(
    time_ser, num_superfeatures, units=[30, 30, 1], file_name=None, save_path=None,num_cluster=15, tol=0.01
):

    # set_up tae
    dataset = TrajectoryDataset(1, time_ser.astype(np.float32))

    n_val = int(len(dataset) * 0.5)
    train_data, val_data = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val]
    )
    loader_train = DataLoader(train_data, batch_size=64, shuffle=False)
    loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    units = [num_superfeatures] + units
    encoder = MLP(
        units,
        nonlinearity=torch.nn.ReLU,
        output_nonlinearity=torch.nn.Sigmoid,
        initial_batchnorm=False,
    )
    decoder = MLP(units[::-1], nonlinearity=torch.nn.ReLU, initial_batchnorm=False)
    tae = TAE(encoder, decoder, learning_rate=1e-3)
    tae.fit(loader_train, n_epochs=30, validation_loader=loader_val)
    tae_model = tae.fetch_model()
    proj = tae_model.transform(time_ser)
    plot_tae_training(tae_model=tae_model, file_name=file_name, save_path=save_path)
    plot_tae_transform(proj=proj, file_name=file_name, save_path=save_path)
    smooth_projection_k_means(proj=proj, num_cluster=num_cluster, tol=tol)
    #TODO:save the trjacetory
    


def smooth_projection_k_means(proj, file_name, save_path, num_cluster=15, tol=0.01):
    """Cluster the projection to get realy discretized values necessary for the MSM 

    Args:
        proj ([type]): [description]
        num_cluster (int, optional): [description]. Defaults to 15.
        tol (float, optional): [description]. Defaults to 0.01.
    """

    random_state = 42
    scores = np.zeros(num_cluster)
    sum_of_squared_distances = np.zeros(num_cluster)
    for i in range(1, num_cluster):
        clf = KMeans(n_clusters=i, random_state=random_state).fit(proj)
        scores[i] = clf.score(proj)
        sum_of_squared_distances[i] = clf.inertia_
    # clf = KMeans(n_clusters=3, random_state=random_state).fit(proj)
    # labels = clf.labels_
    plot_ellbow_kmeans(metric=sum_of_squared_distances, file_name=file_name, save_path=save_path)
    plot_scores_kmeans(metric=scores, file_name=file_name, save_path=save_path)
    return [scores, sum_of_squared_distances]


def plot_tae_training(tae_model, file_name=None, save_path=None):
    """Plots the loss function for the trainig of the TAE model.

    Args:
        tae_model ([type]): [description]
        file_name ([type], optional): [description]. Defaults to None.
        save_path ([type], optional): [description]. Defaults to None.
    """
    plt.semilogy(*tae_model.train_losses.T, label="train")
    plt.semilogy(*tae_model.validation_losses.T, label="validation")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path + "/" + file_name + "tae_model.png", dpi=300)


def plot_tae_transform(proj, num_steps=1000, file_name=None, save_path=None):
    """Plots the transformation obtained by the TAE model.

    Args:
        proj ([type]): [description]
        num_steps (int, optional): [description]. Defaults to 1000.
        file_name ([type], optional): [description]. Defaults to None.
        save_path ([type], optional): [description]. Defaults to None.
    """
    plt.ylabel("State")
    plt.xlabel("Timte $t$")
    if num_steps < len(proj):
        plt.plot(proj[:num_steps])
    else:
        plt.plot(proj[:num_steps])
    plt.legend()
    plt.savefig(save_path + "/" + file_name + "tae_transform.png", dpi=300)

def plot_scores_kmeans(sum_of_squared_distances, file_name=None, save_path=None): 
    """Plots the scores of the k_means finder

    Args:
        sum_of_squared_distances ([type]): [description]
        file_name ([type], optional): [description]. Defaults to None.
        save_path ([type], optional): [description]. Defaults to None.

    Returns:
        None
    """
    plt.xlabel("Number of cluster")
    plt.ylabel("Euclidean Norm $l^2$")
    plt.savefig(save_path + "/" + file_name + "scores_kMeans.png", dpi=300)
    plt.plot(sum_of_squared_distances[1:])
    return None
    

def plot_ellbow_kmeans(metric, file_name=None, save_path=None):
    """Plots the sum of squared distances for K-Means to do the ellbow method visually
    

    Args:
        sum_of_squared_distances ([type]): [description]
        file_name ([type], optional): [description]. Defaults to None.
        save_path ([type], optional): [description]. Defaults to None.

    Returns:
        None
    """
    plt.xlabel("Number of cluster")
    plt.ylabel("Sum of squared distances $R$")
    plt.savefig(save_path + "/" + file_name + "ellbow_kMeans.png", dpi=300)
    plt.plot(sum_of_squared_distances[1:])
    return None


