import matplotlib.pyplot as plt
import umap.umap_ as umap
import torch
import geoopt.manifolds.stereographic.math as gmath
import numpy as np
from scipy.spatial import distance


def main():

    # Get data
    # data_path = '/proj/vondrick/shared/DPC/results/pred_hyp.pth'
    data_path = '/proj/vondrick/didac/results/pred.pth'
    data_poincare = torch.load(data_path, map_location=torch.device('cpu'))
    plot_poincare(data_poincare, path_save='/proj/vondrick/didac/results/plot_poincare.png', show=True)
    plot_histogram(data_poincare)


def plot_poincare(data_poincare, path_save, show=False):
    # Project from Poincare disk to hyperboloid
    k = torch.tensor(-1.)
    data_hyperboloid = gmath.inv_sproj(data_poincare, k=k)
    data_hyperboloid = data_hyperboloid.data.cpu().numpy()

    # Reduce dimensionality
    hyperbolic_mapper = umap.UMAP(output_metric='hyperboloid', random_state=1).fit(data_hyperboloid)
    data_hyperboloid_reduced = hyperbolic_mapper.embedding_
    z = np.sqrt(1 + np.sum(data_hyperboloid_reduced ** 2, axis=1))
    data_hyperboloid_reduced = np.concatenate([data_hyperboloid_reduced, z[:, None]], axis=-1)

    # Project to Poincare
    data_poincare_reduced = gmath.sproj(torch.tensor(data_hyperboloid_reduced), k=k).numpy()

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_poincare_reduced[:, 0], data_poincare_reduced[:, 1], cmap='Spectral')  # c= would be color
    boundary = plt.Circle((0, 0), 1, fc='none', ec='k')
    plt.gca().set_xlim(-1.1, 1.1)
    plt.gca().set_ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.add_artist(boundary)
    ax.axis('off')
    plt.savefig(path_save)

    if show:
        plt.show()


def plot_histogram(data):

    data = data.data

    plt.figure(figsize=(6.4, 15))

    plt.subplot(6, 1, 1)
    # Euclidean distance
    euc_distance = data.norm(dim=-1)
    plt.hist(euc_distance, bins=20)

    plt.subplot(6, 1, 2)
    # Compute distances to origin (real hyperbolic distance)
    hyp_distance = gmath.dist0(data, k=torch.tensor(-1.))
    plt.hist(hyp_distance, bins=20)

    plt.subplot(6, 1, 3)
    # Euclidean distance between pairs of points
    data_expanded_1 = data.unsqueeze(1).expand(data.shape[0], data.shape[0], data.shape[1])
    data_expanded_2 = data.unsqueeze(0).expand(data.shape[0], data.shape[0], data.shape[1])
    euc_distances = torch.pow(data_expanded_1 - data_expanded_2, 2).sum(2)
    euc_distances = euc_distances.view(-1)
    plt.hist(euc_distances[euc_distances != 0], bins=20)

    plt.subplot(6, 1, 4)
    # Compute distances to origin (real hyperbolic distance)
    hyp_distances = gmath.dist(data_expanded_1.contiguous().view(-1, data.shape[1]),
                               data_expanded_2.contiguous().view(-1, data.shape[1]),
                               k=torch.tensor(-1.))
    plt.hist(hyp_distances[hyp_distances != 0], bins=20)

    plt.subplot(6, 1, 5)
    # Euclidean distance every dimension individually
    euc_distance_individual = data.view(-1).abs()
    plt.hist(euc_distance_individual, bins=20)

    plt.subplot(6, 1, 6)
    # Compute distances to origin (real hyperbolic distance) every dimension individually
    hyp_distance_individual = gmath.dist0(data.view(-1, 1), k=torch.tensor(-1.))
    plt.hist(hyp_distance_individual, bins=20)
    plt.show()

    midpoint = gmath.weighted_midpoint(data, k=torch.tensor(-1.))
    hyp_dist_midpoint_origin = gmath.dist0(midpoint, k=torch.tensor(-1.))
    euc_dist_midpoint_origin = midpoint.norm(dim=-1)
    print(f'Midpoint-origin Euclidean: {euc_dist_midpoint_origin:.03f}, Hyperbolic: {hyp_dist_midpoint_origin:.03f}')


if __name__ == '__main__':
    main()
