import matplotlib.pyplot as plt
import umap.umap_ as umap
import torch
import geoopt.manifolds.stereographic.math as gmath
import numpy as np
from sklearn.decomposition import PCA
import pickle


def main():
    # Get data
    # data_path = '/proj/vondrick/shared/DPC/results/pred_hyp.pth'
    data_path = '/proj/vondrick/didac/results/pred.pth'

    # data_path = '/proj/vondrick/didac/results/cone_kinetics_300.pth'
    # data_path = '/proj/vondrick/didac/results/cone_moments_300.pth'
    #
    # data_kinetics = torch.load(data_path, map_location=torch.device('cpu'))
    # data_poincare = data_kinetics['embeddings']
    # labels = data_kinetics['objects']

    # option 1
    # with open('/proj/vondrick/shared/DPC/other_projects_hyp_feature/poincare_glove_100D_dist-sq_init_trick.txt', 'r', errors='ignore') as f:
    #     data = f.read()
    # data = data.split('\n')

    # option 2
    with open('/proj/vondrick/shared/DPC/other_projects_hyp_features/hyperbolic_cones_wordnet_mammals_2_hyp_cones.pth', 'rb') as f:
        data = pickle.load(f)

    all_embeddings = {}
    name_embedding = None
    current_embeddings = []
    current_labels = []

    # option 1
    # for dat in data:
    #     try:
    #
    #         dat_split = dat.split(' ')
    #         a = float(dat_split[1])  # just to check if it is a number
    #         current_labels.append(dat_split[0])
    #         current_embeddings.append(torch.tensor([float(val) for val in dat_split[1:-1]]))
    #     except:
    #         if name_embedding is not None:
    #             if len(current_embeddings) > 0:
    #                 current_embeddings = torch.stack(current_embeddings)
    #                 name_embedding = name_embedding.split('\x00')[-1]
    #                 all_embeddings[name_embedding] = (current_embeddings, current_labels)
    #         current_embeddings = []
    #         current_labels = []
    #         name_embedding = dat

    # option 2
    for i, (data_value, data_name) in enumerate(zip(data[0], data[1])):
        current_labels.append(data_name)
        current_embeddings.append(torch.tensor(data_value))
    current_embeddings = torch.stack(current_embeddings)

    # data_path = '/proj/vondrick/didac/results/cone_activity_net300.pth'
    # data_activitynet = torch.load(data_path, map_location=torch.device('cpu'))
    # data_poincare = data_activitynet['embeddings']
    # labels = data_activitynet['objects']

    # data_poincare, labels = all_embeddings['poincare_glove_100D_dist-sq_init_trick']
    #
    # plot_poincare(data_poincare[:10000], path_save='/proj/vondrick/didac/results/plot_poincare_10000_mds.png', show=True, mds=True, labels=labels[:10000])
    # plot_poincare(data_poincare[:100000], path_save='/proj/vondrick/didac/results/plot_poincare_100000_mds.png', show=True, mds=True, labels=labels[:100000])
    # plot_poincare(data_poincare[:1000], path_save='/proj/vondrick/didac/results/plot_poincare_1000_tsne.png', show=True, mds=False, labels=labels[:1000])

    data_poincare, labels = current_embeddings, current_labels

    plot_poincare(data_poincare[:10000], path_save='/proj/vondrick/didac/results/plot_cones_mamals_10000_mds.png', show=True, mds=True, labels=labels[:10000])
    plot_poincare(data_poincare[:100000], path_save='/proj/vondrick/didac/results/plot_cones_mamals_100000_mds.png', show=True, mds=True, labels=labels[:100000])
    plot_poincare(data_poincare[:1000], path_save='/proj/vondrick/didac/results/plot_cones_mamals_1000_tsne.png', show=True, mds=False, labels=labels[:1000])



def plot_poincare(data_poincare, path_save, show=False, mds=False, labels=None):
    k = torch.tensor(-1.)
    # TODO if this is a 1 (ie nothing) everything goes to the border. Is this because of this specific data? Try with
    #  different models
    # data_poincare = data_poincare /5
    if mds:
        data_hyperboloid_reduced = compute_hmds(data_poincare, distance_matrix=None, model='poincare', dimensions=2)

    else:
        # Project from Poincare disk to hyperboloid
        data_hyperboloid = gmath.inv_sproj(data_poincare, k=k)
        data_hyperboloid = data_hyperboloid.data.cpu().numpy()

        # Reduce dimensionality
        hyperbolic_mapper = umap.UMAP(output_metric='hyperboloid', random_state=1).fit(data_hyperboloid)
        data_hyperboloid_reduced = hyperbolic_mapper.embedding_
        z = np.sqrt(1 + np.sum(data_hyperboloid_reduced ** 2, axis=1))
        data_hyperboloid_reduced = np.concatenate([data_hyperboloid_reduced, z[:, None]], axis=-1)

    # Project to Poincare
    data_poincare_reduced = gmath.sproj(torch.tensor(data_hyperboloid_reduced), k=k).numpy()

    # TODO does not match! Should it match? Maybe we can make it match if I recenter at the original mean.
    #   but read the paper and try to understand it
    # if data_poincare.shape[1] == 2:
    #     data_poincare_reduced = data_poincare

    # Plot
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111)
    ax.scatter(data_poincare_reduced[:, 0], data_poincare_reduced[:, 1], cmap='Spectral', s=5)  # c= would be color

    if labels is not None:
        for i, txt in enumerate(labels):
            ax.annotate(txt, (data_poincare_reduced[i, 0], data_poincare_reduced[i, 1]), size=2)

    boundary = plt.Circle((0, 0), 1, fc='none', ec='k')
    plt.gca().set_xlim(-1.1, 1.1)
    plt.gca().set_ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.add_artist(boundary)
    ax.axis('off')
    plt.savefig(path_save)

    if show:
        plt.show()


def compute_hmds(data=None, distance_matrix=None, model='poincare', dimensions=2):
    """
    Compute h-MDS following the paper "Representation Tradeoffs for Hyperbolic Embeddings" (Algorithm 2)
    It is the closest to PCA in the hyperbolic space

    input model: hyperbolic model where data sits in, in order to compute distances
    If distance matrix is computed, we do not use data.
    The distances are the same independently of the hyperbolic model used.
    """
    assert data is not None or distance_matrix is not None, 'We have to obtain the data somehow'

    # Compute distance matrix, and then Y=cosh(d)
    if distance_matrix is None:
        x = data.unsqueeze(1).expand(data.shape[0], data.shape[0], data.shape[1]).contiguous().view(-1, data.shape[1])
        y = data.unsqueeze(0).expand(data.shape[0], data.shape[0], data.shape[1]).contiguous().view(-1, data.shape[1])
        if model == 'poincare':
            distance_matrix = gmath.dist(x=x, y=y, k=torch.tensor(-1.0))
            Y = torch.cosh(distance_matrix)
        else:  # model == 'hyperboloid'
            Y = hyperboloid_distance(x, y)
    else:
        Y = torch.cosh(distance_matrix)
    Y = Y.view(data.shape[0], data.shape[0]).detach()

    pca = PCA(n_components=dimensions, svd_solver='full')
    data_hyperboloid_reduced = pca.fit_transform(-Y.cpu().numpy())
    x0 = np.sqrt((data_hyperboloid_reduced**2).sum(axis=-1, keepdims=True)+1)
    data_hyperboloid_reduced = np.concatenate([data_hyperboloid_reduced, x0], axis=-1)

    return data_hyperboloid_reduced


def hyperboloid_distance(x, y, include_arcosh=False):
    """
    The distance is with the arccosh, but if we have to compute cosh on top of it, it is better to default without it
    """
    q_func = lambda x: -x[:-1].pow(2).sum() + x[-1].pow(2).sum()
    d = (q_func(x + y) - q_func(x) - q_func(y)) / 2
    if include_arcosh:
        d = torch.acosh(d)

    return d


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

