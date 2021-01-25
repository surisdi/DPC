"""
1. The BoxTensor interface guarantees that you will have valid tensors for .z (min coordinate), .Z (max coordinate),
and .center. So you can create any operations using these attributes.
2. For binary operations you can look at the 'intersection' module for reference. You can even inherit from the base
'Intersection' class as it performs some automatic broadcasting.
3. For unary operations, the 'volume' module is a good reference.
"""

import numpy as np
import torch
from box_embeddings.parameterizations import delta_box_tensor
from box_embeddings.modules.intersection import GumbelIntersection
from box_embeddings.modules.volume import BesselApproxVolume


def box_distance(pred, gt):
    '''
    Takes as inputs vectors in the Euclidean space that do NOT correspond to boxes yet.
    '''



    return pairwise_box_distances(pred, gt)


def pairwise_box_distances(x, y=None):
    '''
    '''
    if y is None:
        y = x

    box_embedding_a = delta_box_tensor.MinDeltaBoxTensor.from_vector(x)
    box_embedding_b = delta_box_tensor.MinDeltaBoxTensor.from_vector(y)

    intersection = GumbelIntersection(beta=1e-3)
    volume = BesselApproxVolume(log_scale=True, beta=1, gumbel_beta=1e-3)

    box_intersection_a_b = intersection(box_embedding_a, box_embedding_b)
    log_vol_a_b = volume(box_intersection_a_b)
    log_vol_b = volume(box_embedding_b)
    log_prob_a_given_b = log_vol_a_b - log_vol_b

    return log_prob_a_given_b


# test
array_1 = torch.tensor([1, 1, 2, 2]).float()
array_2 = torch.tensor([2, 2, 4, 4]).float()
pairwise_box_distances(array_1, array_2)