import torch
import numpy as np
import geoopt
from utils.pairwise_hyp_cone import PairwiseHypConeDist

def cone_distance_sum(pred, samples, K = 0.1, return_index=False):
    '''
    Input: pred [N_pred, D]: list of hyperbolic cones
           samples [N_samples, D]: list of points in the poincare ball with norm < 1
    
    Output: neg_num [N_pred]: number of samples under each cone in pred
            dist_avg [N_pred]: average of distances between samples and each cone in pred
            dist0_pred [N_pred]: hyperbolic distance between cones in pred to origin
            (optional) neg_index: list of indices of samples under each cone in pred
            
    GPU memory:
        can support following input on single GPU:
        pred: [256, 256]
        samples: [2e5, 256]
    '''
    pairwise_dist_fn = PairwiseHypConeDist(K = K)
    dist = pairwise_dist_fn(pred, samples)
    negative = dist < 0
    neg_num = torch.sum(negative, axis=1)
    dist_avg = torch.mean(dist, dim=1)
    manif = geoopt.manifolds.PoincareBall(c=1)
    dist0_pred = manif.dist(pred, torch.zeros_like(pred))
    if return_index:
        neg_index = [torch.nonzero(negative[i], as_tuple=False).squeeze() for i in range(negative.shape[0])]
        return {'neg_index': neg_index, 'neg_num': neg_num, 'dist_avg': dist_avg, 'dist0_pred': dist0_pred}
    return {'neg_num': neg_num, 'dist_avg': dist_avg, 'dist0_pred': dist0_pred}