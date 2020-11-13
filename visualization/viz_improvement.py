"""
Visualize specific examples where hyperbolic model improves wrt the Euclidean one
"""
import torch
import sys
sys.path.append('../')
from losses import compute_mask, compute_scores, compute_loss
import numpy as np
from tqdm import tqdm
import collections


def main(trainer):

    print('\n==== Visualizing improvement ====\n')
    trainer.model.eval()

    dict_accuracies = {}
    for batch_idx, (input_seq, labels, index) in tqdm(enumerate(trainer.loaders['test'])):
        input_seq = input_seq.to(trainer.args.device)
        with torch.no_grad():
            output_model = trainer.model(input_seq)
        pred, feature_dist, sizes_pred = output_model
        target, (B, B2, NS, NP, SQ) = compute_mask(trainer.args, sizes_pred[0], input_seq.shape[0])
        # Some extra info useful to debug
        score = compute_scores(trainer.args, pred, feature_dist, sizes_pred[0], B)
        score = score[:, :, range(16), :, 0, range(16)]
        score = score.permute(0, 2, 1, 3).reshape(-1, input_seq.shape[0], input_seq.shape[0])
        score = ((-score).argsort(-1).cpu() == torch.tensor(range(input_seq.shape[0]))[None, :, None]).int().argmax(-1)
        score = score.float().mean(0)  # "mean" position wrt other elements in the batch

        for i, idx in enumerate(index):
            dict_accuracies[idx.item()] = score[i].item()

    torch.save(dict_accuracies, f'/proj/vondrick/didac/results/acc_{trainer.args.prefix}.pth.tar')

    dict_accuracies_hyper = torch.load(f'/proj/vondrick/didac/results/acc_viz_hyperbolic.pth.tar')
    dist = []
    for (k1, v1), (k2, v2) in zip(collections.OrderedDict(sorted(dict_accuracies_hyper.items())).items(),
                                  collections.OrderedDict(sorted(dict_accuracies.items())).items()):

        dist.append(v1 - v2)
    dist = np.array(dist)
    max_dist = dist.argmax()

    vpath, vlen = trainer.loaders['test'].dataset.video_info.iloc[max_dist]
