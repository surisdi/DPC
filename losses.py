import torch
import utils.utils as utils
import geoopt
from utils.hyp_cone import HypConeDist
from utils.pairwise_hyp_cone import PairwiseHypConeDist
import copy
import numpy as np
from datasets import sizes_hierarchy as sh
from utils.poincare_distance import poincare_distance


def compute_loss(args, feature_dist, pred, labels, target, sizes_pred, sizes_mask, B, indices=None, trainer=None):

    if args.use_labels:
        # results, loss = compute_supervised_loss(args, pred, labels, B)  # TODO go back to notrmal
        to_return = compute_supervised_loss(args, pred, labels, B)
    else:
        results, loss = compute_selfsupervised_loss(args, pred, feature_dist, target, sizes_pred, sizes_mask, B, indices, trainer)

    # to_return = [loss] + [torch.tensor(r).cuda() for r in results]   # TODO go back to notrmal
    return to_return


def compute_supervised_loss(args, pred, labels, B):  #, top_down=False, separate_levels=True):
    """
    Six options to predict:
    1. Predict a single label for each sample (clip). Both num of labels and prediction size are equal to batch size
    2. Predict a single label, but for subclip-level predictions. Prediction size has a temporal element. Repeat label
    3. Same as 1 but also with hierarchical information
    4. Same as 2 but also with hierarchical information
    5. Predict with sub-action level. Prediction size and label size are the same, and larger than batch size
    6. Predict with sub-action level and also predict parent nodes (args.hierarchical_labels)
    7. Same as 5 but we only want to predict the last label given the predictions in the prior-to-last subclip
    8. Similar to 7 but with parent nodes
    """
    # For points 7 and 8. This is not very efficient because it computes unused predictions, but is just a linear layer
    if args.pred_future:
        assert (pred.shape[0] == B * args.num_seq) and (labels.shape[1] == args.num_seq)
        labels = labels[:, -1]
        pred = pred.view(B, args.num_seq, -1)[:, -2]

    if not args.hierarchical_labels:
        hier_accuracies = -1
        if labels.shape[0] < pred.shape[0]:
            if len(labels.shape) == 1:  # Option 2
                assert pred.shape[0] % labels.shape[0] == 0, \
                    'Maybe you are only using some predictions for some time steps and not all of them? In that ' \
                    'case, select the appropriate labels (either in this function, or in the dataloader). In that ' \
                    'case, you should not enter in this "if", and go directly to the "else"'
                gt = labels.repeat_interleave(args.num_seq).to(args.device)
            else:  # We also have temporal information in the labels (subaction labels). Option 5
                gt = labels.view(-1).to(args.device)
        else:  # Option 1
            gt = labels.to(args.device)
        loss = torch.nn.functional.cross_entropy(pred, gt, ignore_index=-1)

        accuracies = (torch.argmax(pred, dim=1) == gt).float()

    else:
        # train with multiple positive labels
        if labels.shape[0] < pred.shape[0]:
            if len(labels.shape) == 2:  # Option 4
                assert pred.shape[0] % labels.shape[0] == 0
                labels = labels.repeat_interleave(args.num_seq, dim=0).to(args.device)
            else:  # labels should have 3 dimensions (batch, temporal, hierarchy). Option 6
                labels = labels.view(-1, labels.shape[-1]).to(args.device)
        else:  # Options 2
            labels = labels.to(args.device)

        # pred = pred[labels[:, 0] != -1]
        # labels = labels[labels[:, 0] != -1]
        #
        # gt = torch.zeros(list(labels.shape[:-1]) + [pred.size(1)]).to(args.device)   # multi-label ground truth tensor
        # indices = torch.tensor(np.indices(labels.shape[:-1])).view(-1, 1).expand_as(labels)
        # gt[indices, labels] = 1
        #
        # loss = (- gt * torch.nn.functional.log_softmax(pred, -1)).sum()/gt.sum()  # CE loss with logit as ground truth
        # accuracies = (torch.argmax(pred[:, :sh[args.dataset][1][0]], dim=1) == labels[:, 0]).float()

        # TODO remove this
        i = 0
        init, end = (int(np.array(sh[args.dataset][1][0:i]).sum()), np.array(sh[args.dataset][1][0:i + 1]).sum())
        a = torch.argmax(pred[:, init:end], dim=1) + int(np.array(sh[args.dataset][1][0:i]).sum())
        i = 1
        init, end = (int(np.array(sh[args.dataset][1][0:i]).sum()), np.array(sh[args.dataset][1][0:i + 1]).sum())
        b = torch.argmax(pred[:, init:end], dim=1) + int(np.array(sh[args.dataset][1][0:i]).sum())
        i = 2
        init, end = (int(np.array(sh[args.dataset][1][0:i]).sum()), np.array(sh[args.dataset][1][0:i + 1]).sum())
        c = torch.argmax(pred[:, init:end], dim=1) + int(np.array(sh[args.dataset][1][0:i]).sum())



        # a = a.view(labels.shape[0]//6, 6)[:, 3]
        # b = b.view(labels.shape[0]//6, 6)[:, 3]
        # labels = labels.view(labels.shape[0]//6, 6, 2)[:, 3]

        return a, b, c, labels
        #####

        hier_accuracies = []
        for top_down in [True, False]:
            for separate_levels in [True, False]:
                hier_accuracy = 0
                reward = 1
                # reward value decay by 50% per level going up (or down)
                for i in (reversed if top_down else lambda x: x)(range(labels.size(1))):
                    if separate_levels:
                        init, end = (int(np.array(sh[args.dataset][1][0:i]).sum()), np.array(sh[args.dataset][1][0:i+1]).sum())
                        hier_accuracy += ((torch.argmax(pred[:, init:end], dim=1) ==
                                           (labels[:, i] - int(np.array(sh[args.dataset][1][0:i]).sum()))).float().mean() * reward)
                    else:
                        hier_accuracy += ((torch.argmax(pred[:, 0:sh[args.dataset][0]], dim=1) == (labels[:, i])).float().mean() * reward)
                    reward = reward / 2
                hier_accuracies.append(hier_accuracy)
        hier_accuracies = torch.tensor(hier_accuracies)

    if args.early_action:
        accuracy = accuracies.view(B, -1).mean(0)
    else:
        accuracy = accuracies.mean()

    results = accuracy, hier_accuracies, loss.item(), labels.shape[0]
    return results, loss


def compute_selfsupervised_loss(args, pred, feature_dist, target, sizes_pred, sizes_mask, B, indices, trainer):
    score = compute_scores(args, pred, feature_dist, sizes_pred, B, indices, trainer)

    _, B2, NS, NP, SQ = sizes_mask
    # score is a 6d tensor: [B, P, SQ, B2, N, SQ]
    # similarity matrix is computed inside each gpu, thus here B == num_gpu * B2
    score_flattened = score.view(B * NP * SQ, B2 * NS * SQ)
    target_flattened = target.view(B * NP * SQ, B2 * NS * SQ)
    target_flattened = target_flattened.float().argmax(dim=1)

    loss = torch.nn.functional.cross_entropy(score_flattened, target_flattened)
    top1, top3, top5 = utils.calc_topk_accuracy(score_flattened, target_flattened, (1, 3, 5))

    results = top1, top3, top5, loss.item(), B

    if args.hyp_cone:  # Add on top of the previous one, instead of replacing it
        dist_fn = PairwiseHypConeDist(K=0.1, fp64_hyper=args.fp64_hyper)
        score_cone = dist_fn(pred, feature_dist)

        score_cone[score_cone < 0] = 0
        pos = score_cone.diag().clone()
        neg = torch.relu(args.margin - score_cone)
        neg.fill_diagonal_(0)
        loss_pos = pos.sum()
        loss_neg = neg.sum() / score_cone.shape[0]
        loss_cone = loss_pos + loss_neg

        _, size_1 = score_cone.shape
        score = score_cone[:size_1, :]
        pos = score.diagonal(dim1=-2, dim2=-1)
        pos_acc = float((pos == 0).sum().item()) / float(pos.flatten().shape[0])
        k = score.shape[0]
        score.as_strided([k], [k + 1]).copy_(torch.zeros(k))
        neg_acc = float((score == 0).sum().item() - k) / float(k ** 2 - k)

        loss_cone = loss_cone / (2 * k)

        loss = loss_cone + loss
        results = pos_acc, neg_acc, loss_cone.item(), B  # These are the reported values

    return results, loss


def compute_scores(args, pred, feature_dist, sizes, B, indices=None, trainer=None):
    last_size, size_gt, size_pred = sizes.cpu().numpy()
    if args.hyperbolic:
        score = poincare_distance(pred, feature_dist)

        # from shutil import copyfile
        # import os
        # score_ = score.view(B, size_pred, last_size ** 2, B, 6, last_size ** 2)[:, :, range(16), :, :, range(16)].mean(
        #     0)
        # source_file_element = '/proj/vondrick/datasets/FineGym/action_videos'
        # source_file = '/proj/vondrick/datasets/FineGym/event_videos'
        #
        # for id in range(50):
        #     dst_file = f'/proj/vondrick/didac/results/fig8/{id}'
        #     os.makedirs(dst_file, exist_ok=True)
        #     query_name = trainer.loaders["test"].dataset.idx2clipidx[indices[0][id].cpu().item()]
        #     copyfile(source_file + f'/{query_name}.mp4', dst_file + f'/query_{query_name}.mp4')
        #     print(id, (score_ <= score_[id, -1].min())[id].sum(dim=[-2, -1]))
        #     list_min = score_[id, -1].view(-1).argsort()
        #
        #     # action gt
        #     segments = trainer.loaders['test'].dataset.clips[query_name]
        #     segments_keys = list(segments.keys())
        #     start = (len(segments) - trainer.loaders['test'].dataset.num_seq) // 2
        #     segments_keys = segments_keys[start:start + trainer.loaders['test'].dataset.num_seq]
        #     segment_key = segments_keys[-1]
        #     action_id = query_name + '_' + segment_key
        #     copyfile(source_file_element + f'/{action_id}.mp4', dst_file + f'/ground_truth_{i:02d}_{action_id}.mp4')
        #
        #     for i, element in enumerate(list_min[:15]):
        #         video_min, clip_min = element // 6, element % 6
        #         idx = indices[0][video_min]
        #         clipidx = trainer.loaders['test'].dataset.idx2clipidx[idx.cpu().item()]
        #         segments = trainer.loaders['test'].dataset.clips[clipidx]
        #         segments_keys = list(segments.keys())
        #         start = (len(segments) - trainer.loaders['test'].dataset.num_seq) // 2
        #         segments_keys = segments_keys[start:start + trainer.loaders['test'].dataset.num_seq]
        #         segment_key = segments_keys[clip_min.cpu().item()]
        #         action_id = clipidx + '_' + segment_key
        #         copyfile(source_file_element + f'/{action_id}.mp4', dst_file + f'/target_{i:02d}_{action_id}.mp4')

        if args.distance == 'squared':
            score = score.pow(2)
        elif args.distance == 'cosh':
            score = torch.cosh(score).pow(2)
        score = - score.float()
        score = score.view(B, size_pred, last_size ** 2, B, size_gt, last_size ** 2)

    else:  # euclidean dot product
        ### Get similarity score ###
        # pred: [B, pred_step, D, last_size, last_size]
        # GT: [B, N, D, last_size, last_size]
        score = torch.matmul(pred, feature_dist.transpose(0, 1))
        score = score.view(B, size_pred, last_size ** 2, B, size_gt, last_size ** 2)

    return score


def compute_features(args, pred, labels, B):
    if args.pred_future:
        assert (pred.shape[0] == B * args.num_seq) and (labels.shape[1] == args.num_seq)
        labels = labels[:, -1]
    pred = pred.view(B, args.num_seq, -1)

    assert args.hierarchical_labels

    if labels.shape[0] < pred.shape[0]:
        if len(labels.shape) == 2:  # Option 4
            assert pred.shape[0] % labels.shape[0] == 0
            labels = labels.repeat_interleave(args.num_seq, dim=0).to(args.device)
        else:  # labels should have 3 dimensions (batch, temporal, hierarchy). Option 6
            labels = labels.view(-1, labels.shape[-1]).to(args.device)
    else:  # Options 2
        labels = labels.to(args.device)

    percent = []
    i = 0
    init, end = (int(np.array(sh[args.dataset][1][0:i]).sum()), np.array(sh[args.dataset][1][0:i + 1]).sum())
    a = torch.argmax(pred[:, :, init:end], dim=-1) + int(np.array(sh[args.dataset][1][0:i]).sum())
    percent.append(torch.nn.functional.softmax(pred[:, :, init:end], dim=-1).max(-1)[0])
    i = 1
    init, end = (int(np.array(sh[args.dataset][1][0:i]).sum()), np.array(sh[args.dataset][1][0:i + 1]).sum())
    b = torch.argmax(pred[:, :, init:end], dim=-1) + int(np.array(sh[args.dataset][1][0:i]).sum())
    percent.append(torch.nn.functional.softmax(pred[:, :, init:end], dim=-1).max(-1)[0])

    if len(sh[args.dataset][1]) == 3:
        i = 2
        init, end = (int(np.array(sh[args.dataset][1][0:i]).sum()), np.array(sh[args.dataset][1][0:i + 1]).sum())
        c = torch.argmax(pred[:, :, init:end], dim=-1) + int(np.array(sh[args.dataset][1][0:i]).sum())
        percent.append(torch.nn.functional.softmax(pred[:, :, init:end], dim=-1).max(-1)[0])

    else:
        c = None

    return a, b, c, labels, percent


def compute_mask(args, sizes, B):
    if args.use_labels:
        return None, None  # No need to compute mask

    last_size, size_gt, size_pred = sizes

    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    mask = torch.zeros((B, size_pred, last_size ** 2, B, size_gt, last_size ** 2), dtype=torch.int8, requires_grad=False).detach().cuda()

    mask[torch.arange(B), :, :, torch.arange(B), :, :] = -3  # spatial neg

    if args.early_action_self:
        pass  # Here NO temporal neg! All steps try to predict the last one
    else:
        for k in range(B):
            mask[k, :, torch.arange(last_size ** 2), k, :, torch.arange(last_size ** 2)] = -1  # temporal neg

    tmp = mask.permute(0,2,1,3,5,4).reshape(B * last_size ** 2, size_pred, B * last_size ** 2, size_gt)

    if args.early_action_self:
        tmp[torch.arange(B * last_size ** 2), :, torch.arange(B * last_size ** 2)] = 1  # pos
    else:
        assert size_gt == size_pred
        for j in range(B * last_size ** 2):
            tmp[j, torch.arange(size_pred), j, torch.arange(size_gt)] = 1  # pos

    mask = tmp.view(B, last_size ** 2, size_pred, B, last_size ** 2, size_gt).permute(0, 2, 1, 3, 5, 4)

    # Now, given task mask as input, compute the target for contrastive loss
    if mask is None:
        return None, None
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size()  # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False

    return target, (B, B2, NS, NP, SQ)


def bookkeeping(args, avg_meters, results):
    if args.use_labels:
        accuracy, hier_accuracy, loss, B = results
        avg_meters['losses'].update(loss, B)
        avg_meters['accuracy'].update(accuracy.float(), B)
        avg_meters['hier_accuracy'].update(hier_accuracy.float(), B)
    else:
        if args.hyp_cone:
            pos_acc, neg_acc, loss, B = results
            avg_meters['pos_acc'].update(pos_acc, B)
            avg_meters['neg_acc'].update(neg_acc, B)
            avg_meters['losses'].update(loss, B)
            avg_meters['accuracy'].update(pos_acc, B)
        else:
            top1, top3, top5, loss, B = results
            avg_meters['top1'].update(top1, B)
            avg_meters['top3'].update(top3, B)
            avg_meters['top5'].update(top5, B)
            avg_meters['losses'].update(loss, B)
            avg_meters['accuracy'].update(top1, B)
            

