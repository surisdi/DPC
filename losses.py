import torch
import utils.utils as utils
import geoopt
from utils.hyp_cone import HypConeDist
import time


def compute_loss(args, score, pred, labels, target, sizes, B, avg_meters, pred_norm, gt_norm):

    if args.finetune or (args.early_action and not args.early_action_self):
        gt = labels.repeat_interleave(args.num_seq).to(args.device)
        loss = torch.nn.functional.cross_entropy(pred, gt)
        accuracy = (torch.argmax(pred, dim=1) == gt).float().mean()

        # Bookkeeping
        avg_meters['losses'].update(loss.item(), pred.shape[0] / args.num_seq)
        avg_meters['accuracy'].update(accuracy, B)

    else:
        if args.hyp_cone:
            if args.hyp_cone_ruoshi:
                # Ruoshi version:
                pred_norm = torch.mean(pred_norm)
                gt_norm = torch.mean(gt_norm)
                loss = score.sum()
            else:
                # Didac version
                score[score < 0] = 0
                pos = score.diag().clone()
                neg = torch.relu(args.margin - score)
                neg.fill_diagonal_(0)
                loss_pos = pos.sum()
                loss_neg = neg.sum() / score.shape[0]
                loss = loss_pos + loss_neg

            [A, B] = score.shape
            score = score[:B, :]
            pos = score.diagonal(dim1=-2, dim2=-1)
            pos_acc = float((pos == 0).sum().item()) / float(pos.flatten().shape[0])
            k = score.shape[0]
            score.as_strided([k], [k + 1]).copy_(torch.zeros(k))
            neg_acc = float((score == 0).sum().item() - k) / float(k ** 2 - k)

            # Bookkeeping
            avg_meters['pos_acc'].update(pos_acc, B)
            avg_meters['neg_acc'].update(neg_acc, B)
            avg_meters['losses'].update(loss.item() / (2 * k ** 2), B)
            avg_meters['accuracy'].update(pos_acc, B)
            if args.hyp_cone_ruoshi:
                avg_meters['p_norm'].update(pred_norm.item())
                avg_meters['g_norm'].update(gt_norm.item())

        else:
            _, B2, NS, NP, SQ = sizes
            # score is a 6d tensor: [B, P, SQ, B2, N, SQ]
            # similarity matrix is computed inside each gpu, thus here B == num_gpu * B2
            score_flattened = score.view(B * NP * SQ, B2 * NS * SQ)
            target_flattened = target.view(B * NP * SQ, B2 * NS * SQ)
            target_flattened = target_flattened.float().argmax(dim=1)

            loss = torch.nn.functional.cross_entropy(score_flattened, target_flattened)
            top1, top3, top5 = utils.calc_topk_accuracy(score_flattened, target_flattened, (1, 3, 5))

            # Bookkeeping
            avg_meters['top1'].update(top1.item(), B)
            avg_meters['top3'].update(top3.item(), B)
            avg_meters['top5'].update(top5.item(), B)
            avg_meters['losses'].update(loss.item(), B)
            avg_meters['accuracy'].update(top1.item(), B)

    return loss


def compute_scores(args, pred, feature_dist, sizes, B):
    if args.finetune or (args.early_action and not args.early_action_self):
        return None, None, None  # No need to compute scores

    last_size, size_gt, size_pred = sizes
    pred_norm = gt_norm = None

    if args.hyperbolic:

        shape_expand = (pred.shape[0], feature_dist.shape[0], pred.shape[1])
        pred_expand = pred.unsqueeze(1).expand(shape_expand)
        gt_expand = feature_dist.unsqueeze(0).expand(shape_expand)

        if args.hyp_cone:

            dist_fn = HypConeDist(K=0.1)
            score = dist_fn(pred_expand.float(), gt_expand.float())

            # loss function (equation 32 of https://arxiv.org/abs/1804.01882)
            score = score.reshape(B * size_pred * last_size ** 2, B * size_gt * last_size ** 2)

            if args.hyp_cone_ruoshi:
                pred_norm = torch.mean(torch.norm(pred_expand, p=2, dim=-1))
                gt_norm = torch.mean(torch.norm(gt_expand, p=2, dim=-1))
                score[score < 0] = 0
                pos = score.diagonal(dim1=-2, dim2=-1)
                score = args.margin - score
                score[score < 0] = 0
                k = score.size(0)
                # positive score is multiplied by number of negative samples
                weight = 1
                score.as_strided([k], [k + 1]).copy_(pos * (k - 1) * weight)

        else:
            # distance can also be computed with geoopt.manifolds.PoincareBall(c=1) -> .dist
            # Maybe more accurate (it is more specific for poincare)
            # But this is much faster... TODO implement batch dist_matrix on geoopt library
            # score = dist_matrix(pred_hyp, feature_inf_hyp)

            manif = geoopt.manifolds.PoincareBall(c=1)

            # score = manif.dist(pred_expand, gt_expand)
            score = manif.dist(pred_expand.half(), gt_expand.half())
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

    return score, pred_norm, gt_norm


def compute_mask(args, sizes, B):
    if args.finetune or (args.early_action and not args.early_action_self):
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

