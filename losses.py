import torch
import utils.utils as utils
import geoopt
from utils.hyp_cone import HypConeDist
import copy


def compute_loss(args, score, pred, labels, target, sizes, B):

    if args.finetune or (args.early_action and not args.early_action_self):
        if not args.hierarchical:
            gt = labels.repeat_interleave(args.num_seq).to(args.device)
            loss = torch.nn.functional.cross_entropy(pred, gt)
            accuracy = (torch.argmax(pred, dim=1) == gt).float().mean()
            
        else:
            if args.method == 1:
                gt = torch.zeros(B, pred.size(1))
                gt[torch.arange(gt.size(0)).unsqueeze(1), labels] = 1 # multi-label ground truth tensor
                gt = torch.repeat_interleave(gt, args.num_seq, dim=0).to(args.device)
                labels = torch.repeat_interleave(labels, args.num_seq, dim=0).to(args.device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, gt) # CE loss with logit as ground truth
                accuracy = (torch.argmax(pred, dim=1) == labels[:, 0]).float().mean()
                hier_accuracy = 0
                reward = 1
                # reward value decay by 50% per level going up
                for i in range(labels.size(1)):
                    hier_accuracy += ((torch.argmax(pred, dim=1) == labels[:, i]).float().mean() * reward)
                    reward = reward / 2 
            elif args.method == 2:
                raise ValueError('method not implemented yet')
        results = accuracy, hier_accuracy, loss.item() / args.num_seq
    else:
        if args.hyp_cone:
            score[score < 0] = 0
            pos = score.diag().clone()
            neg = torch.relu(args.margin - score)
            neg.fill_diagonal_(0)
            loss_pos = pos.sum()
            loss_neg = neg.sum() / score.shape[0]
#             print('pos_loss ', loss_pos.item())
#             print('neg_loss ', loss_neg.item())
            loss = loss_pos + loss_neg

            # TODO check this
            [A, B] = score.shape
            score = score[:B, :]
            pos = score.diagonal(dim1=-2, dim2=-1)
            pos_acc = float((pos == 0).sum().item()) / float(pos.flatten().shape[0])
            k = score.shape[0]
            score.as_strided([k], [k + 1]).copy_(torch.zeros(k))
            neg_acc = float((score == 0).sum().item() - k) / float(k ** 2 - k)

            results = pos_acc, neg_acc, loss.item() / (2 * k ** 2)

        else:
            _, B2, NS, NP, SQ = sizes
            # score is a 6d tensor: [B, P, SQ, B2, N, SQ]
            # similarity matrix is computed inside each gpu, thus here B == num_gpu * B2
            score_flattened = score.view(B * NP * SQ, B2 * NS * SQ)
            target_flattened = target.view(B * NP * SQ, B2 * NS * SQ)
            target_flattened = target_flattened.float().argmax(dim=1)

            loss = torch.nn.functional.cross_entropy(score_flattened, target_flattened)
            top1, top3, top5 = utils.calc_topk_accuracy(score_flattened, target_flattened, (1, 3, 5))

            results = top1, top3, top5, loss.item()

    to_return = [loss] + [torch.tensor(r).cuda() for r in results + (B,)]
    return to_return


def compute_scores(args, pred, feature_dist, sizes, B):
    if args.finetune or (args.early_action and not args.early_action_self):
        return None  # No need to compute scores

    last_size, size_gt, size_pred = sizes.cpu().numpy()

    if args.hyperbolic:

        shape_expand = (pred.shape[0], feature_dist.shape[0], pred.shape[1])
        pred_expand = pred.unsqueeze(1).expand(shape_expand)
        gt_expand = feature_dist.unsqueeze(0).expand(shape_expand)
#         print(pred_expand.shape)
#         print(gt_expand.shape)

        if args.hyp_cone:

            dist_fn = HypConeDist(K=0.1, fp64_hyper=args.fp64_hyper)
            reshape_size = (pred.shape[0] * feature_dist.shape[0], pred.shape[1])
            pred_expand = pred_expand.reshape(reshape_size)
            gt_expand = gt_expand.reshape(reshape_size)
#             print(pred_expand.shape)
#             print(gt_expand.shape)
            score = dist_fn(pred_expand.float(), gt_expand.float())

            # loss function (equation 32 of https://arxiv.org/abs/1804.01882)
            score = score.reshape(B * size_pred * last_size ** 2, B * size_gt * last_size ** 2)

        else:
            # distance can also be computed with geoopt.manifolds.PoincareBall(c=1) -> .dist
            # Maybe more accurate (it is more specific for poincare)
            # But this is much faster... TODO implement batch dist_matrix on geoopt library
            # score = dist_matrix(pred_hyp, feature_inf_hyp)

            manif = geoopt.manifolds.PoincareBall(c=1)

            # score = manif.dist(pred_expand, gt_expand)
            score = manif.dist(pred_expand.float(), gt_expand.float())
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


def bookkeeping(args, avg_meters, results):
    if args.finetune or (args.early_action and not args.early_action_self):
        accuracy, hier_accuracy, loss, B = results
        avg_meters['losses'].update(loss, B)
        avg_meters['accuracy'].update(accuracy, B)
        avg_meters['hier_accuracy'].update(hier_accuracy, B)
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