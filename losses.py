import torch
import utils.utils as utils


def compute_loss(args, score, pred_norm, gt_norm, labels, target, sizes, avg_meters, B):
    if args.finetune or (args.early_action and not args.early_action_self):
        gt = labels.repeat_interleave(args.num_seq).to(args.device)
        loss = torch.nn.functional.cross_entropy(score, gt)
        accuracy = (torch.argmax(score) == gt)

        # Bookkeeping
        avg_meters['losses'].update(loss.item(), score.shape[0] / args.num_seq)
        avg_meters['accuracy'].update(accuracy, B)

    else:
        if args.hyp_cone:
            # TODO Ruoshi version:
            # pred_norm = torch.mean(pred_norm)
            # gt_norm = torch.mean(gt_norm)
            # loss = score_.sum()

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
            avg_meters['p_norm'].update(pred_norm.item())
            avg_meters['g_norm'].update(gt_norm.item())
            avg_meters['losses'].update(loss.item() / (2 * k ** 2), B)
            avg_meters['accuracy'].update(pos_acc, B)

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
