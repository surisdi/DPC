import torch
import sys
sys.path.append('../')
from tqdm import tqdm
from losses import compute_mask
import pickle
import os
import matplotlib.pyplot as plt


def main(trainer):
    # collect features
    all_features = []
    all_preds = []
    all_vpaths = []
    all_idx_blocks = []

    print('\n==== Generating embeddings for pretrain model ====\n')
    with torch.no_grad():
        for idx_batch, (input_seq, labels, index) in \
                tqdm(enumerate(trainer.loaders['test']), total=int(len(trainer.loaders['test']) * trainer.args.partial)):
            input_seq = input_seq.to(trainer.args.device)
            output_model = trainer.model(input_seq)
            pred, feature_dist, sizes_pred = output_model
            target, (B, B2, NS, NP, SQ) = compute_mask(trainer.args, sizes_pred[0], trainer.args.batch_size)

            # Plot
            pred_reshaped = pred.view(B, NP, -1, 2).mean(2)
            batch = 0
            fig = plt.figure(dpi=400)
            ax = fig.add_subplot(111)
            for batch in range(B):
                pred_consider = pred_reshaped[batch, :].cpu().numpy()
                ax.scatter(pred_consider[:, 0], pred_consider[:, 1], cmap='Spectral', s=1)  # c= would be color
                labels = list(range(NP))
                for i, txt in enumerate(labels):
                    ax.annotate(txt, (pred_consider[i, 0], pred_consider[i, 1]), size=2)

            boundary = plt.Circle((0, 0), 1, fc='none', ec='k')
            plt.gca().set_xlim(-1.1, 1.1)
            plt.gca().set_ylim(-1.1, 1.1)
            plt.gca().set_aspect('equal', adjustable='box')
            ax.add_artist(boundary)
            ax.axis('off')
            # plt.savefig(path_save)
            plt.show()
            plt.savefig('/proj/vondrick/didac/results/prova.jpg')

            sizes_pred = sizes_pred.float().mean(0).int()
            _, D = pred.shape
            pred = pred.reshape(B, NP, SQ, D)
            feature_dist = feature_dist.reshape(B, NS, SQ, D)
            pred_pooled = torch.mean(pred, dim=2).reshape(-1, D)
            feature_dist_pooled = torch.mean(feature_dist, dim=2).reshape(-1, D)

            pred_pooled = pred_pooled.reshape(B, NP, D)
            feature_dist_pooled = feature_dist_pooled.reshape(B, NS, D)
            all_features.append(feature_dist_pooled.cpu().detach())
            all_preds.append(pred_pooled.cpu().detach())
            # for i in range(input_seq.shape[0]):
            #     block_info, vpath = trainer.loaders['test'].dataset.get_info(index[0])
            #     all_vpaths.append(vpath)
            #     all_idx_blocks.append(block_info)
            if idx_batch >= int(len(trainer.loaders['test']) * trainer.args.partial):
                break

    all_features = torch.cat(all_features)
    all_preds = torch.cat(all_preds)
    all_idx_blocks = torch.cat(all_idx_blocks)
    
    features_info = {'feature': all_features, 'pred': all_preds, 'vpath': all_vpaths, 'idx_block': all_idx_blocks}
    
    print('\n==== Saving features... ====\n')
    
    model_path = trainer.args.pretrain
    base_path = '/'.join(model_path.split('/')[:-2])
    embedding_path = os.path.join(base_path, 'embeds')
    if not os.path.exists(embedding_path):
        os.makedirs(embedding_path)

    f = open(os.path.join(embedding_path, model_path.split('/')[-1][:-8] + '_embeds.pkl'), 'wb')
    pickle.dump(features_info, f)
    f.close()
