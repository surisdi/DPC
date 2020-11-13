import torch
import sys
sys.path.append('../')
from losses import compute_mask, compute_scores, compute_loss
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import itertools
from tqdm import tqdm


def main(trainer):

    print('\n==== Visualizing trajectories ====\n')
    trainer.model.eval()

    input_seq, labels, index = iter(trainer.loaders['test']).next()
    input_seq = input_seq.to(trainer.args.device)

    with torch.no_grad():
        output_model = trainer.model(input_seq)
    pred, feature_dist, sizes_pred = output_model
    target, (B, B2, NS, NP, SQ) = compute_mask(trainer.args, sizes_pred[0], trainer.args.batch_size)
    # Some extra info useful to debug
    score = compute_scores(trainer.args, pred, feature_dist, sizes_pred[0], B)
    to_return = compute_loss(trainer.args, feature_dist, pred, labels, target, sizes_pred[0], (B, B2, NS, NP, SQ), B)

    selected_indices = True
    if selected_indices:
        # all_combinations = [[1, 9], [1, 10], [2, 9], [2, 10], [4, 6]]
        # all_combinations = [[3, 9], [3, 6], [4, 9], [5, 10]]
        all_combinations = [[10, 10], [11, 10], [5, 10], [6, 10], [7, 10], [1, 10], [2, 10], [3, 10], [10, 5], [5, 5], [1, 5]]
        unit_indices = [146, 121]   # [5, 232]  # [146, 121]  # [5, 146]
    else:
        for idx in index:
            idx_block, vpath = trainer.loaders['test'].dataset.get_info(idx.cpu().numpy())
            print(idx_block, vpath)

        if trainer.args.final_2dim:
            unit_indices = [0, 1]
        else:

            # pred.view(B, 7, 16, 256).permute(0,2,1,3).reshape(-1, 7, 256).var(1).mean(0).argsort()

            # This method is also good
            unit_indices = pred.abs().mean(0).argsort()[-2:].cpu()

            # Another possible method
            # (pred.view(B, 7, 16, 256).permute(0,2,1,3).reshape(-1, 7, 256)[:, -1] - pred.view(B, 7, 16, 256).permute(0,2,1,3).reshape(-1, 7, 256)[:, 0]).abs().mean(0).argsort()

            print('Computing best units')
            results = torch.zeros((256, 256))
            for unit1 in range(256):
                for unit2 in range(256):
                    if unit1 == unit2:
                        pass
                    else:
                        pred_ = torch.index_select(pred.cpu(), -1, torch.tensor([unit1, unit2])).pow(2).sum(-1).reshape(32, 7, trainer.model.last_size**2).permute(0, 2, 1).reshape(-1, 7)
                        results[unit1, unit2] = (pred_[:, -1] - pred_[:, 0]).mean()
            max_index = results.argmax()
            unit1 = max_index // 256
            unit2 = max_index - unit1*256
            unit_indices = [unit1, unit2]

        # If we want to plot all combinations
        all_combinations = []
        list_batch = list(range(15))
        list_positions = [0] if trainer.args.no_spatial else [5, 10]
        list1_permutations = itertools.permutations(list_batch, len(list_positions))
        for each_permutation in list1_permutations:
            zipped = zip(each_permutation, list_positions)
            all_combinations.extend(list(zipped))

    list_colors = ['royalblue', 'springgreen', 'goldenrod', 'mediumorchid', 'lightcoral', 'chocolate',
                   'lightsalmon', 'darkkhaki', 'palegreen', 'mediumturquoise', 'dodgerblue', 'indigo', 'deeppink']
    scale = 1
    fig = plt.figure(dpi=800)
    ax = fig.add_subplot(111)

    size_features = 2 if trainer.args.final_2dim else 256

    x_min = 5
    x_max = -5
    y_min = 5
    y_max = -5
    for j, (batch, position) in tqdm(enumerate(all_combinations), total=len(all_combinations)):
        color = mcolors.CSS4_COLORS[list_colors[j%len(list_colors)]]

        pred_reshaped = pred.view(B, trainer.args.num_seq - trainer.args.pred_step, trainer.model.last_size**2,
                                  size_features)[:, :, position].cpu()
        pred_reshaped = torch.index_select(pred_reshaped, -1, torch.tensor(unit_indices)) * scale
        features = feature_dist.view(B, -1, size_features)[:, position].cpu()
        features = torch.index_select(features, -1, torch.tensor(unit_indices)) * scale
        features = features.numpy()

        pred_consider = pred_reshaped[batch, :].cpu().numpy()

        if pred_consider[:, 0].min() < x_min:
            x_min = pred_consider[:, 0].min()
        if pred_consider[:, 0].max() > x_max:
            x_max = pred_consider[:, 0].max()
        if pred_consider[:, 1].min() < y_min:
            y_min = pred_consider[:, 1].min()
        if pred_consider[:, 1].max() > y_max:
            y_max = pred_consider[:, 1].max()
        if features[batch, 0].min() < x_min:
            x_min = features[batch, 0].min()
        if features[batch, 0].max() > x_max:
            x_max = features[batch, 0].max()
        if features[batch, 1].min() < y_min:
            y_min = features[batch, 1].min()
        if features[batch, 1].max() > y_max:
            y_max = features[batch, 1].max()

        ax.scatter(pred_consider[:, 0], pred_consider[:, 1], cmap='Spectral', s=1, c=color)  # c= would be color
        labels = list(range(NP))
        # for i, txt in enumerate(labels):
        #     ax.annotate(txt, (pred_consider[i, 0], pred_consider[i, 1]), fontsize=3)
        for i in range(trainer.args.num_seq - trainer.args.pred_step):
            point1 = pred_consider[i]
            if i == trainer.args.num_seq - trainer.args.pred_step - 1:
                alpha = 0.4
                point2 = features[batch]
            else:
                alpha = 1.0
                point2 = pred_consider[i+1]
            dx = point2[0]-point1[0]
            dy = point2[1]-point1[1]
            dx_margin = dx / np.sqrt(dx**2 + dy**2) * 0.003
            dy_margin = dy / np.sqrt(dx**2 + dy**2) * 0.003
            increment_x = dx-3*dx_margin if np.abs(3*dx_margin) < np.abs(dx) else 0.001 * dx
            increment_y = dy-3*dy_margin if np.abs(3*dy_margin) < np.abs(dy) else 0.001 * dy
            plt.arrow(point1[0]+dx_margin, point1[1]+dy_margin, increment_x,increment_y, linewidth=0.1,
                      color=color, alpha=alpha)
        ax.scatter(features[batch, 0], features[batch, 1], cmap='Spectral', s=1, c='r')  # c= would be color

    ax.scatter(0, 0, cmap='Spectral', s=1, c='k')  # c= would be color

    radius = 1.0 if trainer.args.final_2dim else 0.9
    # plt.gca().set_xlim(-1.1, 1.1)
    # plt.gca().set_ylim(-1.1, 1.1)
    # plt.gca().set_xlim(0.5, 1.1)
    # plt.gca().set_ylim(-0.3, 0.3)
    plt.gca().set_xlim(x_min-0.1, x_max+0.15)
    plt.gca().set_ylim(y_min-0.1, y_max+0.05)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.add_artist(plt.Circle((0, 0), radius, fc='none', ec='k'))
    ax.add_artist(plt.Circle((0, 0), radius*0.9, fc='none', ec='k', alpha=0.05))
    ax.add_artist(plt.Circle((0, 0), radius*0.8, fc='none', ec='k', alpha=0.05))
    ax.add_artist(plt.Circle((0, 0), radius*0.7, fc='none', ec='k', alpha=0.05))
    angle = 8
    ax.annotate('0.7', (radius * 0.7 * np.cos(angle*np.pi/180)-0.005, radius * 0.7 * np.sin(angle*np.pi/180)-0.005), color='k', alpha=0.3, rotation=angle, fontsize=5)
    ax.annotate('0.8', (radius * 0.8 * np.cos(angle*np.pi/180)-0.005, radius * 0.8 * np.sin(angle*np.pi/180)-0.005), color='k', alpha=0.3, rotation=angle, fontsize=5)
    ax.annotate('0.9', (radius * 0.9 * np.cos(angle*np.pi/180)-0.005, radius * 0.9 * np.sin(angle*np.pi/180)-0.005), color='k', alpha=0.3, rotation=angle, fontsize=5)
    ax.axis('off')
    plt.show()
    plt.savefig(f'/proj/vondrick/didac/results/2dim/trajectories_{"2dim" if trainer.args.final_2dim else ""}.pdf')

    plt.close()

