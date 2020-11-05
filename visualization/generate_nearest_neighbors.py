import torch
import pickle
import numpy as np
import sys
sys.path.append('../')
from utils.poincare_distance import poincare_distance
import os
import imageio
from tqdm import tqdm
import PIL.Image
import argparse
import json



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_path', default=None, type=str, help='path to the embeddings')
    parser.add_argument('--seed', default=2020, type=int, help='random seed for numpy')
    parser.add_argument('--num_examples', default=200, type=int, help='random seed for numpy')
    parser.add_argument('--mode', default=None, type=str, help='early_action / sub_action')
    parser.add_argument('--num_NN', default=10, type=int, help='number of nearest neighbor to generate')
    args = parser.parse_args()

    return args

def get_vpath(args, feature_dict, nn_indices, num_seen, pred_ind, seq_ind, K):
    '''
    Return vpath and list of K-th nearest neighbors
    '''
    if args.mode == 'early_action':
        sample_ind = nn_indices[num_seen-1][seq_ind][K] # pred_ind - 1 because pred_ind starts at 1
    elif args.mode == 'sub_action':
        sample_ind = nn_indices[pred_ind-1][seq_ind][K] # pred_ind - 1 because pred_ind starts at 1
    vpath = feature_dict['vpath'][seq_ind]
    idx_block = feature_dict['idx_block'][sample_ind]
    if args.mode == 'sub_action':
        start = idx_block[25 + (pred_ind-1) * 5] # TODO: generalize the index to arbitrary pred_step and num_seq
        end = start + 15
    elif args.mode == 'early_action':
        start = idx_block[35] # TODO: generalize the index to arbitrary pred_step and num_seq
        end = start + 15
    return vpath, np.arange(start.item(), end.item()), sample_ind

def generate_gif(args, vpath, idx_block, sample_ind=None, save=True, num_seen=None, pred_ind=None, seq_ind=None, K=None):
    '''
    generate gif given vpath and idx_block, save with time, pred, K information
    
    Input:
        vpath: path to the frames
        embed_path: path to the embeddings (used to export gif)
        idx_block: array of frame indices used to generate gifs
        pred_ind: index of predicted step (should be 0 for early action)
        num_seen: number of clips seen during prediction (should be 5 for subaction if 8 pred 3)
        **seq_ind: index of sequence in embeddings ([0, args.num_examples])**
        **sample_ind: index of sequences in NN_indices ([0, len(NN_indices)])**
        K: index of nearest neighbors
    
    '''
    if save:
        assert all(x is not None for x in [sample_ind, num_seen, pred_ind, seq_ind, K])
    images = []
    for idx in idx_block:
        im_path = os.path.join(vpath, 'image_%0.5d.jpg' % idx)
        if os.path.exists(im_path):
            images.append(PIL.Image.open(im_path))
    if save:
        NN_path = os.path.join('/'.join(args.embed_path.split('/')[:-1]), 'NN')
        if not os.path.exists(NN_path):
            os.mkdir(NN_path)
        action = vpath.split('/')[-2].replace(' ', '-')
        gif_path = os.path.join(NN_path, 'seq-%d_seen-%d_pred-%d_K-%d.gif' % \
                                (seq_ind, num_seen, pred_ind, K))
        args.gif_info['seq-%d_seen-%d_pred-%d_K-%d' % (seq_ind, num_seen, pred_ind, K)] = {'sample_ind': int(sample_ind), 'action': action}
        if images is not None and len(images) == len(idx_block):
            images[0].save(gif_path, format='GIF', append_images=images[1:], save_all=True, loop=0)
        return images, gif_path
    return images

def save_gif(args, feature_dict, nn_indices, num_seen=5, pred_ind=0, seq_ind=10, K=5):
    vpath, idx_block, sample_ind = get_vpath(args, feature_dict, nn_indices, num_seen=num_seen,\
                                             pred_ind=pred_ind, seq_ind=seq_ind, K=K)
    images, gif_path = generate_gif(args, vpath, idx_block, num_seen=num_seen,\
                                    sample_ind=sample_ind, save=True, pred_ind=pred_ind, seq_ind=seq_ind, K=K)

def main():
    args = get_args()
    
    embed_path = args.embed_path
    f = open(embed_path,"rb")
    feature_dict = pickle.load(f)
    
    num_embeds = len(feature_dict['vpath'])
    np.random.seed(args.seed)
    seq_index = np.sort(np.random.choice(range(num_embeds), replace=False, size=args.num_examples))
    pred_feature = feature_dict['pred'][seq_index]
    sample_feature = feature_dict['feature']
    
    # get list of score matrix with poincare distance
    score_list = []
    for i in range(pred_feature.shape[1]):
        score_list.append(poincare_distance(pred_feature[:, i, :], sample_feature[:, 0, :]))
    score_list = torch.stack(score_list)
    
    # calculate index metrix for nearest neighbors and the values for distance
    nn_indices = []
    nn_values = []
    for i in range(pred_feature.shape[1]):
        nn_indices.append(torch.topk(score_list[i], 10, dim=1, largest=False).indices)
        nn_values.append(torch.topk(score_list[i], 10, dim=1, largest=False).values)
    nn_indices = torch.stack(nn_indices)
    nn_values = torch.stack(nn_values)
    
    # dict object to save sequence and action information for later use
    args.gif_info = {}
    
    # generate all original video
    print('\n==== generating original videos====\n')
    for seq_ind in tqdm(range(len(seq_index))):
        sample_ind = seq_index[seq_ind]
        vpath = feature_dict['vpath'][sample_ind]
        idx_block = feature_dict['idx_block'][sample_ind]
        start = idx_block[0]
        end = start + 35 if args.mode == 'early_action' else start + 25 # TODO: make it general to all num_seq and pred_step
        idx_block = np.arange(start.item(), end.item())
        images = []
        for idx in idx_block:
            im_path = os.path.join(vpath, 'image_%0.5d.jpg' % idx)
            if os.path.exists(im_path):
                images.append(PIL.Image.open(im_path))

        NN_path = os.path.join('/'.join(embed_path.split('/')[:-1]), 'NN')
        if not os.path.exists(NN_path):
            os.mkdir(NN_path)
        action = vpath.split('/')[-2].replace(' ', '-')
        gif_path = os.path.join(NN_path, 'seq-%d.gif' % seq_ind)
        args.gif_info['seq-%d' % seq_ind] = {'sample_ind': int(sample_ind), 'action': action}
        images[0].save(gif_path, format='GIF', append_images=images[1:], save_all=True, loop=0)
        
    # generate all nearest neighbor videos
    print('\n==== generating nearest neighbors ====\n')
    for seq_ind in tqdm(range(len(seq_index))): # number of sequence sampled
        for step in range(1, pred_feature.shape[1]+1): # number of predicted features
            for K in range(args.num_NN): 
                if args.mode == 'early_action':
                    save_gif(args, feature_dict, nn_indices, num_seen=step, pred_ind=1, seq_ind=seq_ind, K=K)
                elif args.mode == 'sub_action':
                    save_gif(args, feature_dict, nn_indices, num_seen=5, pred_ind=step, seq_ind=seq_ind, K=K)

    with open(os.path.join(NN_path, 'gif_info.json'), 'w') as fp:
        json.dump(args.gif_info, fp)
    
if __name__ == '__main__':
    main()
    
    