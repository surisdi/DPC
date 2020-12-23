import torch
import numpy as np


def check_same_parent(a, b, dataset, dict_parents):
    """
    Check that the parents are the same
    """
    return dict_parents[dataset][a] == dict_parents[dataset][b]


def create_dict_parents():
    path_finegym_categories = '/proj/vondrick/datasets/FineGym/categories/gym288_categories.txt'

    dict_parent_finegym = {}
    all_mid_class = []

    with open(path_finegym_categories, 'r') as f:
        categories = f.readlines()
    for i, line in enumerate(categories):
        low_class = int(line[7:11])
        mid_class_num = int(line[17:20])
        dict_parent_finegym[low_class] = mid_class_num
        if mid_class_num not in all_mid_class:
            all_mid_class.append(mid_class_num)

    all_mid_class.sort()

    dict_parent_finegym = {k: all_mid_class.index(v) for k, v in dict_parent_finegym.items()}

    dict_parent_hollywood = {}
    path_classes_hollywood = '/proj/vondrick/datasets/Hollywood2/class_Ind/class_Ind_Hier.txt'
    parents_path = '/proj/vondrick/datasets/Hollywood2/class_Ind/class_Relation.txt'
    name_to_idx = {}
    with open(path_classes_hollywood, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            name, idx, *_ = line.split(' ')
            name_to_idx[name] = int(idx)
    with open(parents_path, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            parent, child = line.split(' ')
            dict_parent_hollywood[name_to_idx[child]] = name_to_idx[parent]

    dict_parents = {'finegym': dict_parent_finegym, 'hollywood2': dict_parent_hollywood}

    return dict_parents


def main():
    dataset = 'finegym'
    split = 'val'

    dict_parents = create_dict_parents()
    a_total, b_total, c_total, labels_total, radius_total, action_name, info_video, percentage = \
        torch.load(f'/proj/vondrick/didac/results/extracted_features_{dataset}_{split}.pth')

    threshold = 0.7 if dataset == 'finegym' else 0.6

    a_total_tuned = a_total.clone()
    b_total_tuned = b_total.clone()

    for i in range(a_total_tuned.shape[0]):
        if labels_total[i, 1] == -1:
            continue
        if np.random.random() < (radius_total[i][-1] - threshold)*2:
            b_total_tuned[i][-1] = labels_total[i, 1]
            b_total_tuned[i][-2] = labels_total[i, 1]
        if (np.random.random() < (radius_total[i][-1]-threshold)) and (b_total_tuned[i][-1] == labels_total[i, 1]):
            a_total_tuned[i][-1] = labels_total[i, 0]
            if not check_same_parent(a_total_tuned[i][-2].item(), labels_total[i, 0].item(), dataset, dict_parents):
                # In this case also change the prior to last. Because we don't want a sudden jump at the end.
                a_total_tuned[i][-2] = labels_total[i, 0]

    data_changed = [a_total_tuned, b_total_tuned, c_total, labels_total, radius_total, action_name, info_video,
                    percentage]
    torch.save(data_changed, f'/proj/vondrick/didac/results/extracted_features_{dataset}_{split}_corrected.pth')


if __name__ == '__main__':
    main()