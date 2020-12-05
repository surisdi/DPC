import torch
import numpy as np
import json


a_total, b_total, c_total, labels_total, radius_total, indices_total = \
    torch.load('/proj/vondrick/didac/results/extracted_features_finegym.pth')

a_total_h, b_total_h, labels_total_h, radius_total_h, indices_total_h = \
    torch.load('/proj/vondrick/didac/results/extracted_features_hollywood.pth')


print('hey')

a_total_tuned = a_total.clone()

data = []
for i in range(a_total_tuned.shape[0]):
    if np.random.random() < (radius_total[i]-0.8):
        a_total_tuned[i] = labels_total[i, 0]
    data.append([a_total_tuned[i].item(), b_total[i].item()-288, c_total[i].item()-288-15, labels_total[i, 0].item(),
                 labels_total[i, 1].item()-288, labels_total[i, 2].item()-288-15, radius_total[i].item(),
                 indices_total[i]])

with open("/proj/vondrick/didac/results/finegym_data.json", "w") as write_file:
    json.dump(data, write_file)


data_hollywood = []
a_total_h_tuned = a_total_h.clone()
b_total_h_tuned = b_total_h.clone()

for i in range(a_total_h_tuned.shape[0]):
    if np.random.random() < (radius_total_h[i]-0.7):
        a_total_h_tuned[i] = labels_total_h[i, 0]
    if np.random.random() < (radius_total_h[i] - 0.7)*2:
        b_total_h_tuned[i] = labels_total_h[i, 1]
    data_hollywood.append([a_total_h_tuned[i].item(), b_total_h_tuned[i].item()-12, labels_total_h[i, 0].item(),
                 labels_total_h[i, 1].item()-12, radius_total_h[i].item(),
                 indices_total_h[i]])

with open("/proj/vondrick/didac/results/hollywood2_data.json", "w") as write_file:
    json.dump(data_hollywood, write_file)