import os
import random

root = '/proj/vondrick/datasets/Hollywood2'

data = []

for folder, video_folder, k in [('ClipSets', 'AVIClips_mp4', 1), ('ClipSetsScenes', 'AVIClipsScenes_mp4', 0)]:
    for file in os.listdir(os.path.join(root, folder)):
        if file.endswith('.txt'):
            name_file = file.split('.')[0].split('_')
            class_name, train_test = name_file[0], name_file[1]
            if class_name == 'scenes' or class_name == 'actions':
                continue
            with open(os.path.join(root, folder, file), 'r') as f:
                for line in f:
                    split_line = line.replace('  ', ' ').replace('\n', '').split(' ')
                    if int(split_line[1]) == 1:
                        data.append((os.path.join(video_folder, split_line[0] + '.mp4'), class_name, k, train_test))

random.shuffle(data)

final_path = '/proj/vondrick/didac/www/hyperbolic/explore/hollywood2_videos.txt'

with open(final_path, 'w') as f:
    for path, class_name, k, train_test in data:
        f.writelines(f'{path} {class_name} {k} {train_test}\n')
