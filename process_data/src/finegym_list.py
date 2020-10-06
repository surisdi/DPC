import os
import random
import json

root = '/proj/vondrick/datasets/FineGym'

data = []

with open(os.path.join(root, 'annotations/finegym_annotation_info_v1.1.json'), 'r') as f:
    annotations = json.load(f)

labels_train = {}
with open(os.path.join(root, 'annotations/gym288_train_element_v1.1.txt'), 'r') as f:
    for line in f:
        data_split = line.replace('\n', '').split()
        labels_train[data_split[0]] = int(data_split[1])

labels_test = {}
with open(os.path.join(root, 'annotations/gym288_val_element.txt'), 'r') as f:
    for line in f:
        data_split = line.replace('\n', '').split()
        labels_test[data_split[0]] = int(data_split[1])

categories = {}
with open(os.path.join(root, 'categories/gym288_categories.txt'), 'r') as f:
    for line in f:
        data_split = line.replace('\n', '').split(';')
        clabel = int(data_split[0].split(' ')[-1])
        set_label = int(data_split[1].split(' ')[-1])
        sport_label = data_split[-1][2:4]
        sport_label = {"VT": "vault", "FX": "floor", "BB": "balance beams", "UB": "uneven bars"}[sport_label]
        text_label = data_split[-1][6:]
        categories[clabel] = (sport_label, set_label, text_label)

set_labels = {}
with open(os.path.join(root, 'categories/set_categories.txt'), 'r') as f:
    for line in f:
        data_split = line.split(';')
        set_label = int(data_split[0].split(' ')[-1])
        name = data_split[-1][1:].replace('\n', '')
        set_labels[set_label] = name

event_labels = {
    1: 'vault (w)',
    2: 'floor (w)',
    3: 'balance beams (w)',
    4: 'uneven bars (w)',
    5: 'vault (m)',
    6: 'floor (m)',
    7: 'pommel horse (m)',
    8: 'still rings (m)',
    9: 'parallel bars (m)',
    10: 'horizontal bar (m)'
}

events_with_actions = {
    "vault": ["1"],
    "balance beams": ["21", "22", "23", "24", "25"],
    "floor": ["31", "32", "33", "34", "35"],
    "uneven bars": ["41", "42", "43", "44"]
}

for video_id, events in annotations.items():
    for event_id, event_data in events.items():
        event_label = event_data['event']
        event_timestamp = event_data['timestamps']
        name_clip = video_id + '_' + event_id
        path_clip = os.path.join(root, 'event_videos', f'{name_clip}.mp4')
        if os.path.isfile(path_clip):
            data.append([f'event_videos/{name_clip}.mp4', event_label, 0, None])

        if event_data['segments'] is not None:
            for segment_id, segment_data in event_data['segments'].items():
                name_subclip = video_id + '_' + event_id + '_' + segment_id
                path_subclip = os.path.join(root, 'action_videos', f'{name_subclip}.mp4')
                if os.path.isfile(path_subclip):
                    if name_subclip in labels_train:
                        action_label = labels_train[name_subclip]
                        train_test = 1
                    elif name_subclip in labels_test:
                        action_label = labels_test[name_subclip]
                        train_test = 0
                    else:
                        continue
                    # sport_label, set_label, text_label = categories[action_label]
                    # set_name = set_labels[set_label]
                    data.append([f'action_videos/{name_subclip}.mp4', action_label, 1, train_test])

random.shuffle(data)

final_path = '/proj/vondrick/didac/www/hyperbolic/finegym/'

# Create three json files with the previous information and a txt with the examples
with open(os.path.join(final_path, 'finegym_categories.json'), 'w') as f:
    json.dump(categories, f)

with open(os.path.join(final_path, 'finegym_sets.json'), 'w') as f:
    json.dump(set_labels, f)

with open(os.path.join(final_path, 'finegym_events.json'), 'w') as f:
    json.dump(event_labels, f)

with open(os.path.join(final_path, 'finegym_events_actions.json'), 'w') as f:
    json.dump(events_with_actions, f)

with open(os.path.join(final_path, 'finegym_videos.txt'), 'w') as f:
    for path, class_name, k, train_test in data:
        f.writelines(f'{path} {class_name} {k} {train_test}\n')
