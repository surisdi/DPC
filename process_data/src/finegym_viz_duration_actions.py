import os
import json
import matplotlib.pyplot as plt
import numpy as np
import itertools

folder_dataset = '/proj/vondrick/datasets/FineGym'


def main():
    with open(os.path.join(folder_dataset, 'annotations/finegym_annotation_info_v1.1.json'), 'r') as f:
        annotations = json.load(f)

    total_result = []
    for video_id, events in annotations.items():
        number = []
        for event_id, event_data in events.items():
            if event_data['segments'] is not None:
                # Optional, if we want to filter out videos that we are not going to use during training
                if len(event_data['segments']) > 6:
                    for segment_id, segment_data in event_data['segments'].items():
                        number.append(segment_data['timestamps'][0][1] - segment_data['timestamps'][0][0])

        total_result.append(number)

    total_result = np.array(list(itertools.chain.from_iterable(total_result)))

    n, bins, patches = plt.hist(x=total_result, bins=100, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.title('Duration of actions (s)')
    plt.savefig('/proj/vondrick/didac/results/hist_timestamps_6actions.jpg')


if __name__ == '__main__':
    main()
