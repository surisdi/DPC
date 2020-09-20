"""
Cut FineGym dataset videos into clips and sublicps.
Each clip contains the whole exercise (event)
Each subclip contains a move/step (segment, action)
"""

import os
import json
import subprocess
from multiprocessing import Pool

folder_dataset = '/proj/vondrick/datasets/FineGym'
extract_event = False  # otherwise, extract segments from previously extracted events


def main():
    with open(os.path.join(folder_dataset, 'annotations/finegym_annotation_info_v1.1.json'), 'r') as f:
        annotations = json.load(f)

    pool = Pool(processes=30)
    pool.map(process_video, annotations.items())


def process_video(inputs):
    video_id, events = inputs
    timestamps = []
    paths_new = []
    paths_original = []
    path_original_video = os.path.join(folder_dataset, 'videos', video_id, f'{video_id}_reduced.mp4')
    for event_id, event_data in events.items():
        event_label = event_data['event']
        event_timestamp = event_data['timestamps']
        name_clip = video_id + '_' + event_id
        path_clip = os.path.join(folder_dataset, 'event_videos', f'{name_clip}.mp4')
        if extract_event:
            paths_original.append(path_original_video)
            paths_new.append(path_clip)
            timestamps.append(event_timestamp[0])
        elif event_data['segments'] is not None:
            for segment_id, segment_data in event_data['segments'].items():
                name_subclip = video_id + '_' + event_id + '_' + segment_id
                path_subclip = os.path.join(folder_dataset, 'action_videos', f'{name_subclip}.mp4')
                if len(segment_data['timestamps']) > 1:  # this is only to extract the clips with more than 1 stage that were extracted incorrectly before
                    paths_original.append(path_clip)
                    paths_new.append(path_subclip)
                    ts = segment_data['timestamps']
                    timestamps.append([ts[0][0], ts[len(ts)-1][1]])

    extract_video(paths_original, paths_new, timestamps)


def extract_video(paths_original, paths_new, timestamps):
    for path_original, path_new, timestamp in zip(paths_original, paths_new, timestamps):
        if os.path.isfile(path_original):  # and not os.path.isfile(path_new):
            # -y overwrites
            instruction = f'ffmpeg -y -i {path_original} -ss {timestamp[0]} -to {timestamp[1]} -c:v libx264 -c:a copy {path_new}'
            subprocess.call(instruction, shell=True)


if __name__ == '__main__':
    main()
