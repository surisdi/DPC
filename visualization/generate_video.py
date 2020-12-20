import subprocess
from subprocess import DEVNULL, STDOUT
import sys

import seaborn as sns
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip

import generate_tree_video


def main():
    # Paths
    dataset = 'finegym'  # 'finegym'
    split = 'test'  # 'val', 'train'
    path_data = f'/proj/vondrick/didac/results/extracted_features_{dataset}_{split}.pth'

    a = torch.load(path_data, map_location=torch.device('cpu'))

    indexes_good = [index for index in range(len(a[0])) if a[0][index][-1] == a[3][index][0]]

    folder_final_video = f'/proj/vondrick/shared/hypvideo/created_videos/{dataset}/{split}'
    os.makedirs(folder_final_video, exist_ok=True)
    max_videos = 5
    for index in indexes_good[1:max_videos]:
        event_name = a[5][index][:-12] if dataset == "finegym" else a[5][index].split('/')[-1]
        path_final_video = os.path.join(folder_final_video, f"{index}_{event_name}.mp4")

        create_video_index(index, a, dataset, path_final_video)


def create_video_index(index, a, dataset, path_final_video):
    # provisional_folder = f'/proj/vondrick/didac/code/DPC/create_video_{index}'
    provisional_folder = f'/tmp/create_video_{index}'
    os.makedirs(provisional_folder, exist_ok=True)

    num_actions = 6

    pred_1 = a[0][index]
    pred_2 = a[1][index]
    pred_3 = a[2][index]
    gt = a[3][index]
    radius = a[4][index]
    action_name = a[5][index]
    info_video = a[6][index]
    percentage = [a[7][i][index] for i in range(len(a[7]))]

    percentiles = []
    for val in {'finegym': [33, 66], 'hollywood2': [50]}[dataset]:
        percentiles.append(np.percentile(a[4].view(-1).numpy(), val))

    ########## VIDEO ##########

    if dataset == "finegym":
        root_finegym = f'/proj/vondrick/datasets/FineGym/'

        info_keys = list(info_video[0].keys())
        last_index = info_keys.index(action_name[-11:])
        used_keys = info_keys[last_index-num_actions+1:last_index+1]
        info_video_actions = {k: info_video[0][k] for k in used_keys}

        time_init = float(info_video_actions[list(info_video_actions.keys())[0]]['timestamps'][0][0])
        times_end = np.array([float(v['timestamps'][0][-1])-time_init for v in info_video_actions.values()])

        actions = list(info_video_actions.keys())
        actions.sort()

        event_name = action_name[:-12]
        video_name = action_name[:11]

        start_actions = info_video_actions[actions[0]]['timestamps'][0][0]
        end_actions = info_video_actions[actions[-1]]['timestamps'][-1][-1]

        # This is for low-quality videos
        # start_video = start_actions
        # end_video = end_actions
        # path_video = os.path.join(root_finegym, 'event_videos', f'{event_name}.mp4')

        # This is for high quality videos
        start_event, end_event = event_name[14:].split('_')
        start_video = int(start_event) + start_actions
        end_video = int(start_event) + end_actions + 1  # extra second to have time to visualize the last prediction
        path_video = os.path.join(root_finegym, 'videos', f'{video_name}/{video_name}.mp4')
        path_input_video = f"{provisional_folder}/{event_name}_{dataset}.mp4"
        ffmpeg_extract_subclip(path_video, start_video, end_video, targetname=path_input_video)

        # fps = get_frame_rate(path_input_video)
        clip = VideoFileClip(path_input_video)
        fps = clip.fps

    else:
        root_hollywood2 = f'/proj/vondrick/datasets/Hollywood2/'
        event_name = action_name.split('/')[-1]
        path_video = os.path.join(root_hollywood2, 'AVIClips', f'{event_name}.avi')
        clip = VideoFileClip(path_video)
        duration = clip.duration
        fps = clip.fps
        # n_frames = round(fps*duration)  # not needed
        times_end = (info_video[0][:, -1]-info_video[0][0, 0])/fps
        start_video = np.max([0, info_video[0][0, 0]/fps - 0.5])
        end_video = np.min([duration, info_video[0][-1, -1]/fps + 0.5])
        path_input_video_avi = f"{provisional_folder}/{event_name}_{dataset}.avi"
        path_input_video = f"{provisional_folder}/{event_name}_{dataset}.mp4"

        ffmpeg_extract_subclip(path_video, start_video, end_video, targetname=path_input_video_avi)

        # Convert avi to mp3
        os.system(f"ffmpeg -i {path_input_video_avi} -c:v copy -c:a copy -y {path_input_video}")

    if clip.h < 720:
        path_input_video_resized = f"{provisional_folder}/{event_name}_{dataset}_resized.mp4"
        # Dropping the audio because I do not want to think of a codec for it
        os.system(f"ffmpeg -i {path_input_video} -vf scale={2 * (int(clip.w * 720 / clip.h) // 2)}x720 -c:v libx264 "
                  f"-an -y {path_input_video_resized}")
        path_input_video = path_input_video_resized

    print('Initial video done')

    ############# BOTTOM PLOT #############

    path_input_plot = f'{provisional_folder}/plot_video_{dataset}.mp4'

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'))

    fig, ax = plt.subplots(dpi=400)
    fig.set_figheight(3)
    fig.set_figwidth(10)
    fig.set_tight_layout(True)

    # ax.get_xaxis().set_visible(False)

    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ylim_min = a[4][index].min() - 0.02
    ylim_max = a[4][index].max() + 0.02

    ax.set_ylim(ylim_min, ylim_max)
    plt.tick_params(
        axis='both',  # changes apply to both axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right=False,
        left=False)  # labels along the bottom edge are off

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    size_ball = 130
    points_all = [None] + list(radius.numpy())

    last_num_points = -1

    def animate_plot(i):
        "The function to call at each frame. Basically what to plot at every frame"

        second = i/fps
        num_points = (np.array(times_end) < second).sum()

        # We would use this if we do not want to update the graph with time marker
        # global last_num_points
        # if num_points != last_num_points:
        # last_num_points = num_points

        points = points_all[:num_points+1]

        ax.clear()
        ax.invert_yaxis()
        ax.set_xlim(-1, end_video-start_video + 1)
        ax.set_facecolor('white')

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Poincare radius', fontsize=12)
        ax.set_xticks([])

        for p in percentiles:
            ax.axhline(linewidth=3, color=(128 / 255., 0 / 255., 200 / 255., 0.9), y=p, linestyle=':', zorder=0)

        # Time marker
        ax.plot((second, second), (ylim_min, ylim_max), linestyle='-', color='b', alpha=0.6, zorder=0)

        point_old = None
        for j, point in enumerate(points):
            if point is None:
                continue
            ax.plot((times_end[j-1], times_end[j-1]), (ylim_min, ylim_max), linestyle='--', color='k', alpha=0.3, zorder=0)
            ax.scatter([times_end[j-1]], [point], color='#ff57f0ff', s=size_ball)
            # This was to create the line only when the new point is already there
            # if point_old is not None:
            #     plt.arrow(times_end[j-2], point_old, times_end[j-1]-times_end[j-2], point - point_old, head_width=0,
            #               head_length=0, zorder=0, color='k')

            if j < len(points_all)-1:
                final_time = np.min([second, times_end[j]])
                plt.arrow(times_end[j - 1], point, final_time - times_end[j - 1],
                          (points_all[j+1] - point)*(final_time-times_end[j-1])/(times_end[j]-times_end[j - 1]),
                          head_width=0, head_length=0, zorder=0, color='k')

            point_old = point

    ani = matplotlib.animation.FuncAnimation(fig, animate_plot, frames=int(fps*(end_video-start_video)),
                                             repeat=True)
    ani.save(path_input_plot, writer=writer, dpi=500)
    # plt.clf()

    print('Bottom plot done')

    ############## HIERARCHY TREE ################

    path_input_tree = f'{provisional_folder}/hierarchy_tree_video_{dataset}.mp4'
    path_pngs = f'{provisional_folder}/pngs_{dataset}'

    if os.path.isdir(path_pngs):
        files = os.listdir(path_pngs)
        for f in files:
            f = os.path.join(path_pngs, f)
            if not os.path.isdir(f):
                os.remove(f)  # In case there are frames from previous videos

    num_frames = int(fps*(end_video-start_video))
    os.makedirs(path_pngs, exist_ok=True)

    last_num_points = -1
    last_saved_image = None
    for i in range(num_frames):
        second = i/fps
        num_points = (np.array(times_end) < second).sum()
        if num_points > last_num_points:  # There is a new prediction
            last_num_points = num_points
            if dataset == 'hollywood2':
                gt_tree = list(gt.numpy())
                if num_points == 0:
                    pred_tree = None
                    selected_level = None
                else:
                    pred_tree = [pred_1[num_points-1].item(), pred_2[num_points-1].item()]
                    selected_level = 1 if radius[num_points-1] < percentiles[0] else 2
            else:  # finegym
                gt_tree = gt[0].item()
                if num_points == 0:
                    pred_tree = None
                    selected_level = None
                else:
                    selected_level = 1 if radius[num_points - 1] < percentiles[0] else \
                        (2 if radius[num_points - 1] < percentiles[1] else 3)
                    if selected_level == 3:
                        pred_tree = pred_1[num_points-1].item()
                    elif selected_level == 2:
                        pred_tree = pred_2[num_points-1].item() - 288
                    else:  # level 1
                        pred_tree = pred_3[num_points-1].item() - 288 - 15

            percentage_pred = percentage[::-1][selected_level-1][num_points-1].item() if selected_level is not None \
                else None
            tree_video = generate_tree_video.create_figure(dataset, selected_level, gt_tree, pred_tree, percentage_pred)
            tree_video.save_image(os.path.join(path_pngs, f'tree_video_{i:05d}.png'))
            last_saved_image = os.path.join(path_pngs, f'tree_video_{i:05d}.png')
        else:
            # Create symlink
            assert last_saved_image is not None
            os.symlink(src=last_saved_image, dst=os.path.join(path_pngs, f'tree_video_{i:05d}.png'))

    # -y overwrites
    # command_ffmpeg_tree = f"ffmpeg -y -i {os.path.join(path_pngs, f'tree_video_%05d.png')} -c:v libx264 " \
    #                       f"-vf fps={fps} -pix_fmt yuv420p {path_input_tree}"
    command_ffmpeg_tree = f"ffmpeg -y -framerate {fps} -i {os.path.join(path_pngs, f'tree_video_%05d.png')} {path_input_tree}"
    # subprocess.call(command_ffmpeg_tree, stdout=DEVNULL, stderr=STDOUT, shell=True)
    os.system(command_ffmpeg_tree)

    print('Tree done')

    ############ FINAL VIDEO #############

    # Upscaling the video to the tree or graph resolution results in choppy results! Also, it would probably bee too
    # heavy. So better to create figures that are easy to visualize at a lower (~300*720) resolution

    # scale1 = "[v0][2:v]scale2ref='oh*mdar':'if(lt(main_h,ih),ih,main_h)'[v0][v2];" \
    #          "[v2][v0]scale2ref='oh*mdar':'if(lt(main_h,ih),ih,main_h)'[v2][v0];"

    # scale1 = "[2:v][v0]scale2ref='oh*mdar':ih[v2][v0];"
    # filter_complex = "[0:v]pad=iw+5:color=black[v0];" \
    #                  f"{scale1}" \
    #                  "[v0][v2]hstack[top];" \
    #                  "[top]pad=h=ih+5:color=black[top];" \
    #                  "[1:v]crop=trunc(iw/2)*2:trunc(ih/2)*2[bottom];" \
    #                  "[bottom]pad=iw*3/2:ih:0:0:color=black[bottom];" \
    #                  "[bottom][top]scale2ref=iw:'ow/mdar'[bottom][top];" \
    #                  "[bottom]crop=trunc(iw/2)*2:trunc(ih/2)*2[bottom];" \
    #                  "[top]crop=trunc(iw/2)*2:trunc(ih/2)*2[top];" \
    #                  "[top][bottom]vstack[vid]"

    filter_complex = "[0:v]pad=h=ih+5:color=black[v0];" \
                     "[1:v][v0]scale2ref=iw:'ow/mdar'[v1][v0];" \
                     "[v0][v1]vstack[left];" \
                     "[left]pad=w=iw+5:color=black[left];" \
                     "[2:v]crop=trunc(iw/2)*2:trunc(ih/2)*2[right];" \
                     "[left][right]scale2ref='oh*mdar':ih[left][right];" \
                     "[left]crop=trunc(iw/2)*2:trunc(ih/2)*2[left];" \
                     "[right]crop=trunc(iw/2)*2:trunc(ih/2)*2[right];" \
                     "[left][right]hstack[vid]"

    command_ffmpeg = f"ffmpeg -y " \
                     f"-i {path_input_video} " \
                     f"-i {path_input_plot} " \
                     f"-i {path_input_tree} " \
                     f"-filter_complex \"{filter_complex}\" " \
                     f"-map [vid] " \
                     f"-c:v libx264 " \
                     f"-pix_fmt yuv420p " \
                     f"-crf 23 " \
                     f"-preset veryfast " \
                     f"-r {fps} " \
                     f"-top 1 " \
                     f"{path_final_video}"
    # subprocess.call(command_ffmpeg, stdout=DEVNULL, stderr=STDOUT, shell=True)
    os.system(command_ffmpeg)

    sns.set(rc={'axes.facecolor': 'lightgrey', 'figure.facecolor': 'lightgrey', 'figure.edgecolor': 'black',
                'axes.grid': False})

    print('Video finished')


if __name__ == '__main__':
    main()


