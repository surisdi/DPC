import os

input_folder = '/proj/vondrick/datasets/Hollywood2/AVIClips'  # '/proj/vondrick/datasets/Hollywood2/AVIClips'
output_folder = '/proj/vondrick/datasets/Hollywood2/AVIClips_mp4'  # '/proj/vondrick/datasets/Hollywood2/AVIClips_mp4'

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    full_path = os.path.join(input_folder, file)
    if os.path.isfile(full_path):
        file_output = os.path.join(output_folder, file.replace('.avi', '.mp4'))
        os.system(f'ffmpeg -i {full_path} {file_output}')
