import argparse
import glob

from PIL import Image
from tqdm import tqdm


def make_gif(path):
    # Set the output video file name
    output_gif_file = f'{path}/output.gif'
    photo_files = []
    photo_files.append(path + '/' + '1.png')
    photo_files.append(path + '/' + '2.png')
    photo_files.append(path + '/' + '3.png')
    photo_files.append(path + '/' + '4.png')
    photo_files.append(path + '/' + '5.png')
    photo_files.append(path + '/' + '6.png')
    photo_files.append(path + '/' + '7.png')
    photo_files.append(path + '/' + '8.png')
    photo_files.append(path + '/' + '9.png')

    frames = [Image.open(fn) for fn in photo_files]

    frame_one = frames[0]
    frame_one.save(output_gif_file, format="GIF", append_images=frames,
               save_all=True, duration=50)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)

    args = parser.parse_args()

    make_gif(args.path)