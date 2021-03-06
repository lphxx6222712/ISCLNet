import argparse
import json
import numpy as np
from PIL import Image
from os import path as osp


def compute_mean_std(data_dir, list_dir):
    image_list_path = osp.join(list_dir, 'train_images.txt')
    image_list = [line.strip() for line in open(image_list_path, 'r')]
    np.random.shuffle(image_list)
    pixels = []
    for i, image_path in enumerate(image_list):
        print(i)
        image = Image.open(osp.join(image_path), 'r')
        pixels.append(np.asarray(image).reshape(-1, 3))
    pixels = np.vstack(pixels)
    mean = np.mean(pixels, axis=0) / 255
    std = np.std(pixels, axis=0) / 255
    print(mean, std)
    info = {'mean': mean.tolist(), 'std': std.tolist()}
    with open(osp.join(list_dir, 'info.json'), 'w') as fp:
        json.dump(info, fp)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute mean and std of a dataset.')
    parser.add_argument('-d', '--data-dir', default=None,
                        help='data folder where train_images.txt resides.')
    parser.add_argument('-l', '--list-dir',
                        default='/home/hxx/Documents/my_code/RETOUCH-seed-region-growing-weakly-supervised-segmentation/data/data_path',
                        help='data folder where train_images.txt resides.')
    args = parser.parse_args()
    if args.list_dir is None:
        args.list_dir = args.data_dir
    return args


def main():
    args = parse_args()
    compute_mean_std(args.data_dir, args.list_dir)


if __name__ == '__main__':
    main()
