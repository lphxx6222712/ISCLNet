import cv2
import numpy as np
import os
import os.path as osp
from os.path import exists
import numpy as np
import random

def gen_point_label(label_ratio, img_path, image_path):
    for j, pid in enumerate(sorted(os.listdir(img_path))):
        print(j)
        if pid.startswith('.'):
            continue
        pic_list = sorted(os.listdir(osp.join(img_path, pid)), key=lambda x: int(x[:-4]))
        for i, pic_name in enumerate(pic_list):

            label = cv2.imread(osp.join(img_path, pid, pic_list[i]),0)
            labels = np.unique(label)
            point_label = np.zeros_like(label)
            if np.sum(labels):
                for l in labels[1:]:
                    label_x = np.where(label==l,label,0)
                    index = np.where(label_x==l)
                    len_index = len(index[0])
                    random_points = np.random.randint(0,len_index-1,int(len_index*label_ratio))
                    for random_point in random_points:
                        #random_point = random.randint(0,len_index-1)
                        random_point = [index[0][random_point], index[1][random_point]]
                        point_label[random_point[0], random_point[1]] = l
                        # try:
                        #     point_label[random_point[0], random_point[1]+1] = l
                        # except:
                        #     print(random_point)
                        #
                        # point_label[random_point[0], random_point[1]-1] = l
                        # try:
                        #     point_label[random_point[0]+1, random_point[1]] = l
                        # except:
                        #     print(random_point)
                        # point_label[random_point[0]-1, random_point[1]] = l
                        # try:
                        #     point_label[random_point[0]+1, random_point[1]+1] = l
                        # except:
                        #     print(random_point)
                        #
                        # try:
                        #     point_label[random_point[0] + 1, random_point[1] -1] = l
                        # except:
                        #     print(random_point)
                        #
                        # try:
                        #     point_label[random_point[0] - 1, random_point[1] + 1] = l
                        # except:
                        #     print(random_point)

                        #point_label[random_point[0] - 1, random_point[1] - 1] = l
            save_dir = osp.join(image_path, pid)
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(osp.join(save_dir, '%d.bmp' % (i + 1)), point_label)


def main():
    label_ratio = [0.01,0.05,0.1,0.2,0.3,0.4,0.5]
    root_list = ['/media/hxx/9D76-E202/dataset/RETOUCH/RETOUCH-TrainingSet-Cirrus/scans',
                 '/media/hxx/9D76-E202/dataset/RETOUCH/RETOUCH-TrainingSet-Spectralis/scans',
                 '/media/hxx/9D76-E202/dataset/RETOUCH/RETOUCH-TrainingSet-Topcon/scans']
    for label_ra in label_ratio:
        for root in root_list:
            gen_point_label(label_ra, osp.join(root, 'label'), osp.join(root, 'point_label_%0.2f'%label_ra))
            print('dataset #{} has been generated!'.format(root))


if __name__ == '__main__':
    main()
