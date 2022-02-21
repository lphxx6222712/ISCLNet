import cv2
import numpy as np
import os
import os.path as osp
from os.path import exists
import numpy as np
import random
import imutils
def gen_point_label(img_path, image_path):
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

                    cnts = cv2.findContours(label_x.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

                    #cv2.imwrite('label_before.jpg',label)
                    for c in cnts:
                        M = cv2.moments(c)
                        try:
                            cY = int(M["m10"]/M["m00"])
                            cX = int(M["m01"]/M["m00"])
                        except:
                            print(M)

                    #     cv2.drawContours(label,[c],-1,(0,255,0),2)
                    #     cv2.circle(label,(cX,cY),7,(255,225,225),-1)
                    #     cv2.putText(label,"center",(cX-20,cY-20),
                    #                 cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
                    # cv2.imwrite('label_center_%d.jpg'%i,label)

                    random_point = [cX, cY]
                    point_label[random_point[0],random_point[1]] = l
                    point_label[random_point[0], random_point[1]+1] = l
                    point_label[random_point[0], random_point[1]-1] = l
                    point_label[random_point[0]+1, random_point[1]] = l
                    point_label[random_point[0]-1, random_point[1]] = l
                    point_label[random_point[0]+1, random_point[1]+1] = l
                    point_label[random_point[0] + 1, random_point[1] -1] = l
                    point_label[random_point[0] - 1, random_point[1] + 1] = l
                    point_label[random_point[0] - 1, random_point[1] - 1] = l
            save_dir = osp.join(image_path, pid)
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(osp.join(save_dir, '%d.bmp' % (i + 1)), point_label)


def main():
    root_list = ['/media/hxx/9D76-E202/dataset/RETOUCH/RETOUCH-TrainingSet-Cirrus/scans',
                 '/media/hxx/9D76-E202/dataset/RETOUCH/RETOUCH-TrainingSet-Spectralis/scans',
                 '/media/hxx/9D76-E202/dataset/RETOUCH/RETOUCH-TrainingSet-Topcon/scans'
                 ]
    for root in root_list:
        gen_point_label(osp.join(root, 'label'), osp.join(root, 'sequential_point_label'))
        print('dataset #{} has been generated!'.format(root))


if __name__ == '__main__':
    main()