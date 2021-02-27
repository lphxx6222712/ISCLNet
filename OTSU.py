import os
import os.path as osp
import cv2
import time

def OTSU(filepath,outputpath):


    for j, pid in enumerate(sorted(os.listdir(filepath))):
        if pid.startswith('.'):
            continue
        pic_list = sorted(os.listdir(osp.join(filepath, pid)), key=lambda x: int(x[:-4]))
        start = time.time()
        for i, pic_name in enumerate(pic_list):

            img_path = os.path.join(filepath,pid, pic_list[i])
            img = cv2.imread(img_path,0)

            blur = cv2.GaussianBlur(img, (5, 5), 0)
            ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            save_dir = osp.join(outputpath, pid)
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
        now = time.time()
        print('time:%.4f'%(now-start))
        cv2.imwrite(osp.join(save_dir, '%s.bmp' % (i+1)), th)
        print('%d/%d'%((j+1),len(os.listdir(filepath))))
        print('saved')

if __name__ == '__main__':
    root_list = [
        'trainingset/',
        'validationset']
    for root in root_list:
        OTSU(osp.join(root, 'original_images'), osp.join(root, 'Threshold_images'))
        print('dataset #{} has been generated!'.format(root))

