from torch.utils.data import Dataset
from os.path import join, exists
from PIL import Image
import torch
import os
import numpy as np
import pandas as pd


class datalist(Dataset):
    def __init__(self, data_dir, mode, phase, transforms_OCT, transforms_fundus, transforms_ROI,
                 cross, start_from=0, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms_OCT = transforms_OCT
        self.transforms_fundus = transforms_fundus
        self.transforms_ROI = transforms_ROI
        self.image_list_OCT = None
        self.image_list_fundus = None
        self.image_list_ROI = None
        self.mode = mode
        self.read_lists(cross, start_from)
        self.simple_label = pd.read_csv(
            '/media/hxx/9D76-E202/dataset/Multi-modal-retinal-image/training/multi-modal-label-eye-levelv3.csv'
        ).fillna(0).iloc[:, :].values
        #print(self.image_list)
    def __getitem__(self, index):
        if self.phase == 'train':

            path_OCT = join(self.data_dir, self.image_list_OCT[index])
            path_fundus = join(self.data_dir, self.image_list_fundus[index])
            path_ROI = join(self.data_dir, self.image_list_ROI[index])
            # print(path)
            # print(index)
            # print(self.data_dir, self.image_list[index])
            data_OCT = [Image.open(path_OCT)]
            data_OCT = list(self.transforms_OCT(*data_OCT))[0]
            data_fundus = [Image.open(path_fundus)]
            data_fundus = list(self.transforms_OCT(*data_fundus))[0]
            data_ROI = [Image.open(path_ROI)]
            data_ROI = list(self.transforms_OCT(*data_ROI))[0]

            id = os.path.split(path_OCT)[1].split('_')[0]
            eye = os.path.split(path_OCT)[1].split('_')[1]

            informations = self.simple_label[np.where(self.simple_label[:, 0] == int(id))]

            if informations.shape[0] == 1:
                if 'ou' in informations:
                    information = informations[0]
                    gender = information[2]
                    age = information[3]
                    label = information[4:]
                    label = torch.from_numpy(label.astype('float32'))
                elif eye in informations:
                    information = informations[0]
                    gender = information[2]
                    age = information[3]
                    label = information[4:]
                    label = torch.from_numpy(label.astype('float32'))
                else:
                    print(eye, informations, path_OCT, path_fundus)
                    #print('no',eye,'in', informations)
                    return

            if informations.shape[0] == 2:
                if eye in informations[0]:
                    information = informations[0]
                    gender = information[2]
                    age = information[3]
                    label = information[4:]
                    label = torch.from_numpy(label.astype('float32'))
                elif eye in informations[1]:
                    information = informations[1]
                    gender = information[2]
                    age = information[3]
                    label = information[4:]
                    label = torch.from_numpy(label.astype('float32'))
                else:
                    print(eye, informations, path_OCT, path_fundus)
                    #print('no', eye, 'in', informations)
                    return


            return id, data_OCT, data_fundus, data_ROI, label, gender, age, path_OCT, path_fundus

        if self.phase == 'val':
            path_OCT = join(self.data_dir, self.image_list_OCT[index])
            path_fundus = join(self.data_dir, self.image_list_fundus[index])
            path_ROI = join(self.data_dir, self.image_list_ROI[index])
            # print(path)
            # print(index)
            # print(self.data_dir, self.image_list[index])
            data_OCT = [Image.open(path_OCT)]
            data_OCT = list(self.transforms_OCT(*data_OCT))[0]
            data_fundus = [Image.open(path_fundus)]
            data_fundus = list(self.transforms_OCT(*data_fundus))[0]
            data_ROI = [Image.open(path_ROI)]
            data_ROI = list(self.transforms_OCT(*data_ROI))[0]

            # print(data.shape)
            id = os.path.split(path_OCT)[1].split('_')[0]
            eye = os.path.split(path_OCT)[1].split('_')[1]

            # print(id)
            '''discriminate os and od'''
            informations = self.simple_label[np.where(self.simple_label[:, 0] == int(id))]

            if informations.shape[0] == 1:
                if 'ou' in informations:
                    information = informations[0]
                    gender = information[2]
                    age = information[3]
                    label = information[4:]

                elif eye in informations:
                    information = informations[0]
                    gender = information[2]
                    age = information[3]
                    label = information[4:]
                else:
                    print(eye, informations)
                    return

            if informations.shape[0] == 2:
                if eye in informations[0]:
                    information = informations[0]
                    gender = information[2]
                    age = information[3]
                    label = information[4:]

                elif eye in informations[1]:
                    information = informations[1]
                    gender = information[2]
                    age = information[3]
                    label = information[4:]


            label = torch.from_numpy(label.astype('float32'))


            # print(label.shape)
            # print(id, path)

            return id, data_OCT, data_fundus,data_ROI, label, gender, age, path_OCT, path_fundus


    def __len__(self):
        return len(self.image_list_OCT)

    def read_lists(self, cross, start_from):
        if self.mode == 'train':
            image_path_OCT = join(self.list_dir, self.phase + '_'  + 'OCT' + '_images_cross_%d.txt' % cross)
            image_path_fundus = join(self.list_dir, self.phase + '_' + 'fundus' + '_images_cross_%d.txt' % cross)
            image_path_ROI = join(self.list_dir, self.phase + '_' + 'OCT_ROI' + '_images_cross_%d.txt' % cross)
        if self.mode == 'val':
            image_path_OCT = join(self.list_dir, self.phase + '_' + 'OCT' + '_images_cross_%d.txt' % cross)
            image_path_fundus = join(self.list_dir, self.phase + '_' + 'fundus' + '_images_cross_%d.txt' % cross)
            image_path_ROI = join(self.list_dir, self.phase + '_' + 'OCT_ROI' + '_images_cross_%d.txt' % cross)

        self.image_list_OCT = [line.strip() for line in open(image_path_OCT, 'r')][start_from:]
        self.image_list_fundus = [line.strip() for line in open(image_path_fundus, 'r')][start_from:]
        self.image_list_ROI = [line.strip() for line in open(image_path_ROI, 'r')][start_from:]

        if self.phase == 'train':
            print('Total train OCT image is : %d' % len(self.image_list_OCT))
            print('Total train fundus image is : %d' % len(self.image_list_fundus))
            print('Total train ROI image is : %d' % len(self.image_list_ROI))
        else:
            print('Total val OCT image is : %d' % len(self.image_list_OCT))
            print('Total val fundus image is : %d' % len(self.image_list_fundus))
            print('Total val ROI image is : %d' % len(self.image_list_ROI))