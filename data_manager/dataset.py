import cv2
import numpy as np
import os
import torch.utils.data as data
from .transformer import  *
import torchvision.transforms as transforms
import random
from PIL import Image


class dataset_Aptos(data.Dataset):
    def __init__(self, data_path, DF, transform = None, need_img_path=False):

        self.data_path = data_path
        self.transform = transform
        self.cls_num = 5
        self.DF = DF
        self.img_path = []
        self.labels = []
        self.need_img_path = need_img_path
        for index in range(len(self.DF)):
                self.img_path.append(self.DF[index, 0])
                self.labels.append(self.DF[index, 1])
        self.CLASSES = ['None', 'Mild', 'Moderate', 'Severe', 'PDR']

    def get_cls_num_list(self):
        cls_num_list = []
        for tempLabel in range(self.cls_num):
            cls_num_list.append(np.sum(np.array(self.DF[:,1])==tempLabel))
        return cls_num_list

    def __getitem__(self, index):

        try:
            imgName = os.path.join(self.data_path, self.DF[index, 0])
            imgName = imgName + '.png'
            imgName = imgName.replace('\\', '/')

            Img = Image.open(imgName)
            # Img = cv2.imread(imgName)
            # Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
            # Img = transforms.ToPILImage()(Img)

            if self.transform is not None:
                Img = self.transform(Img)

            # Get the Labels
            label = self.DF[index, 1]
            label_onehot = np.zeros(5)
            label_onehot[label] = 1
            
        except:
            index = 0
            imgName = os.path.join(self.data_path, self.DF[index, 0])
            imgName = imgName + '.png'
            imgName = imgName.replace('\\', '/')

            Img = Image.open(imgName)
            # Img = cv2.imread(imgName)
            # Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
            # Img = transforms.ToPILImage()(Img)
            if self.transform is not None:
                Img = self.transform(Img)

            # Get the Labels
            label = self.DF[index, 1]
            label_onehot = np.zeros(5)
            label_onehot[label] = 1
        if self.need_img_path:
            return Img, label, label_onehot, imgName
        else:
            return Img, label, label_onehot

    def __len__(self):
        return len(self.DF)

class dataset_DDR(data.Dataset):
    def __init__(self, data_path, DF, transform = None, need_img_path=False):

        self.data_path = data_path
        self.transform = transform
        self.cls_num = 5
        self.DF = DF
        self.img_path = []
        self.labels = []
        self.need_img_path = need_img_path
        for index in range(len(self.DF)):
                self.img_path.append(self.DF[index, 0])
                self.labels.append(self.DF[index, 1])
        self.CLASSES = ['None', 'Mild', 'Moderate', 'Severe', 'PDR']

    def get_cls_num_list(self):
        cls_num_list = []
        for tempLabel in range(self.cls_num):
            cls_num_list.append(np.sum(np.array(self.DF[:,1])==tempLabel))
        return cls_num_list

    def __getitem__(self, index):

        try:
            imgName = os.path.join(self.data_path, self.DF[index, 0])
            imgName = imgName
            imgName = imgName.replace('\\', '/')

            Img = Image.open(imgName)
            # Img = cv2.imread(imgName)
            # Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
            # Img = transforms.ToPILImage()(Img)

            if self.transform is not None:
                Img = self.transform(Img)

            # Get the Labels
            label = self.DF[index, 1]
            label_onehot = np.zeros(5)
            label_onehot[label] = 1
            
        except:
            index = 0
            imgName = os.path.join(self.data_path, self.DF[index, 0])
            imgName = imgName
            imgName = imgName.replace('\\', '/')

            Img = Image.open(imgName)
            # Img = cv2.imread(imgName)
            # Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
            # Img = transforms.ToPILImage()(Img)
            if self.transform is not None:
                Img = self.transform(Img)

            # Get the Labels
            label = self.DF[index, 1]
            label_onehot = np.zeros(5)
            label_onehot[label] = 1
        if self.need_img_path:
            return Img, label, label_onehot, imgName
        else:
            return Img, label, label_onehot

    def __len__(self):
        return len(self.DF)
class dataset_Messidor2(data.Dataset):
    def __init__(self, data_path, DF, transform = None):

        self.data_path = data_path
        self.transform = transform
        self.DF = DF
        self.cls_num = 5
        self.img_path = []
        self.labels = []
        for index in range(len(self.DF)):
                self.img_path.append(self.DF[index, 0])
                self.labels.append(self.DF[index, 1])
        self.CLASSES = ['None', 'Mild', 'Moderate', 'Severe', 'PDR']

    def get_cls_num_list(self):
        cls_num_list = []
        for tempLabel in range(self.cls_num):
            cls_num_list.append(np.sum(np.array(self.DF[:,1])==tempLabel))
        return cls_num_list

    def __getitem__(self, index):

        try:
            imgName = os.path.join(self.data_path, self.DF[index, 0])
            imgName = imgName.replace('\\', '/')

            Img = Image.open(imgName)
            if self.transform is not None:
                Img = self.transform(Img)

            # Get the Labels
            label = self.DF[index, 1]
            label_onehot = np.zeros(5)
            label_onehot[label] = 1
            
        except:
            print(index)
            index = 0
            imgName = os.path.join(self.data_path, self.DF[index, 0])
            imgName = imgName.replace('\\', '/')

            Img = Image.open(imgName)
            if self.transform is not None:
                Img = self.transform(Img)

            # Get the Labels
            label = self.DF[index, 1]
            label_onehot = np.zeros(5)
            label_onehot[label] = 1

        return Img, label, label_onehot

    def __len__(self):
        return len(self.DF)

class dataset_EyePACS(data.Dataset):
    def __init__(self, data_path, DF, transform = None):

        self.data_path = data_path
        self.transform = transform
        self.DF = DF
        self.img_path = []
        self.labels = []
        for index in range(len(self.DF)):
                self.img_path.append(self.DF[index, 0])
                self.labels.append(self.DF[index, 1])

    def __getitem__(self, index):

        imgName = os.path.join(self.data_path, self.DF[index, 0])
        imgName = imgName + '.jpeg'
        imgName = imgName.replace('\\', '/')

        Img = Image.open(imgName)
        # Img = cv2.imread(imgName)
        # Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
        # Img = transforms.ToPILImage()(Img)

        if self.transform is not None:
            Img = self.transform(Img)

        # Get the Labels
        label = self.DF[index, 1]
        label_onehot = np.zeros(5)
        label_onehot[label] = 1
            

        return Img, label, label_onehot

    def __len__(self):
        return len(self.DF)


class dataset_RFMiD(data.Dataset):
    def __init__(self, data_path, DF, transform = None):

        self.data_path = data_path
        self.transform = transform
        self.DF = DF


    def __getitem__(self, index):

        try:
            imgName = os.path.join(self.data_path, str(self.DF.loc[index, 'ID']))
            imgName = imgName + '.png'
            imgName = imgName.replace('\\', '/')

            Img = Image.open(imgName)
            # Img = cv2.imread(imgName)
            # Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
            # Img = transforms.ToPILImage()(Img)

            if self.transform is not None:
                Img = self.transform(Img)
            label = self.DF.loc[index, 'Disease_Risk']

            
        except:
            index = 0
            imgName = os.path.join(self.data_path, str(self.DF.loc[index, 'ID']))
            imgName = imgName + '.png'
            imgName = imgName.replace('\\', '/')

            Img = Image.open(imgName)
            # Img = cv2.imread(imgName)
            # Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
            # Img = transforms.ToPILImage()(Img)

            if self.transform is not None:
                Img = self.transform(Img)

            # Get the Labels
            label = self.DF.loc[index, 'Disease_Risk']
            # label_onehot = np.zeros(2)
            # label_onehot[label] = 1

        return Img, label

    def __len__(self):
        return len(self.DF)

class dataset(data.Dataset):
    def __init__(self, DF, transform = None):

        self.transform = transform
        self.DF = DF
        self.image_label = []
        self.image_path = []


    def __getitem__(self, index):

        try:
            imgName = self.DF[index, 0]
            imgName = imgName.replace('\\', '/')
            self.image_path.append(imgName)

            Img = Image.open(imgName)

            if self.transform is not None:
                Img = self.transform(Img)
            label = self.DF[index, 1]
            self.image_label.append(label)
   
        except:
            imgName = self.DF[index, 0]
            imgName = imgName.replace('\\', '/')
            self.image_path.append(imgName)

            Img = Image.open(imgName)

            if self.transform is not None:
                Img = self.transform(Img)
            label = self.DF[index, 1]
            self.image_label.append(label)

        return Img, label, imgName

    def __len__(self):
        return len(self.DF)

