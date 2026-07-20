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

class dataset_RFMiD_5fold(data.Dataset):
    def __init__(self, data_path, DF, numb, transform = None):

        self.data_path = data_path
        self.transform = transform
        self.DF = DF
        self.numb = numb

    def __getitem__(self, index):

        try:
            if int(self.DF.loc[index, 'index']) < self.numb[0]:
                imgName = os.path.join(self.data_path[0], str(self.DF.loc[index, 'ID']))
            elif (int(self.DF.loc[index, 'index']) >= self.numb[0]) & (int(self.DF.loc[index, 'index']) < self.numb[1]):
                imgName = os.path.join(self.data_path[1], str(self.DF.loc[index, 'ID']))
            else:
                imgName = os.path.join(self.data_path[2], str(self.DF.loc[index, 'ID']))
 
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

class dataset_qilu(data.Dataset):
    def __init__(self, data_path, DF, transform = None):

        self.data_path = data_path
        self.transform = transform
        self.DF = DF
        self.image_path = []
        self.image_label = []
        self.xuhao = []

        for index in range(len(self.DF)):
            Date = np.str(self.DF[index, 9])
            peple_name = self.DF[index, 3]
            img_path = os.path.join(self.data_path, Date[0:4]+ '/' + Date[0:4] + Date[5:7] + Date[8:10])

            for root, dirs, files in os.walk(img_path):
                for file in files:
                    a_test = file.split('_')[0]
                    if file.split('_')[0]==peple_name:
                        imgName = img_path + '/' + file
                        imgName = imgName.replace('\\', '/')
                        self.image_path.append(imgName)
                        self.image_label.append(self.DF[index, 7])
                        self.xuhao.append(self.DF[index, 0])
    

    def __getitem__(self, index):

        imgName = self.image_path[index]
        label = self.image_label[index]
        Img = Image.open(imgName)

        if self.transform is not None:
            Img = self.transform(Img)
        return Img, label, self.xuhao[index], imgName

    def __len__(self):
        return len(self.image_label)
    
    def image_paths(self):
        return self.image_path


class dataset_Qilu(data.Dataset):
    def __init__(self, data_path, DF, transform = None):

        self.data_path = data_path
        self.transform = transform
        self.DF = DF
        self.image_path = []
        self.image_label = []

        for index in range(len(self.DF)):
            Date = np.str(self.DF.loc[index, '上传日期'])
            peple_name = self.DF.loc[index, '患者姓名']
            img_path = os.path.join(self.data_path, Date[0:4]+ '/' + Date[0:4] + Date[5:7] + Date[8:10])

            for root, dirs, files in os.walk(img_path):
                for file in files:
                    a_test = file.split('_')[0]
                    if file.split('_')[0]==peple_name:
                        imgName = img_path + '/' + file
                        imgName = imgName.replace('\\', '/')
                        self.image_path.append(imgName)
                        self.image_label.append(self.DF.loc[index, 'lable'])
    

    def __getitem__(self, index):

        imgName = self.image_path[index]
        label = self.image_label[index]
        Img = Image.open(imgName)

        if self.transform is not None:
            Img = self.transform(Img)
        return Img, label

    def __len__(self):
        return len(self.image_label)
    
    def image_paths(self):
        return self.image_path

class dataset_DDR_2class(data.Dataset):
    def __init__(self, data_path, DF, transform = None):

        self.data_path = data_path
        self.transform = transform
        self.DF = DF


    def __getitem__(self, index):

        try:
            imgName = os.path.join(self.data_path, str(self.DF[index, 0]))
            imgName = imgName.replace('\\', '/')

            Img = Image.open(imgName)
            # Img = cv2.imread(imgName)
            # Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
            # Img = transforms.ToPILImage()(Img)

            if self.transform is not None:
                Img = self.transform(Img)
            label = self.DF[index, 1]

            
        except:
            index = 0
            imgName = os.path.join(self.data_path, str(self.DF[index, 0]))
            imgName = imgName.replace('\\', '/')

            Img = Image.open(imgName)
            # Img = cv2.imread(imgName)
            # Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
            # Img = transforms.ToPILImage()(Img)

            if self.transform is not None:
                Img = self.transform(Img)
            label = self.DF[index, 1]

        return Img, label, imgName

    def __len__(self):
        return len(self.DF)

class dataset_DDR_test_train(data.Dataset):
    def __init__(self, data_path, len_train, DF, transform = None):

        self.data_path = data_path
        self.transform = transform
        self.DF = DF
        self.len_train = len_train


    def __getitem__(self, index):

        try:
            if index<self.len_train:
                imgName = os.path.join(self.data_path[0], str(self.DF[index, 0]))
            else:
                imgName = os.path.join(self.data_path[1], str(self.DF[index, 0]))
            imgName = imgName.replace('\\', '/')

            Img = Image.open(imgName)
            # Img = cv2.imread(imgName)
            # Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
            # Img = transforms.ToPILImage()(Img)

            if self.transform is not None:
                Img = self.transform(Img)
            label = self.DF[index, 1]
            label_onehot = np.zeros(3)
            label_onehot[label] = 1

            
        except:
            index = 0
            print('index == 0')
            imgName = os.path.join(self.data_path[0], str(self.DF[index, 0]))
            imgName = imgName.replace('\\', '/')

            Img = Image.open(imgName)
            # Img = cv2.imread(imgName)
            # Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
            # Img = transforms.ToPILImage()(Img)

            if self.transform is not None:
                Img = self.transform(Img)
            label = self.DF[index, 1]
            label_onehot = np.zeros(3)
            label_onehot[label] = 1

        return Img, label, label_onehot

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

class dataset_train_DDR_QILU(data.Dataset):
    def __init__(self, data_path, len, DF, transform = None):

        self.data_path = data_path
        self.transform = transform
        self.DF = DF
        self.len = len


    def __getitem__(self, index):

        try:
            if index<self.len[0]:
                imgName = os.path.join(self.data_path[0], str(self.DF[index, 0]))
            elif index<self.len[1]:
                imgName = os.path.join(self.data_path[1], str(self.DF[index, 0]))
            elif index<self.len[2]:
                imgName = os.path.join(self.data_path[2], str(self.DF[index, 0]))
            else: 
                imgName = self.DF[index, 0]
            imgName = imgName.replace('\\', '/')

            Img = Image.open(imgName)

            if self.transform is not None:
                Img = self.transform(Img)
            label = self.DF[index, 1]

            
        except:
            index = 0
            print('index == 0')
            imgName = os.path.join(self.data_path[0], str(self.DF[index, 0]))
            imgName = imgName.replace('\\', '/')

            Img = Image.open(imgName)

            if self.transform is not None:
                Img = self.transform(Img)
            label = self.DF[index, 1]

        return Img, label, imgName

    def __len__(self):
        return len(self.DF)

class dataset_mean_std_DDR_QILU():
    def __init__(self, data_path, len, DF, transform = None):

        self.data_path = data_path
        self.transform = transform
        self.DF = DF
        self.len = len
        self.sum1 = [0, 0, 0]
        self.shape = 0
        self.sum2 = [0, 0, 0]
        self.mean = [0, 0, 0]
        self.std = [0, 0, 0]

    def compute_mean(self):
        for index in range(len(self.DF)):
            if index<self.len[0]:
                imgName = os.path.join(self.data_path[0], str(self.DF[index, 0]))
            elif index<self.len[1]:
                imgName = os.path.join(self.data_path[1], str(self.DF[index, 0]))
            elif index<self.len[2]:
                imgName = os.path.join(self.data_path[2], str(self.DF[index, 0]))
            else: 
                imgName = self.DF[index, 0]
            imgName = imgName.replace('\\', '/')

            Img = Image.open(imgName)

            if self.transform is not None:
                Img = self.transform(Img)
            im = np.array(Img)
            self.sum1 = self.sum1 + im.sum(axis=(0,1))
            self.shape = self.shape + im.shape[0]*im.shape[1]
        self.mean = self.sum1 / self.shape
    def compute_std(self):
        for index in range(len(self.DF)):
            if index<self.len[0]:
                imgName = os.path.join(self.data_path[0], str(self.DF[index, 0]))
            elif index<self.len[1]:
                imgName = os.path.join(self.data_path[1], str(self.DF[index, 0]))
            elif index<self.len[2]:
                imgName = os.path.join(self.data_path[2], str(self.DF[index, 0]))
            else: 
                imgName = self.DF[index, 0]
            imgName = imgName.replace('\\', '/')

            Img = Image.open(imgName)

            if self.transform is not None:
                Img = self.transform(Img)
            im = np.array(Img)
            self.sum2 = self.sum2 + np.sum(np.square(im-self.mean[np.newaxis, np.newaxis, :]), axis=(0,1))
        self.std = np.sqrt(self.sum2 / self.shape)
    def mean(self):
        return self.mean
    def std(self):
        return self.std


    def __len__(self):
        return len(self.DF)



if __name__ == "__main__":
    import sys
    CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
    config_path = CURRENT_DIR.rsplit('/', 3)[0]  # 上三级目录
    from utils import MinCrop
    import pandas as pd
    data_root = 'data/DDR/DR_grading/'
    root = 'data/'
    """Basic Setting"""
    data_path_train = data_root + 'train/'
    data_path_val = data_root + 'valid/'
    data_path_test = data_root + 'test/'
    Name_train = root + 'data_hu/data/DDR/01_0_1234train.txt'
    Name_val = root + 'data_hu/data/DDR/01_0_1234valid.txt'
    Name_test = root + 'data_hu/data/DDR/01_0_1234test.txt'
    DF0= pd.read_csv(Name_train, sep=' ') #6259
    DF1= pd.read_csv(Name_val, sep=' ') #2502
    DF2= pd.read_csv(Name_test, sep=' ') #3758
    DF = pd.read_csv("data/MyDRdataset/part_of_qilu_path_label.csv", encoding='UTF')
    DF_train_ddr_qilu = np.append(np.append(np.append(DF0.values, DF1.values, axis=0), DF2.values, axis=0), DF.values, axis=0)
    transform_train = transforms.Compose([
        transforms.Resize(800),
        MinCrop()])
    # dataset_train_ddr_qilu= dataset_train_DDR_QILU([data_path_train, data_path_val, data_path_test],[len(DF0),len(DF0)+len(DF1),len(DF0)+len(DF1)+len(DF2)], DF_train_ddr_qilu, transform = transform_train)
    # train_loader_ddr_qilu = DataLoader(dataset_train_ddr_qilu, 1, num_workers=8,  drop_last=True, shuffle=True)
    # for batch_idx, (inputs, labels, img_paths) in enumerate(train_loader_ddr_qilu):
    data = dataset_mean_std_DDR_QILU([data_path_train, data_path_val, data_path_test],[len(DF0),len(DF0)+len(DF1),len(DF0)+len(DF1)+len(DF2)], DF_train_ddr_qilu, transform = transform_train)
    data.compute_mean()
    data.compute_std()
    print('mena=',data.mean)
    print('std=', data.std)
    print('sum1=', data.sum1)
    print('sum2=', data.sum2)
    print('shape=', data.shape)
