import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, EMNIST

import os
import os.path
import logging
import sys
import torchvision.transforms as transforms
import random
import torch

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        
        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)
        
        data = mnist_dataobj.data.numpy()  # Convert torch.Tensor to numpy array
        target = np.array(mnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)  # Convert numpy array to PIL Image
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)



class CIFAR100_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # root-数据集文件的存储路径 train-是否为训练集true训练集false测试集 
        # transform-加载数据前的格式转换 target_transform download-是否下载
        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs      # 指定要加载的数据集的子集索引，若为None则加载整个数据集
        self.train = train      # 一个布尔值，指示是否加载训练集。如果为 True，则加载训练集，否则加载测试集
        self.transform = transform      # 一个可选的数据变换函数，用于对图像数据进行预处理
        self.target_transform = target_transform      # 一个可选的目标变换函数，用于对目标标签进行预处理
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # 创建cifar10数据集对象cifar_dataobj
        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
        # 根据torchvision的版本选择数据和标签的加载方式
        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else: 
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        # 若指定了dataidxs则根据其截取数据集的子集
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    # 将数据集中指定index的图像通道值设置为0，通常用于数据预处理或数据修改
    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    # 根据index获取数据集中的样本，返回元组为(img, target)，包含图像和标签
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # 如果指定了transform则会对图像进行预处理
        if self.transform is not None:
            img = self.transform(img)

        # 如果指定了target_transform则对标签进行预处理
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
    
    # 返回数据集中样本数量
    def __len__(self):
        return len(self.data)
    

    
