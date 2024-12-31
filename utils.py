import os
import logging
from cv2 import accumulate
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix
import sys
import torchvision
import random
from PIL import Image
from sklearn.cluster import AgglomerativeClustering,KMeans
from sklearn.metrics import silhouette_score

from model import *
from datasets import CIFAR10_truncated, CIFAR100_truncated, MNIST_truncated
from collections import defaultdict


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def compute_max_update_norm(cluster):
    return np.max([torch.norm(flatten(client.state_dict())).item() for client in cluster])

def compute_mean_update_norm(cluster):
    return torch.norm(torch.mean(torch.stack([flatten(client.state_dict()) for client in cluster]), 
                                dim=0)).item()

def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])

def cluster_clients(n_clusters, S):
    # clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete", n_clusters=None, distance_threshold = 0.01).fit(-S)
    my_kmeans = KMeans(n_clusters)
    my_kmeans.fit(-S)
    # Get cluster labels
    cluster_labels = my_kmeans.fit_predict(-S)
    # Initialize a dictionary to store indices of data points in each cluster
    cluster_indices = defaultdict(list)

    # Iterate through data points and their corresponding cluster labels
    for i, label in enumerate(cluster_labels):
        cluster_indices[label].append(i)

    # Convert the defaultdict to a regular dictionary
    cluster_indices = dict(cluster_indices)
    
    # c1 = np.argwhere(clustering.labels_ == 0).flatten() 
    # c2 = np.argwhere(clustering.labels_ == 1).flatten() 
    return cluster_indices

# def find_optimal_clusters(S, max_clusters):
#     silhouette_scores = []
#     # 标记每个客户端所属的簇，长度与客户端数量一致，每个元素表示对应数据点所属的簇的标签
#     cluster_labels = []
    
#     # 从2到max_clusters尝试不同数量的簇
#     for n_clusters in range(2, max_clusters + 1):
#         clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="complete").fit(-S)
#         labels = clustering.labels_
#         cluster_labels.append(labels)
        
#         # 计算 Silhouette Score 来评估簇的质量
#         silhouette_avg = silhouette_score(S, labels)
#         silhouette_scores.append(silhouette_avg)
    
#     # 找到具有最高 Silhouette Score 的簇数
#     optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # 加2是因为从2开始尝试的
    
#     return cluster_labels[optimal_n_clusters - 2], optimal_n_clusters

# def cluster_clients(S):
#     cluster_labels, n_clusters = find_optimal_clusters(S, max_clusters=10)  # 设置最大尝试的簇数
    
#     # 初始化一个字典，用于存储每个簇的数据点索引
#     clusters = {}
#     for i in range(n_clusters):
#         clusters[i] = []

#     # 将数据点按簇标签分组
#     for i, label in enumerate(cluster_labels):
#         clusters[label].append(i)

#     return clusters

def get_n_params(model):
    """return the number of parameters in the model"""

    n_params = sum([np.prod(tensor.size()) for tensor in list(model.parameters())])
    return n_params

def get_std(model_A, model_B, noise):
    """get the standard deviation at iteration 2 with the proposed heuristic"""

    list_tens_A = [tens_param.detach() for tens_param in list(model_A.parameters())]
    list_tens_B = [tens_param.detach() for tens_param in list(model_B.parameters())]

    if noise == "plain":
        return [0, 0]

    if noise == "disguised":

        sum_abs_diff = 0

        for tens_A, tens_B in zip(list_tens_A, list_tens_B):
            sum_abs_diff += torch.sum(torch.abs(tens_A - tens_B))

        std = sum_abs_diff / get_n_params(model_A)
        return [0, std]

def linear_noising(
    model,
    list_std: list,
    list_power: list,
    iteration: int,
    noise_type: str,
    std_multiplicator,
):
    """Return the noised model of the free-rider"""

    if noise_type == "disguised":
        for idx, layer_tensor in enumerate(model.parameters()):

            mean_0 = torch.zeros(layer_tensor.size())
            std_tensor = torch.zeros(
                layer_tensor.size()
            ) + std_multiplicator * list_std[1] * iteration ** (-list_power[1])
            noise_additive = torch.normal(mean=mean_0, std=std_tensor)

            layer_tensor.data += noise_additive

    return model

def compute_pairwise_similarities(clients):
    sources = [client.state_dict() for client in clients]
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i,j] = torch.sum(s1*s2)/(torch.norm(s1)*torch.norm(s2)+1e-12)

    return angles.numpy()

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)

def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_mnist_data(datadir):
    # 对MNIST数据进行预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 使用MNIST_truncated类来加载数据集
    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    # 获取数据和标签
    # X_train, y_train = np.array([mnist_train_ds[i][0] for i in range(len(mnist_train_ds))]), np.array([mnist_train_ds[i][1] for i in range(len(mnist_train_ds))])
    # X_test, y_test = np.array([mnist_test_ds[i][0] for i in range(len(mnist_test_ds))]), np.array([mnist_test_ds[i][1] for i in range(len(mnist_test_ds))])

    # 预分配空间
    X_train = np.empty((len(mnist_train_ds), 1, 28, 28))
    y_train = np.empty(len(mnist_train_ds))
    X_test = np.empty((len(mnist_test_ds), 1, 28, 28))
    y_test = np.empty(len(mnist_test_ds))
    
    for i in range(len(mnist_train_ds)):
        X_train[i], y_train[i] = mnist_train_ds[i]
    
    for i in range(len(mnist_test_ds)):
        X_test[i], y_test[i] = mnist_test_ds[i]

    return (X_train, y_train, X_test, y_test)

def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts_dict = {}
    net_cls_counts_npy = np.array([])
    # 数据集中的总类别数，+1确保索引从0开始
    num_classes = int(y_train.max()) + 1

    for net_i, dataidx in net_dataidx_map.items():
        
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        # edit
        unq = unq.astype(int)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts_dict[net_i] = tmp
        tmp_npy = np.zeros(num_classes)
        for i in range(len(unq)):
            tmp_npy[unq[i]] = unq_cnt[i]
        net_cls_counts_npy = np.concatenate(
                        (net_cls_counts_npy, tmp_npy), axis=0)
    # 包含每个客户端/子网络中各个类别的样本数量
    net_cls_counts_npy = np.reshape(net_cls_counts_npy, (-1,num_classes))
    data_list=[]
    for net_id, data in net_cls_counts_dict.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    print(net_cls_counts_npy.astype(int))
    return net_cls_counts_npy


def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4, n_niid_parties=5, global_imbalanced=False):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    
    # y_train中元素个数,即训练集大小
    n_train = y_train.shape[0]

    if partition == "noniid-labeldir" or partition == "noniid": # corresponds to NIID-1 in our paper 
        min_size = 0
        min_require_size = 10
        # 分类目标数
        K = len(np.unique(y_train))
        # N为训练集大小
        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            # e.g. n_parties=10, idx_batch = [[], [], [], [], [], [], [], [], [], []]
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                # 标签为k的所有数据的索引
                idx_k = np.where(y_train == k)[0]
                # 打乱
                np.random.shuffle(idx_k)

                if global_imbalanced:   # global dataset is imbalanced, following an exponential decay
                    ratio = 10
                    num_k = int(n_train/n_parties * (ratio ** (-k/(K-1))))
                    idx_k = idx_k[:num_k]
                    print("(for global imbalanced) k: ", k, " num: ", len(idx_k))

                # 
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))  
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # np.split(idx_k, proportions)将idx_k数组按proportions切分
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            # net_dataidx_map是一个字典，字典的键表示不同客户端的标识，值对应客户端的训练数据索引列表
            net_dataidx_map[j] = idx_batch[j]
            print(len(net_dataidx_map[j]))
        class_dis = np.zeros((n_parties, K))

        for j in range(n_parties):
            for m in range(K):
                class_dis[j,m] = int((np.array(y_train[idx_batch[j]])==m).sum())
        print(class_dis.astype(int))

    elif partition == 'noniid-4' and dataset == 'cifar10': # corresponds to NIID-2 in our paper  
        labels = y_train
        num_non_iid_client = n_niid_parties                              # has 2 classes, should satisfy num_non_iid_client%5=0
        num_iid_client =  n_parties-num_non_iid_client      # has 10 classes
        num_classes = int(labels.max()+1)
        num_sample_per_client = n_train//n_parties
        num_sample_per_class = n_train//num_classes
        num_per_shard = int(n_train/num_classes/(num_non_iid_client+num_iid_client))     # num_non_iid_client+num_iid_client: non_iid_client has 2 classes，while iid_client has 10 classes, so non_iid_client has 5 times data per class

        net_dataidx_map = {i: np.array([]).astype(int) for i in range(n_parties)}
        idxs = np.arange(n_train).astype(int)

        # sort labels
        idxs_labels = np.vstack((idxs, labels)).astype(int)     # each class has 5000 samples
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]                    # index follows the sequnce of label

        # partition of non-iid clients
        for i in range(num_non_iid_client):
            net_dataidx_map[i] = np.concatenate(
                    (net_dataidx_map[i], idxs[((2*i)%10)*num_sample_per_class+num_per_shard*(i//5)*5:((2*i)%10)*num_sample_per_class+num_per_shard*(i//5+1)*5]), axis=0)
            print(((2*i)%10)*num_sample_per_class+num_per_shard*(i//5)*5,((2*i)%10)*num_sample_per_class+num_per_shard*(i//5+1)*5)
            net_dataidx_map[i] = np.concatenate(
                    (net_dataidx_map[i], idxs[((2*i+1)%10)*num_sample_per_class+num_per_shard*(i//5)*5:((2*i+1)%10)*num_sample_per_class+num_per_shard*(i//5+1)*5]), axis=0)
            print(((2*i+1)%10)*num_sample_per_class+num_per_shard*(i//5)*5,((2*i+1)%10)*num_sample_per_class+num_per_shard*(i//5+1)*5)
            np.random.shuffle(net_dataidx_map[i])
            net_dataidx_map[i] = list(net_dataidx_map[i])
            
        # partition of iid clients
        for i in range(num_non_iid_client,n_parties):
            for j in range(num_classes):
                net_dataidx_map[i] = np.concatenate(
                        (net_dataidx_map[i], idxs[j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client): \
                                                    j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client+1)]), axis=0)
                print(j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client),j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client+1))
            np.random.shuffle(net_dataidx_map[i])
            net_dataidx_map[i] = list(net_dataidx_map[i])
    # 统计各个客户端或子网络数据集中各个类别的样本数量，即各个客户端的数据分布
    # traindata_cls_counts[j,m]表示第j个客户端或子网络中属于类别m的样本数量
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)

    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def get_trainable_parameters(net, device='cpu'):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    N = 0
    for params in paramlist:
        N += params.numel()
    X = torch.empty(N, dtype=torch.float64, device=device)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset + numel].copy_(params.data.view_as(X[offset:offset + numel].data))
        offset += numel
    return X

def put_trainable_parameters(net, X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset + numel].data.view_as(params.data))
        offset += numel

def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False, dataset=None, get_testacc_class=False):
    # model = model.to(device)

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    K = 10
    correct_class = np.zeros(K)  # to calculate test accuracy on each class
    total_class = np.zeros(K) 

    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()

    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(loader):
                    if device != 'cpu':
                        x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                    _, _, out = model(x)

                    if len(target)==1:
                        out= out.unsqueeze(0)
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                    if device == "cpu":
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                        true_labels_list = np.append(true_labels_list, target.data.numpy())
                    else:
                        pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                        true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                if device != 'cpu':
                    # x, target = x.to(device), target.to(dtype=torch.int64).to(device)
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                _,_,out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                # correct classification on each class
                if get_testacc_class:
                    for i in range(K):
                        correct_class[i] += ((pred_label == i) & (pred_label == target.data)).sum().item()
                        total_class[i] += (target.data == i).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        if get_testacc_class:
            return correct / float(total), conf_matrix, avg_loss, correct_class / total_class
        else:
            return correct / float(total), conf_matrix, avg_loss
    
    return correct / float(total), avg_loss


def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir + "trained_local_model" + str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return

def load_model(model, model_index, device="cpu"):
    with open("trained_local_model" + str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    if device == "cpu":
        model.to(device)
    else:
        model.cuda()
    return model


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, drop_last=True, noise_level=0):
    if dataset in('cifar10', 'cifar100', 'mnist'):
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated
            
            # 对训练集进行transformation
            transform_train=transforms.Compose([
                transforms.ToPILImage(),
                # 随机裁剪
                transforms.RandomCrop(32, padding=4),
                # 随机水平翻转
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # 数据标准化
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            
            # 对测试集进行transformation
            transform_test=transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated

            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            
            transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])            
        
        elif dataset == 'mnist':
            dl_obj = MNIST_truncated
            
            transform_train = transforms.Compose([
                transforms.RandomRotation(5, fill=(0,)),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])
        
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])

        # 训练集
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        # 测试集
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
        # 从训练集中批量加载数据
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=drop_last, shuffle=True)
        # 从测试集中批量加载数据
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)
    
    return train_dl, test_dl, train_ds, test_ds
