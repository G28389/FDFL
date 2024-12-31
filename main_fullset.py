"""Our codes are adapted from codes in Model-Contrastive Federated Learning"""

from importlib_metadata import distribution
import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
import sys
import itertools
from scipy import spatial
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *

from distribution_aware.utils import get_distribution_difference
from sklearn.metrics import precision_score, recall_score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple-cnn-mnist', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    ################Original code###################
    # parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    # parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    ###################Ends Here#############################
    parser.add_argument('--alg', type=str, default='fair',
                        help='communication strategy: FDFL/fair')
    parser.add_argument('--comm_round', type=int, default=3, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./dataset/mnist", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:1', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model',type=int,default=0)
    parser.add_argument('--use_project_head', type=int, default=0)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    # Experiments on FedDisco
    parser.add_argument('--n_niid_parties', type=int, default=5, help='number of niid workers')         # for non-iid-4
    parser.add_argument('--distribution_aware', type=str, default='division', help='Types of distribution aware e.g. division')
    # parser.add_argument('--distribution_aware', type=str, default='not', help='Types of distribution aware e.g. division')
    parser.add_argument('--measure_difference', type=str, default='l2', help='How to measure difference. e.g. only_iid, cosine')
    # parser.add_argument('--difference_operation', type=str, default='square', help='Conduct operation on difference. e.g. square or cube')
    parser.add_argument('--disco_a', type=float, default=0.5, help='Under sub mode, n_k-disco_a*d_k+disco_b')
    parser.add_argument('--disco_b', type=float, default=0.1)
    parser.add_argument('--disco_partition', type=int, default=0, help='Whether to apply model partition, 0 means not. 1 means apply')
    parser.add_argument('--global_imbalanced', action='store_true') # imbalanced global dataset
    # My experiment
    parser.add_argument('--n_freerider', type=int, default=1, help='the number of free-riders')
    parser.add_argument('--n_normal_client', type=int, default=29, help='number of normal clients')
    parser.add_argument('--coef', type=int, default=1, help='multiplicative for the heuristic std')
    parser.add_argument('--power', type=float, default=1, help='Gamma for the noise')
    parser.add_argument('--n_clusters', type=int, default=4, help='the number of cluster')
    parser.add_argument('--budget', type=int, default=80, help='the budget of server')
    parser.add_argument('--im', type=str, default='mg', help='reward allocation rule:quality_aware/knapsack/price_first')
    parser.add_argument('--free_rider_detection', action='store_true', help='Whether to detect and delete free-rider client before aggregation')
    # MG
    parser.add_argument('--beta_mg', type=float, default=1, help='relative weight')
    args = parser.parse_args()
    return args

def compute_freerider_loss(model, dataloader, device="cpu", dataset=None):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()

    loss_collector = []

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            _,_,out = model(x)
            loss = criterion(out, target)
            loss_collector.append(loss.item())

        avg_loss = sum(loss_collector) / len(loss_collector)
    
    if was_training:
        model.train()    

    return avg_loss

def find_smallest_k(sorted_rank, q, B):
    
    for k_candidate in range(1, len(sorted_rank)):
        sum_value = 0
        k = 0
        for idx, value in sorted_rank.items():
            if k >= k_candidate:
                break
            if value != 0:
                tmp1 = q[idx] / q[k_candidate-1]
                sum_value += tmp1
            if sum_value > B:
                if k == k_candidate:
                    return k_candidate
                else:
                    break
            k += 1
    return k_candidate        
        
def find_smallest_w(sorted_rank, q, B):
    w = 0
    total_sum = 0
    for idx, value in sorted_rank.items():
        total_sum += q[idx]
        total_bid = total_sum / value
        if total_bid > B:
            return w #返回的是满足条件的客户端数量
        w += 1
    return w

def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    # 根据数据集确定类别数
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 26
    elif args.dataset == 'emnist':
        n_classes = 47
    
    for net_i in range(n_parties):
        net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs, args.dataset)
        if device == 'cpu':
            net.to(device)
        else:
            net = net.cuda()
        nets[net_i] = net

    # 模型每一层权重的shape
    model_meta_data = []
    # 模型每层的名称或表示
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type

# 每个客户端进行本地训练
def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    # 选择优化器，设置学习率和权重衰减
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    # 选择损失函数
    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0   
    for epoch in range(epochs):     
        epoch_loss_collector = []    
        for batch_idx, (x, target) in enumerate(train_dataloader):  
            x, target = x.cuda(), target.cuda()
            # 清零优化器的梯度缓冲区，准备计算新的梯度
            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()
            # 使用神经网络net对输入数据x进行前向转播，获取网络输出out
            _,_,out = net(x)
            # 使用前面定义的损失函数计算输出与目标标签之间的损失
            loss = criterion(out, target)           
            # 计算损失函数相对于神经网络权重的梯度
            loss.backward()
            # 使用计算得到的梯度更新神经网络的权重
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')

    logger.info(' ** Training complete **')
    return test_acc, epoch_loss

def compute_model_accuracy(net, test_dataloader, device="cpu"):
    net.to(device)
    net.eval()  # 设置模型为评估模式，关闭Dropout等

    correct = 0
    total = 0
    with torch.no_grad():  # 关闭梯度计算，节省计算资源
        for batch_idx, (x, target) in enumerate(test_dataloader):
            x, target = x.to(device), target.to(device)
            target = target.long()
            _, _, out = net(x)  # 假设net的输出方式与上述代码相同
            _, predicted = torch.max(out.data, 1)  # 获取预测类别
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    logger.info('>> Model accuracy: %f' % accuracy)
    net.to('cpu')

    return accuracy

def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, datasize = None, global_model = None, prev_model_pool = None, server_c = None, clients_c = None, round=None, device="cpu", current_round=0):
    avg_acc = 0.0
    acc_list = []
    loss_list = []
    # clients_params = []

    if global_model:
        global_model.cuda()
    if server_c:
        server_c.cuda()
        server_c_collector = list(server_c.cuda().parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)
    # 训练每个客户端的模型
    if args.alg == 'FDFL':
        for net_id, net in nets.items():
            if net_id < args.n_normal_client:
            # net_dataidx_map是一个字典，字典的键表示不同客户端的标识，值对应客户端的训练数据索引列表
                dataidxs = net_dataidx_map[net_id]
                train_dl_local=train_dl[net_id]   
                n_epoch = args.epochs
                testacc, loss = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                            device=device)
                print("Training network %s, n_training: %d, final test acc %f." % (str(net_id), len(dataidxs), testacc))
                acc_list.append(testacc)
                loss_list.append(loss)
            # else:
            #     testacc = compute_model_accuracy(net, test_dl, device=device)
            #     print("Training network %s, n_training: %d, final test acc %f." % (str(net_id), datasize[net_id], testacc))
            #     acc_list.append(testacc)

    if args.alg == 'fair':
        for net_id, net in nets.items():
            if net_id < args.n_normal_client:
            # net_dataidx_map是一个字典，字典的键表示不同客户端的标识，值对应客户端的训练数据索引列表
                dataidxs = net_dataidx_map[net_id]
                train_dl_local=train_dl[net_id]   
                n_epoch = args.epochs
                testacc, loss = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                            device=device)
                print("Training network %s, n_training: %d, final test acc %f." % (str(net_id), len(dataidxs), testacc))
                acc_list.append(testacc)
                loss_list.append(loss)
    if args.alg == 'MG':
        for net_id, net in nets.items():
            if net_id < args.n_normal_client:
            # net_dataidx_map是一个字典，字典的键表示不同客户端的标识，值对应客户端的训练数据索引列表
                dataidxs = net_dataidx_map[net_id]
                train_dl_local=train_dl[net_id]   
                n_epoch = args.epochs
                testacc, loss = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                            device=device)
                print("Training network %s, n_training: %d, final test acc %f." % (str(net_id), len(dataidxs), testacc))
                acc_list.append(testacc)
                loss_list.append(loss)
     
    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    # 每个客户端模型的测试精度以及训练损失
    return acc_list, loss_list

def update_utility(utility, decisions, attendance, n_clients):
    utility = utility - decisions * attendance / n_clients
    return utility

if __name__ == '__main__':

    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    print(now_time)
    vis_dir = os.path.join('visualize/exp_vis', now_time)
    os.mkdir(vis_dir)
    dataset_logdir = os.path.join(args.logdir, args.dataset)
    mkdirs(dataset_logdir)

    writer = SummaryWriter(os.path.join(args.logdir, 'board'))

    std_original = 10 ** -3
    
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % (now_time)
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(dataset_logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (now_time)
    log_path = args.log_file_name + '.log'

    logging.basicConfig(
        filename=os.path.join(dataset_logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    # data partition
    logger.info("Partitioning data")
    # 为normal client分配数据
    # net_dataidx_map是一个字典，字典的键表示不同客户端的标识，值对应客户端的训练数据索引列表
    # traindata_cls_counts 统计各个客户端或子网络数据集中各个类别的样本数量，即各个客户端的数据分布
    # traindata_cls_counts[j,m]表示第j个客户端或子网络中属于类别m的样本数量
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, dataset_logdir, args.partition, args.n_normal_client, beta=args.beta, n_niid_parties=args.n_niid_parties, global_imbalanced=args.global_imbalanced)
    
    # 正常客户端集合
    normal_clients_list = [i for i in range(args.n_normal_client)]
    # 求所有客户端集合
    if args.n_freerider != 0:
        # 搭便车客户端集合
        freerider_list = [i for i in range(args.n_freerider)]
        party_list = list(range(len(normal_clients_list)+len(freerider_list)))   
    else:
        party_list = normal_clients_list 
    # 求每一轮参与FL的客户端列表 party_list_rounds
    party_list_rounds = []
    for i in range(args.comm_round):
        party_list_rounds.append(party_list)

    # 计算训练集包含的分类类别数目
    n_classes = len(np.unique(y_train))

    # get testing dataloader
    train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               32)

    print("len train_dl_global:", len(train_ds_global))
    train_dl=None
    data_size = len(test_ds_global)
    
    # net initialization for clients
    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, len(party_list), args, device='cpu')
    # 初始化全局模型
    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
    global_model = global_models[0]
    # 初始全局模型的损失
    # _, init_global_loss = compute_accuracy(global_model, test_dl, device=device, dataset = args.dataset)
    _, init_global_loss = compute_accuracy(global_model, test_dl, device='cpu', dataset = args.dataset)

    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    print(device)
    
    # get training dataloader for normal clients
    train_local_dls=[]  
    for net_id, net in nets.items():
        if net_id < args.n_normal_client:
            dataidxs = net_dataidx_map[net_id]
            train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
            # 训练数据
            train_local_dls.append(train_dl_local)

    best_test_acc=0
    record_test_acc_list = []
    record_winner = []
    record_recall = []
    record_precision = []
    acc_dir = os.path.join(dataset_logdir, 'acc_list')
    winner_dir = os.path.join(dataset_logdir, 'winner_list')
    precision_dir = os.path.join(dataset_logdir, 'precision_list')
    recall_dir = os.path.join(dataset_logdir, 'recall_list')
    if not os.path.exists(acc_dir):
        os.mkdir(acc_dir)
    if not os.path.exists(winner_dir):
        os.mkdir(winner_dir)
    if not os.path.exists(precision_dir):
        os.mkdir(precision_dir)
    if not os.path.exists(recall_dir):
        os.mkdir(recall_dir)

    acc_path = os.path.join(dataset_logdir, f'acc_list/{now_time}.npy')
    winner_path = os.path.join(dataset_logdir, f'winner_list/{now_time}.npy')
    precision_path = os.path.join(dataset_logdir, f'precision_list/{now_time}.npy')
    recall_path = os.path.join(dataset_logdir, f'recall_list/{now_time}.npy')

    if args.alg == 'fair':

        for round in range(n_comm_rounds):
            # 获取当前全局模型的loss
            if round == 0:
                global_loss = init_global_loss
            else:
                global_loss = avg_loss

            logger.info("in comm round:" + str(round))
            print("In communication round:" + str(round))
            party_list_this_round = copy.deepcopy(party_list_rounds[round])
            print(f'Clients this round : {party_list_this_round}')

            if args.n_freerider != 0:
                # Free-riders' local data distribution
                freerider_total_data = []
                for j in range(args.n_freerider):
                    freerider_cls_count = [random.randint(450, 520) for _ in range(n_classes)]  
                    freerider_total_data.append(sum(freerider_cls_count))
                    traindata_cls_counts = np.vstack([traindata_cls_counts, freerider_cls_count])

            datasize = {}
            for client in range(len(party_list_this_round)):
                datasize[client] = sum(traindata_cls_counts[client])

            # Global model Initialization
            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)
            
            # Free-riders’ local update
            if args.n_freerider != 0:
                noise_type = 'disguised'
                list_std = [0, std_original]
                list_power = [0, args.power]
                if round != 0:
                    list_std = get_std(global_model, m_previous, noise_type)
                for net_id, net in nets.items():
                    if net_id in range(args.n_normal_client, len(party_list_this_round)):
                        nets[net_id] = linear_noising(
                            net,
                            list_std,
                            list_power,
                            max(i, 1),
                            noise_type,
                            args.coef,
                        )
            # Free-riders' local data size
            freerider_total_data = {}
            for client in range(args.n_normal_client, len(party_list_this_round)):
                freerider_total_data[client] = random.randint(2500, 3000)
            # Normal client local update
            test_acc_list, loss_list = local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, device=device)
            # Free-rider's loss
            for net_id, net in nets.items():
                if net_id in range(args.n_normal_client, len(party_list_this_round)):
                    loss = compute_freerider_loss(net, test_dl)
                    loss_list.append(loss)
            # Learning quality quantification
            # 数据质量
            data_quality = {}
            # 学习质量
            learning_quality = {}
            for client in party_list_this_round:
                data_quality[client] = global_loss - loss_list[client]
                # if client not in client_learning_quality:
                    # client_learning_quality[client] = []
                if client in range(args.n_normal_client):
                    # learning_quality[client] = len(net_dataidx_map[client]) * data_quality[client]
                    learning_quality[client] = datasize[client] * data_quality[client]
                else:
                    learning_quality[client] = datasize[client] * data_quality[client]
            # total_quality = sum([learning_quality[r] for r in range(len(party_list_this_round))])
            # learning_quality = [learning_quality[r] / total_quality for r in range(len(party_list_this_round))]

            # Quality-aware incentive mechanism
            # 各个客户端的报价
            client_bid = {}
            exp = {}
            client_exp = {}
            server_budget = args.budget
            honest_client = []
            rank = {}
            for client in party_list_this_round:
                exp[client] = random.randint(1, 3)
                if client >= args.n_normal_client:
                        client_bid[client] = 1
                else:
                    client_bid[client] = random.randint(1, 3)  
                rank[client] = learning_quality[client] / client_bid[client]
            client_exp = {key:client_bid.get(key,0) + exp.get(key,0) for key in set(client_bid) | set(exp)}
            sorted_rank = dict(sorted(rank.items(), key=lambda item: item[1], reverse = True))
            new_sorted = {k: v for k, v in sorted_rank.items() if v >= 0}
            k_value = find_smallest_k(new_sorted, learning_quality, server_budget)
            
            tmp = 0
            reward = {}
            for client in new_sorted:
                tmp += 1
                if tmp == k_value:
                    break
                reward[client] = client_bid[k_value - 1] * rank[client] / rank[k_value - 1]


            count = 0
            for key in new_sorted:
                count += 1
                if count == k_value:
                    break
                else:
                    if round >= 3:
                        if (reward[key]) >= client_exp[key]:
                            honest_client.append(key)
                    else:
                        honest_client.append(key)
            winner_sum = len(honest_client)
            print(f'winner clients this round : {winner_sum}')

            # Model aggregation
            total_D = 0
            for client in honest_client:
                total_D += learning_quality[client]
            num_a = 0
            for net_id in honest_client:
                # 获得客户端模型参数
                net_para = nets[net_id].state_dict() 
                weight_factor = learning_quality[net_id] / total_D
                if num_a == 0:
                # 若为第一个客户端，则直接按权重存储
                    for key in net_para:
                        # global_w[key] = net_para[key] * learning_quality[net_id] / total_D
                        global_w[key] = net_para[key] * weight_factor
                    num_a = 1
                else:
                # 其他客户端则乘以权重后添加到全局模型
                    for key in net_para:
                        global_w[key] = global_w[key] + net_para[key] * weight_factor    
                        
            global_model.load_state_dict(global_w)

            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            test_acc, conf_matrix, avg_loss = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device, dataset = args.dataset)
            record_test_acc_list.append(test_acc)
            record_winner.append(winner_sum)

            writer.add_scalar('testing accuracy',
                            test_acc,
                            round)
            if(best_test_acc<test_acc):
                best_test_acc=test_acc
                logger.info('New Best best test acc:%f'% test_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
            print('>> Global Model Test accuracy: %f, Best: %f' % (test_acc, best_test_acc))
           
            mkdirs(args.modeldir+'fedavg/')
            global_model.to('cpu')

            # My edit
            m_previous = copy.deepcopy(global_model)

            torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')
        np.savetxt(acc_path, np.array(record_test_acc_list))
        np.savetxt(winner_path, np.array(record_winner))

    if args.alg == 'FDFL':
        for round in range(n_comm_rounds):

            logger.info("in comm round:" + str(round))
            print("In communication round:" + str(round))
            party_list_this_round = copy.deepcopy(party_list_rounds[round])
            if args.n_freerider != 0:
                print(f'Clients this round : {party_list_this_round}')

                # Free-riders' local data distribution
                freerider_total_data = []
                for j in range(args.n_freerider):
                    freerider_cls_count = [random.randint(450, 520) for _ in range(n_classes)]  
                    freerider_total_data.append(sum(freerider_cls_count))
                    traindata_cls_counts = np.vstack([traindata_cls_counts, freerider_cls_count])

            datasize = {}
            for client in range(len(party_list_this_round)):
                datasize[client] = sum(traindata_cls_counts[client])
        
            # Model Initialization
            # state_dict()返回一个包含模型权重的字典，仅包含模型的参数值而不包含模型的结构或其他信息
            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            # Free-riders’ local update
            if args.n_freerider != 0:
                noise_type = 'disguised'
                list_std = [0, std_original]
                list_power = [0, args.power]
                if round != 0:
                    list_std = get_std(global_model, m_previous, noise_type)
                for net_id, net in nets.items():
                    if net_id in range(args.n_normal_client, len(party_list_this_round)):
                        nets[net_id] = linear_noising(
                            net,
                            list_std,
                            list_power,
                            max(i, 1),
                            noise_type,
                            args.coef,
                        )
            
            # Normal client's local update 同时会更新本地权重
            # test_acc_list 每个本地客户端模型的测试精度
            test_acc_list, _ = local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, device=device, datasize = datasize)

            # free-riders detection
            qualified_client_list = []
            if args.free_rider_detection:
                # Step 1: clustering based on the weight of each client
                EPS_1 = 0.4
                EPS_2 = 1.6
                # 初始客户端群组
                cluster_indices = np.arange(len(party_list_this_round)).astype("int")
                similarities = compute_pairwise_similarities(nets_this_round.values())

                # 更新后的客户端群组
                # cluster_indices_new = []
                # for idc in cluster_indices:
                max_norm = compute_max_update_norm([nets[i] for i in cluster_indices])
                mean_norm = compute_mean_update_norm([nets[i] for i in cluster_indices])

                clusters = cluster_clients(args.n_clusters, similarities) 
            
                # Step 2: calculate the similarity of each clients' data distribution
                # 首先计算每个簇中各个客户端之间数据分布的余弦相似度
                cos_sim = {}
                num_correct = 0
                total_freeriders_index = []
                qualified_client_list = []
                y_true = [0 for i in range(len(party_list_this_round))]
                y_pred = [0 for i in range(len(party_list_this_round))]
                for client in party_list_this_round:
                    if client >= args.n_normal_client:
                        y_true[client] = 1
                for cluster_name, cluster_clientset in clusters.items():
                    data_array = {}
                    for client in cluster_clientset:
                        # data_array是该簇内所有客户端的数据分布
                        data_array[client] = traindata_cls_counts[client]
                    # 转换为NumPy数组
                    # data_array = np.array(data_array)
                    pairwise_sim = {}
                    # flag记录每个客户端和其他客户端差异超出阈值的个数，flag[i]=a表示客户端i和其他a个客户端相似性差异过大
                    flag = {}
                    for i in cluster_clientset:
                        flag[i] = 0
                        for j in cluster_clientset:
                            if i != j:
                                # 计算两两客户端之间的余弦相似度
                                pairwise_sim[(i,j)] = 1 - spatial.distance.cosine(data_array[i], data_array[j])
                                # 搭便车检测阈值
                                if pairwise_sim[(i,j)] < 0.08:
                                    flag[i] = flag[i] + 1
                    cos_sim[cluster_name] = pairwise_sim
                    # 检测簇内freerider
                    for client, value in flag.items():
                        # 客户端k为恶意
                        if value == len(flag) - 1:
                            # 获得新的客户端集合
                            total_freeriders_index.append(client)
                            # 更新客户端数据分布集合
                            if client in range(args.n_normal_client, len(party_list_this_round)):
                                num_correct = num_correct + 1
                        # 生成合格客户端集合
                        else:
                            # qualified_client_list不是顺序存放的
                            qualified_client_list.append(client)
                precision = precision_score(y_true, y_pred, average='binary')
                recall = recall_score(y_true, y_pred, average='binary')
                record_precision.append(precision)
                record_recall.append(recall)
                print(f'Freerider detection results : {total_freeriders_index}')
                print(f'Precision results : {record_precision}')
                print(f'Recall results : {record_recall}')
            else:
                qualified_client_list = party_list_this_round

            # calculate the degree of class-imbalance for qualified client
            # traindata_cls_counts[j,m]表示第j个客户端或子网络中属于类别m的样本数量
            traindata_array = []
            for client in party_list_this_round:
                traindata_array.append(traindata_cls_counts[client])
            traindata_array = np.array(traindata_array)
            distribution_difference = get_distribution_difference(traindata_array, participation_clients=qualified_client_list)
            if np.sum(distribution_difference) == 0:
                distribution_difference = np.array([0 for _ in range(len(distribution_difference))])
            else:
                # 正则化
                distribution_difference = distribution_difference / np.sum(distribution_difference)

            Icid = dict(enumerate(distribution_difference))

            # calculate the quality of client
            model_quality = {}
            for client in qualified_client_list:
                model_quality[client] = datasize[client] / Icid[client]
            
            client_bid = {}
            for client in qualified_client_list:
                client_bid[client] = random.randint(1, 3) 
            server_budget = args.budget

            # incentive mechanism
            rank = {}            
            if args.im == 'quality_aware':    
                for client in qualified_client_list:
                    client_bid[client] = random.randint(1, 3) 
                    rank[client] = model_quality[client] / client_bid[client]
                sorted_rank = dict(sorted(rank.items(), key=lambda item: item[1], reverse = True))
                # 获胜客户端数量
                k_value = find_smallest_w(sorted_rank, model_quality, server_budget)
                qualified_client_list = list(itertools.islice(sorted_rank.keys(), k_value))

            
            if args.im == 'knapsack':
                for client in qualified_client_list:
                    if client >= args.n_normal_client:
                        client_bid[client] = 1
                    else:
                        client_bid[client] = random.randint(3, 5) 
                    rank[client] = datasize[client] / client_bid[client] 
                sorted_rank = dict(sorted(rank.items(), key=lambda item: item[1], reverse = True))
                k_value = 0
                total_bid =0
                for idx, value in sorted_rank.items():
                    total_bid += client_bid[idx]
                    if total_bid > server_budget:
                        break
                    k_value += 1
                qualified_client_list = list(itertools.islice(sorted_rank.keys(), k_value))

            if args.im == 'price_first':
                for client in qualified_client_list:
                    if client >= args.n_normal_client:
                        client_bid[client] = 1
                    else:
                        client_bid[client] = random.randint(3, 5) 
                sorted_rank = dict(sorted(client_bid.items(), key=lambda item: item[1], reverse = True))
                k_value = 0
                total_bid = 0
                for idx, value in sorted_rank.items():
                    total_bid += client_bid[idx]
                    if total_bid > server_budget:
                        break
                    k_value += 1
                qualified_client_list = list(itertools.islice(sorted_rank.keys(), k_value))
            
            if args.im == 'random':
                total_bid = 0
                winner = []
                for client in qualified_client_list:
                    if client >= args.n_normal_client:
                        client_bid[client] = 1
                    else:
                        client_bid[client] = random.randint(3, 5) 
                random_client = qualified_client_list
                random.shuffle(random_client)
                for client in random_client:
                    if total_bid + client_bid[client] <= server_budget:
                        winner.append(client)
                        total_bid += client_bid[client]
                qualified_client_list = winner
            
            winner_sum = len(qualified_client_list)
            print(f'winner clients this round : {winner_sum}')
            
            # calculate the aggregation weight
            # 求合格客户端的本地数据总量
            total_data_points = 0
            for r in qualified_client_list:
                if r < args.n_normal_client:
                    total_data_points += len(net_dataidx_map[r])
                else:
                    total_data_points += freerider_total_data[r-args.n_normal_client]
            # 求各个客户端的数据大小占总数据大小的比例,即n_k
            fed_avg_freqs = {}
            for client_id in qualified_client_list:
                fed_avg_freqs[client_id] = datasize[client_id] / total_data_points

            if args.im == 'quality_aware':
                a = args.disco_a
                b = args.disco_b
                total_normalizer = 1
                tmp = {}
                # idx = 0
                for client_id in qualified_client_list:
                    tmp[client_id] = fed_avg_freqs[client_id] - a * distribution_difference[client_id] + b
                    # idx = idx + 1
                if sum(tmp[i] > 0 for i in qualified_client_list) > 0:
                # if np.sum(tmp[i]>0 for i in qualified_client_list)>0:         
                    fed_avg_freqs = tmp
                    # fed_avg_freqs[fed_avg_freqs<0.0]=0.0
                    for k,v in fed_avg_freqs.items():
                        if v < 0.0:
                            fed_avg_freqs[k] = 0.0
                    total_normalizer = sum([fed_avg_freqs[r] for r in qualified_client_list])
                for r in qualified_client_list:
                    fed_avg_freqs[r] = fed_avg_freqs[r] / total_normalizer
            
            ############### Model Aggregation ############### 
            num_a = 0
            for net_id in qualified_client_list:
                # 获得客户端模型参数
                net_para = nets[net_id].state_dict() 
                if num_a == 0:
                # 若为第一个客户端，则直接按权重存储
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                    num_a = 1
                else:
                # 其他客户端则乘以权重后添加到全局模型
                    for key in net_para:
                        global_w[key] = global_w[key] + net_para[key] * fed_avg_freqs[net_id]
            
            # 将更新后的全局模型参数global_w加载到全局模型global_model中
            global_model.load_state_dict(global_w)

            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device, dataset = args.dataset)
            record_test_acc_list.append(test_acc)
            record_winner.append(winner_sum)

            writer.add_scalar('testing accuracy',
                            test_acc,
                            round)
            if(best_test_acc<test_acc):
                best_test_acc=test_acc
                logger.info('New Best best test acc:%f'% test_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
            print('>> Global Model Test accuracy: %f, Best: %f' % (test_acc, best_test_acc))
           
            mkdirs(args.modeldir+'fedavg/')
            global_model.to('cpu')

            m_previous = copy.deepcopy(global_model)

            torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')
        
        np.savetxt(acc_path, np.array(record_test_acc_list))
        np.savetxt(precision_path, np.array(record_precision))
        np.savetxt(recall_path, np.array(record_recall))
        np.savetxt(winner_path, np.array(record_winner))

    if args.alg == 'MG':
        
        historical_utility = torch.zeros(args.n_freerider+args.n_normal_client)
        decisions = torch.randint(0, 2, (args.n_freerider+args.n_normal_client,)).float() * 2 - 1
        utility = torch.zeros(args.n_freerider+args.n_normal_client)
        utility_negative = torch.zeros(args.n_freerider+args.n_normal_client)

        for round in range(n_comm_rounds):

            logger.info("in comm round:" + str(round))
            print("In communication round:" + str(round))
            party_list_this_round = copy.deepcopy(party_list_rounds[round])
            if args.n_freerider != 0:
                print(f'Clients this round : {party_list_this_round}')

                # Free-riders' local data distribution
                freerider_total_data = []
                for j in range(args.n_freerider):
                    freerider_cls_count = [random.randint(450, 520) for _ in range(n_classes)]  
                    freerider_total_data.append(sum(freerider_cls_count))
                    traindata_cls_counts = np.vstack([traindata_cls_counts, freerider_cls_count])

            datasize = {}
            for client in range(len(party_list_this_round)):
                datasize[client] = sum(traindata_cls_counts[client])
        
            # Model Initialization
            # state_dict()返回一个包含模型权重的字典，仅包含模型的参数值而不包含模型的结构或其他信息
            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            # Free-riders’ local update
            if args.n_freerider != 0:
                noise_type = 'disguised'
                list_std = [0, std_original]
                list_power = [0, args.power]
                if round != 0:
                    list_std = get_std(global_model, m_previous, noise_type)
                for net_id, net in nets.items():
                    if net_id in range(args.n_normal_client, len(party_list_this_round)):
                        nets[net_id] = linear_noising(
                            net,
                            list_std,
                            list_power,
                            max(i, 1),
                            noise_type,
                            args.coef,
                        )
            
            # Normal client's local update 同时会更新本地权重
            # test_acc_list 每个本地客户端模型的测试精度
            test_acc_list, _ = local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, device=device, datasize = datasize)

            # free-riders detection
            qualified_client_list = []
            if args.n_freerider != 0:
                    
                # 1. 计算attendance
                count_ones = torch.sum(decisions == 1).item()
                attendance = 2 * count_ones - len(decisions)
                # 2. minority side determination
                if attendance > 0:
                    win = -1
                else:
                    win = 1

                for client in party_list_this_round:
                    if decisions[client] == win:
                        qualified_client_list.append(client)
                # 3. update the utility function
                utility = update_utility(utility, decisions, attendance, len(party_list_this_round))
                utility_negative = update_utility(utility, -decisions, attendance, len(party_list_this_round))
                # 归一化
                min_utility = torch.min(utility)
                utility_shifted = utility - min_utility + 1e-10
                min_utility_negative = torch.min(utility_negative)
                utility_negative_shifted = utility_negative - min_utility_negative + 1e-10
                utility_sum = torch.sum(utility_shifted)
                utility_negative_sum = torch.sum(utility_negative_shifted)
                utility_normalized = utility_shifted / utility_sum
                utility_negative_normalized = utility_negative_shifted / utility_negative_sum

                # for i in range(len(decisions)):
                #     if decisions[i].item() != win:
                #         if random.randint(0,len(decisions)-1) < abs(attendance) - 1:
                #             decisions[i] = (decisions[i] + 1) % 2

                
                exp = torch.exp(args.beta_mg * utility_normalized)
                exp_negative = torch.exp(args.beta_mg * utility_negative_normalized)
                probabilities = exp / (exp + exp_negative)
                if torch.isnan(probabilities).any():
                    probabilities = torch.nan_to_num(probabilities, nan=1.0/len(decisions))
                decisions = torch.bernoulli(probabilities).float() * 2 - 1
            else:
                qualified_client_list = party_list_this_round

            # calculate the degree of class-imbalance for qualified client
            # traindata_cls_counts[j,m]表示第j个客户端或子网络中属于类别m的样本数量
            traindata_array = []
            for client in party_list_this_round:
                traindata_array.append(traindata_cls_counts[client])
            traindata_array = np.array(traindata_array)
            distribution_difference = get_distribution_difference(traindata_array, participation_clients=qualified_client_list)
            if np.sum(distribution_difference) == 0:
                distribution_difference = np.array([0 for _ in range(len(distribution_difference))])
            else:
                # 正则化
                distribution_difference = distribution_difference / np.sum(distribution_difference)

            Icid = dict(enumerate(distribution_difference))

            # calculate the quality of client
            # model_quality = {}
            # for client in qualified_client_list:
            #     model_quality[client] = datasize[client] / Icid[client]
            
            # client_bid = {}
            # for client in qualified_client_list:
            #     client_bid[client] = random.randint(1, 3) 
            # server_budget = args.budget                   
            
            # calculate the aggregation weight
            # 求合格客户端的本地数据总量
            total_data_points = 0
            for r in qualified_client_list:
                if r < args.n_normal_client:
                    total_data_points += len(net_dataidx_map[r])
                else:
                    total_data_points += freerider_total_data[r-args.n_normal_client]
            # 求各个客户端的数据大小占总数据大小的比例,即n_k
            fed_avg_freqs = {}
            for client_id in qualified_client_list:
                fed_avg_freqs[client_id] = datasize[client_id] / total_data_points

            
            ############### Model Aggregation ############### 
            num_a = 0
            for net_id in qualified_client_list:
                # 获得客户端模型参数
                net_para = nets[net_id].state_dict() 
                if num_a == 0:
                # 若为第一个客户端，则直接按权重存储
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                    num_a = 1
                else:
                # 其他客户端则乘以权重后添加到全局模型
                    for key in net_para:
                        global_w[key] = global_w[key] + net_para[key] * fed_avg_freqs[net_id]
            
            # 将更新后的全局模型参数global_w加载到全局模型global_model中
            global_model.load_state_dict(global_w)

            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device, dataset = args.dataset)
            record_test_acc_list.append(test_acc)

            writer.add_scalar('testing accuracy',
                            test_acc,
                            round)
            if(best_test_acc<test_acc):
                best_test_acc=test_acc
                logger.info('New Best best test acc:%f'% test_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
            print('>> Global Model Test accuracy: %f, Best: %f' % (test_acc, best_test_acc))
           
            mkdirs(args.modeldir+'fedavg/')
            global_model.to('cpu')

            m_previous = copy.deepcopy(global_model)

            torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')
        
        np.savetxt(acc_path, np.array(record_test_acc_list))
        np.savetxt(precision_path, np.array(record_precision))
        np.savetxt(recall_path, np.array(record_recall))
    
    
    
    print('>> Global Model Best accuracy: %f' % best_test_acc)
    print(args)
    print(now_time)