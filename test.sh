################################### fair ###################################
# CUDA_VISIBLE_DEVICES=0,1 python main_fullset.py --distribution_aware='sub'  --disco_a 0.5 --disco_b 0.1 --disco_partition 0 --beta=0.5 --n_normal_client=29 --n_freerider=1 --epochs=20 --comm_round=100 --lr=0.01 --use_project_head 0 --dataset=cifar10 --device cuda:0 --model=simple-cnn --budget=20 --alg=fair --partition=noniid --logdir='./logs/' --datadir='dataset/cifar10'
CUDA_VISIBLE_DEVICES=1 python main_fullset.py --distribution_aware='sub'  --disco_a 0.5 --disco_b 0.1 --disco_partition 0 --beta=0.5 --n_normal_client=29 --n_freerider=1 --epochs=20 --comm_round=100 --lr=0.01 --use_project_head 0 --dataset=mnist --device cuda:1 --model=simple-cnn-mnist --budget=20 --alg=fair --partition=noniid --logdir='./logs/' --datadir='dataset/mnist'
################################### FDFL ###################################
# simplecnn-mnist
# CUDA_VISIBLE_DEVICES=1 python main_fullset.py --distribution_aware='sub'  --disco_a 0.5 --disco_b 0.1 --disco_partition 0 --beta=0.5 --n_normal_client=29 --n_freerider=1 --epochs=20 --comm_round=100 --lr=0.01 --use_project_head 0 --dataset=mnist --device cuda:1 --model=simple-cnn-mnist --free_rider_detection=True --alg=FDFL --partition=noniid --logdir='./logs/' --datadir='dataset/minist'
# mlp-mnist
# CUDA_VISIBLE_DEVICES=0 python main_fullset.py --distribution_aware='sub'  --disco_a 0.5 --disco_b 0.1 --disco_partition 0 --beta=0.5 --n_normal_client=29 --n_freerider=1 --epochs=20 --comm_round=100 --lr=0.01 --use_project_head 0 --dataset=mnist --device cuda:0 --model=mlp --free_rider_detection=True --alg=FDFL --partition=noniid --logdir='./logs/' --datadir='dataset/cifar10'
# resnet18-cifar10
# resnet50-cifar10
# resnet50-cifar100
# simplecnn-cifar10
# CUDA_VISIBLE_DEVICES=0 python main_fullset.py --distribution_aware='sub'  --disco_a 0.5 --disco_b 0.1 --disco_partition 0 --beta=0.5 --n_normal_client=29 --n_freerider=1 --epochs=20 --comm_round=100 --lr=0.01 --use_project_head 0 --dataset=cifar10 --device cuda:0 --model=simple-cnn --free_rider_detection=True --alg=FDFL --partition=noniid --logdir='./logs/' --datadir='dataset/cifar10'