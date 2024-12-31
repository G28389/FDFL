import numpy as np

loaded_array = np.load("./logs/mnist/acc_list/2023-10-30-1606-36.npy")
for i in loaded_array:
    print(i)
