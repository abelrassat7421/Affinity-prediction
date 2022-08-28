import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats.stats import pearsonr

import Utilities as ut

import torch.optim as optim

import os

import json

# ===========================

CWD = os.getcwd()
JSON_CONFIG_FILE_PATH = '%s/%s' % (CWD, 'config.json')

CONFIG_PROPERTIES = {}

# Open the config.json, parse the values and store them in dictionary
try:
    with open(JSON_CONFIG_FILE_PATH) as df:
        CONFIG_PROPERTIES = json.load(df)
except IOError as err:
    print(err)
    print('IOError: Unable to open config.json. Terminating execution')
    exit(1)


# ===========================

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        ONE_HOT_DICT = CONFIG_PROPERTIES["grid_config"]["one_hot_dict"]
        VOX_SIZE = ut.get_vox_size()

        # input grid: (1, 5, VOX_SIZE, VOX_SIZE, VOX_SIZE)
        self.conv1 = nn.Conv3d(len(ONE_HOT_DICT), 6, 5)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16, 5)
        self.fc1 = nn.Linear(16 * int(pow(((VOX_SIZE - 4) / 2 - 4) / 2, 3)), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Modifications to be made to have an instance of DataLoader as an argument in order to have mini-batches of
# adjustable sizes (for now batch_size=1)
def train_model(voxelized_train_data):
    """
    :param voxelized_train_data: (VoxelizedPPIDataset) On the cluster give an instance of DataLoader instead.
    :return: (Net) The trained network.
    """

    ONE_HOT_DICT = CONFIG_PROPERTIES["grid_config"]["one_hot_dict"]
    VOX_SIZE = ut.get_vox_size()

    net = Net()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training the network
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i in range(len(voxelized_train_data)):
        # for i, data in enumerate(train_dataloader, 0):

            sample = voxelized_train_data[i]

            # voxelized_interface, log10_Kd_val = data
            # voxelized_interface, log10_Kd_val = torch.from_numpy(voxelized_interface).float(), torch.tensor([log10_Kd_val]).float()

            voxelized_interface, log10_Kd_val = torch.from_numpy(sample['voxelized_interface']).float(), torch.tensor([sample['log10_Kd_val']]).float()

            voxelized_interface = voxelized_interface.resize(len(ONE_HOT_DICT), VOX_SIZE, VOX_SIZE, VOX_SIZE)  # check if this makes sense and is the right thing to do
            voxelized_interface = voxelized_interface.unsqueeze(0)  # batch_size is 1 for the moment (delete when using DataLoader argument for the function)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(voxelized_interface)
            loss = criterion(outputs, log10_Kd_val)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print(
                    f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

            # not all the pdbs have been loaded from the cluster thus we look at the first 5 here -> for testing purposes
            # if i == 5:
                # break

    print('Finished Training')

    return net


# do similar modifications as above to accept an instance of DataLoader as a parameter
def testing_model(voxelized_test_data, path_trained_model='./ppi_affinity_net.pth'):
    """
    :param voxelized_test_data: (VoxelizedPPIDataset) On the cluster give an instance of DataLoader instead.
    :param path_trained_model: path to pth file with the trained nn.
    :return: (float tuple) The product-moment correlation coefficient between the list of real Kd values from the
             testing set and the output Kd values from the nn. The second value is the p-value testing for non-correlation.

             Other option -> uncomment first return statement (double or float list) list with the relative errors of the Kd values.
    """

    ONE_HOT_DICT = CONFIG_PROPERTIES["grid_config"]["one_hot_dict"]
    VOX_SIZE = ut.get_vox_size()

    net = Net()
    net.load_state_dict(torch.load(path_trained_model))

    # We store the real Kd values from the testing set
    Kd_val = [pow(10, voxelized_test_data[i]['log10_Kd_val']) for i in range(len(voxelized_test_data))]
    outputs = []

    for i in range(len(voxelized_test_data)):

        voxelized_interface = torch.from_numpy(voxelized_test_data[i]['voxelized_interface']).float()
        voxelized_interface = voxelized_interface.resize(len(ONE_HOT_DICT), VOX_SIZE, VOX_SIZE,VOX_SIZE)
        voxelized_interface = voxelized_interface.unsqueeze(0)  # batch_size is 1 for the moment (delete when using DataLoader argument for the function)

        # add computed Kd value to the outputs from the training set
        outputs.append(pow(10, net(voxelized_interface)))

    # return ut.relative_err(Kd_val, outputs)

    # computing the Pearson correlation coefficient
    outputs = torch.Tensor(outputs).float()
    outputs.numpy()
    Kd_val = np.asarray(Kd_val)

    return pearsonr(Kd_val, outputs)
