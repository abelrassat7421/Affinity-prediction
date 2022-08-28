import sys

import numpy as np

import json
import os

from sklearn.model_selection import train_test_split

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

def get_vox_size():
    """
    :return: (int) The length of the ndarray for voxelization obtained from dividing the grid size by the resolution
             size. These last two constants are defined in the json file in angstroms
    """
    GRID_SIZE = CONFIG_PROPERTIES["grid_config"]["grid_size"]
    GRID_RESOLUTION = CONFIG_PROPERTIES["grid_config"]["grid_resolution"]

    VOX_SIZE = GRID_SIZE / GRID_RESOLUTION

    # GRID_SIZE/ GRID_RESOLUTION should return a natural number
    if round(VOX_SIZE) != VOX_SIZE or GRID_SIZE < 0 or GRID_RESOLUTION < 0:
        sys.exit("CONSTANT Error: The grid_size and the grid resolution must be natural numbers as well as the " \
                 "the grid_size over the grid_resolution. Apply modifications to the json file.")

    return int(VOX_SIZE)


def read_labels(file_path):
    """
    :param file_path: relative path to file containing names of the pdbs with the interacting chains and the associated Kd values.
    :return: tuple with the names of the pdb with the interacting chains (list of strings) and the associated
     Kd values (list of floats)
    """

    with open(file_path, 'r') as f:
        data_split = [x.split() for x in f.readlines()]
        data_split = [(x[0], float(x[1])) for x in data_split]

    X = [x[0] for x in data_split]
    y = [x[1] for x in data_split]

    return X, y


def relative_err(real_list, comp_list):
    """
    :param real_list: (double or float list) list with the real values of a given parameter. Should be the same size as comp_list.
    :param comp_list: (double or float list) list with the computed values of the same parameter. Should be the same size as real_list.
    :return: (double or float list) list with the relative errors.
    """
    return [np.abs(real_list[i] - comp_list[i]) / real_list[i] for i in range(len(real_list))]


def split_from_training(train_pdbs='../data/train_pdbs.txt'):
    with open(train_pdbs, 'r') as f:
        data_split = [x.split() for x in f.readlines()]
        # casting to have the KDs as floats -> a list of a list with a string and a float
        data_split = [(x[0], float(x[1])) for x in data_split]

    # transform the data_split into X_train and y_train
    x_train = [x[0] for x in data_split]
    y_train = [x[1] for x in data_split]

    return train_test_split(x_train, y_train, test_size=0.2, random_state=8)
