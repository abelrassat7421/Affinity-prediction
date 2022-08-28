import torch

from torch.utils.data import DataLoader

import voxel_representation as vox
import Neural_net as nen
import Utilities as ut


def main():
    x_train, y_train = ut.read_labels('../data/train_pdbs.txt')
    x_test, y_test = ut.read_labels('../data/test_pdbs.txt')

    # We instantiate the dataset classes
    voxelized_train_data = vox.VoxelizedPPIDataset(pdb_names=x_train, Kd_vals=y_train)
    # train_dataloader = DataLoader(voxelized_train_data, batch_size=128, shuffle=False) -> needs to be done on the cluster with the entire dataset (will have to uncomment lines in train_model)
    voxelized_test_data = vox.VoxelizedPPIDataset(pdb_names=x_test, Kd_vals=y_test)
    # train_dataloader = DataLoader(voxelized_test_data, batch_size=128, shuffle=False)  -> needs to be done on the cluster with the entire dataset

    net = nen.train_model(voxelized_train_data)
    # net = nen.train_model(train_dataloader)

    # saving the trained model
    PATH = './ppi_affinity_net.pth'
    torch.save(net.state_dict(), PATH)

    # testing
    correlation_coeff, pvalue = nen.testing_model(voxelized_test_data, path_trained_model='./ppi_affinity_net.pth')
    print(f'Pearson correlation coefficient: {correlation_coeff}, p-value: {pvalue}')


if __name__ == '__main__':
    main()
