import torch
from torch.utils.data import Dataset
import numpy as np

import os

from Bio.PDB.PDBParser import PDBParser

import warnings

import json

import Utilities as ut

# Ignore warnings
warnings.filterwarnings("ignore")

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

def read_atoms(name_pdb):
    """
    :param name_pdb: name of the pdb file with the interacting chains (string)
    :return: tuple with the list of all atoms from the two selected interacting chains
    """
    # check if better import warnings package than use quiet flag
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    name = name_pdb[:4]
    file_path = f'../data/protonated_pdbs/{name}.pdb'
    structure = parser.get_structure(name_pdb, file_path)
    ch1, ch2 = structure[0][name_pdb[5]], structure[0][name_pdb[7]]
    atoms_ch1 = [atom for atom in ch1.get_atoms()]
    atoms_ch2 = [atom for atom in ch2.get_atoms()]

    return atoms_ch1, atoms_ch2


# Function not used for the moment. Here we are selecting atoms that are at a distance smaller than a certain cutoff
# distance with an atom from another chain
def get_interface_atoms(atoms_ch1, atoms_ch2):
    INTERACTION_DIST = CONFIG_PROPERTIES["interaction_dist"]["atom_distance"]
    interacting_atoms = []
    for atom_ch1 in atoms_ch1:
        for atom_ch2 in atoms_ch2:
            if np.linalg.norm(atom_ch1.get_coord() - atom_ch2.get_coord()) < INTERACTION_DIST:
                interacting_atoms += atom_ch1 + atom_ch2

    # converting to a set and back to a list to only have distinct atoms
    return list(set(interacting_atoms))


# Here we are selecting atoms from residues that have their CA at a distance smaller than a certain cutoff distance
# with the CA of a residue from another chain
def get_interface_atoms_entire_res(atoms_ch1, atoms_ch2):
    """
    :param atoms_ch1: all atoms form the first interacting chain
    :param atoms_ch2: all atoms form the second interacting chain
    :return: all atoms from residues that have their CA at a distance smaller than CA_DIST from the CA of a residue
        from the other chain
    """
    CA_DIST = CONFIG_PROPERTIES["interaction_dist"]["ca_distance"]
    interacting_atoms = []
    # need to extract the CA atoms to compare their distances from those of the opposite chain
    CA_atoms_ch1 = [atom for atom in atoms_ch1 if atom.get_name() == "CA"]
    CA_atoms_ch2 = [atom for atom in atoms_ch2 if atom.get_name() == "CA"]

    for CA_atom_ch1 in CA_atoms_ch1:
        for CA_atom_ch2 in CA_atoms_ch2:
            if np.linalg.norm(CA_atom_ch1.get_coord() - CA_atom_ch2.get_coord()) < CA_DIST:
                inter_atoms_ch1 = [atom for atom in CA_atom_ch1.get_parent().get_list()]
                inter_atoms_ch2 = [atom for atom in CA_atom_ch2.get_parent().get_list()]
                interacting_atoms += inter_atoms_ch1 + inter_atoms_ch2

    # converting to a set and back to a list to have distinct atoms
    return list(set(interacting_atoms))


def get_COM(atoms):
    """
    :param atoms: list of atoms
    :return: Center of mass of the atoms (ndarray)
    """
    atom_coordinates = np.array([atom.get_coord() for atom in atoms])
    return np.sum(atom_coordinates, axis=0) / np.shape(atom_coordinates)[0]


def voxelize_atoms(atoms):
    """
    :param atoms: list of atoms
    :return: normalized 4D grid encoding the presence of different types of atoms in 3D space
        i.e. (x, y, z, one-hot encoded atoms) (ndarray)
    """
    GRID_SIZE = CONFIG_PROPERTIES["grid_config"]["grid_size"]
    GRID_RESOLUTION = CONFIG_PROPERTIES["grid_config"]["grid_resolution"]
    ONE_HOT_DICT = CONFIG_PROPERTIES["grid_config"]["one_hot_dict"]

    # need to convert the values of the dictionary to nd.arrays
    for keys in ONE_HOT_DICT:
        ONE_HOT_DICT[keys] = np.asarray(ONE_HOT_DICT[keys])

    VOX_SIZE = ut.get_vox_size()

    voxelized_interface = np.zeros((VOX_SIZE, VOX_SIZE, VOX_SIZE, len(ONE_HOT_DICT)))
    COM = get_COM(atoms)

    for atom in atoms:
        x, y, z = ((atom.get_coord() - COM + GRID_SIZE / 2) / GRID_RESOLUTION).astype(int)

        # ignore atoms that are outside the grid
        if 0 <= x < VOX_SIZE and 0 <= y < VOX_SIZE and 0 <= z < VOX_SIZE:

            if not atom.element[0] in ONE_HOT_DICT.keys():
                warning_message = "The atom element" + atom.element[0] + "is not considered." + "This element can be" \
                                  "considered by adding it to the one_hot_dict and by incrementing by one the value" \
                                  "of 'different_atoms' in the json file."
                warnings.warn(warning_message)
                continue

            one_hot = ONE_HOT_DICT[atom.element[0]]

            # normalization
            nonzero_el = np.count_nonzero(voxelized_interface[x][y][z])
            if nonzero_el != 0:
                voxelized_interface[x][y][z] = (np.multiply(voxelized_interface[x][y][z], nonzero_el) + one_hot) / (
                        nonzero_el + 1)
            else:
                voxelized_interface[x][y][z] += one_hot

    return voxelized_interface


class VoxelizedPPIDataset(Dataset):
    """Voxelized PPI dataset."""

    # change the presentation for initialising the attributes of the class
    # here we also take the log base 10 of the Kds
    def __init__(self, pdb_names, Kd_vals, transform=None):
        """
        Args:
            pdb_names (list of strings): names of the pdbs with the chains involved in the PPI
            Kd_vals (list of floats): Kd values associated with the pdbs
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pdb_names = pdb_names
        self.log10_Kd_vals = np.log10(Kd_vals)  # becomes a ndarray
        self.transform = transform

    def __len__(self):
        return len(self.pdb_names)

    # samples of the dataset will be dictionaries
    def __getitem__(self, idx):
        # should we understand that idx can be a list or a tensor?
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pdb_name = self.pdb_names[idx]
        atoms_ch1, atoms_ch2 = read_atoms(pdb_name)
        voxelized_interface = voxelize_atoms(get_interface_atoms_entire_res(atoms_ch1, atoms_ch2))
        log10_Kd_val = self.log10_Kd_vals[idx]

        sample = {'voxelized_interface': voxelized_interface, 'log10_Kd_val': log10_Kd_val}

        if self.transform:
            sample = self.transform(sample)

        return sample
