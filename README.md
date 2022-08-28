# Affinity-prediction: Prediction affinity of protein-protein interactions from pdb files.

---

## Overview 

- ### 🧊 voxel_representation : 
  Library that allow to process the pdb files, determine the interacting atoms of a protein-protein
  interface and voxelize these selected atoms in a cube whose center is  the center of mass of the 
  atoms. In this process the **biopython** library is leveraged with its SMCRA (Structure/Chain/Residue/Atom) 
  architecture. A dataset is finally built with **pyTorch**. 
   
- ### Neural_net :
  We define the CNN and functions for training and testing the model. Due to the unavailability of most of the 
  database at the moment of finishing this project, the DataLoader class was not used but is recommended once
  the clusters are no longer in maintenance. 
  

- ### 🛠 Utilities : 
  Various functions are defined for general tasks throughout the project.

---

## Configuration Options
- A json file allows to modify various parameters including: 
  - The dimensions of the voxel representation of the protein-protein interface. 
  - The dictionary for a one-hot representation of the different atoms found in a voxel. For example, one could add 'P' 
    for phosphorus atoms if the dataset contains kinase interactions.
  - The cutoff distance to define two interacting atoms from different chains. 
  

- **Testing options**: 
    In order to test the trained model two different metrics are available, namely computing the correlation coefficient
    between the list of Kd values from the testing set and the computed Kd values and computing the relative absolute error.
    The difficulty is to have a metric that can deal with Kd values having very different orders of magnitude. 


- To select the atoms taking part in protein-protein interactions two different methods can be applied:
  1. Using the distance between the CA of two residues from two different chains as a cutoff distance (cf. json file). 
     *This method is the one implemented at the moment.* 
  2. Using the distance between atoms of two different chains as a cutoff distance (cf. json file).


