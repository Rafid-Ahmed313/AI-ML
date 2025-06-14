
from torch_geometric.data import Dataset, Data 
import numpy as np 
import os 
from rdkit.Chem import rdmolops
from tqdm import tqdm
from rdkit import Chem
import pandas as pd 
import torch 

class MoleculeDataset(Dataset):

    def __init__(self,root,transform=None, pre_transform=None):

        super(MoleculeDataset,self).__init__(root,transform,pre_transform)

    @property 
    def raw_file_names(self):
        return "HIV.csv"


    @property 
    def processed_file_names(self):
        return "not_implemented.pt"

    def download(self):
       pass

    def process(self):
       self.data = pd.read_csv(self.raw_paths[0])
       for index,mol in tqdm(self.data.interrows(), total = self.data.shape[0]):
           mol_obj = Chem.MolFromSmiles(mol["smiles"])

           node_feats = self._get_node_features(mol_obj)

           edge_feats = self._get_edge_features(mol_obj)

           edge_index = self._get_adjacency_info(mol_obj)

           label = self._get_labels(mol["HIV_active"])

           data = Data(x = node_feats,edge_index=edge_index,edge_attr = edge_feats,y=label,smiles=mol["smiles"])


           if self.test:
               torch.save(data,os.path.join(self.processed_dir, f"data_test_{index}.pt"))
            
           else:
               torch.save(data,os.path.join(self.processed_dir,f"data_{index}.pt"))





        

    def _get_node_features(self,mol):
        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            node_feats.append(atom.GetAtomicNum())
            node_feats.append(atom.GetDegree())
            node_feats.append(atom.GetFormalCharge())
            node_feats.append(atom.GetHybridization())
            node_feats.append(atom.GetIsAromatic())
            node_feats.append(atom.GetTotalNumHs())
            node_feats.append(atoms.GetNumRadicalElectrons())
            node_feats.append(atoms.IsInRing())
            node_feats.append(atoms.GetChiralTag())
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats,dtype=torch.float)

        
    def _get_edge_features(self,mol):
       all_edge_feats = []

       for bond in mol.GetBonds():
           edge_feats = []
           edge_feats.append(bond.GetBondTypeAsDouble())
           edge_feats.append(bond.IsInRing())
           all_edge_feats += [edge_feats,edge_feats]

       all_edge_feats = np.asarray(all_edge_feats)
       return torch.tensor(all_edge_feats,dtype=torch.float)


    def _get_adjacency_info(self,mol):
       edge_indices = []
       for bond in mol.GetBonds():
           i = bond.GetBeginAtomIdx()
           j = bond.GetEndAtomIdx()
           edge_indices += [[i,j],[j,i]]


       edge_indices = torch.tensor(edge_indices)
       edge_indices = edge_indices.t().to(torch.long).view(2,-1)
       return edge_indices

    def len(self):
       return self.data.shape[0]

    def get(self,idx):
       if self.test:
           data = torch.load(os.path.join(self.processed_dir,f"data_test_{idx}.pt"))

       else:
           data = torch.load(os.path.join(self.processed_dir,f"data_{idx}.pt"))


       return data


        

