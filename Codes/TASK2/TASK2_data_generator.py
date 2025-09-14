import torch
import torch.utils.data as data
import numpy as np
import pickle

class dataSet(data.Dataset):
    def __init__(self, window_size, encode_file_A=None, encode_file_B=None, protein_list_A_file=None,
                 protein_list_B_file=None):
        super(dataSet, self).__init__()


        with open(encode_file_A, "rb") as fp_enc_A:
            self.all_encodes_A = pickle.load(fp_enc_A)


        with open(encode_file_B, "rb") as fp_enc_B:
            self.all_encodes_B = pickle.load(fp_enc_B)


        with open(protein_list_A_file, "rb") as list_label_A:
            self.protein_list_A = pickle.load(list_label_A)



        with open(protein_list_B_file, "rb") as list_label_B:
            self.protein_list_B = pickle.load(list_label_B)

        self.window_size = window_size

        self.index_list = []
        self.label_list = []

    def __getitem__(self, index):

        protein_info_A = self.protein_list_A[index]
        id_idx_A, protein_name_A, seq_A, phosphorylation_position, phosphorylated_aa, label = protein_info_A

        protein_info_B = self.protein_list_B[index]
        id_idx_B, protein_name_B, seq_B = protein_info_B


        all_seq_features_A = self.all_encodes_A[id_idx_A]
        all_seq_features_B = self.all_encodes_B[id_idx_B]


        win_start = max(0, phosphorylation_position - 1 - self.window_size)
        win_end = min(len(all_seq_features_A) - 1, phosphorylation_position - 1 + self.window_size)

        local_features_A = all_seq_features_A[win_start:win_end + 1]
        label = np.array(label, dtype=np.float32)
        
        all_seq_features_A = np.array(all_seq_features_A).astype(float)
        local_features_A = np.array(local_features_A).astype(float)
        all_seq_features_B = np.array(all_seq_features_B).astype(float)
        

        self.index_list.append(index)
        self.label_list.append(label)
        

        print(f"Index: {index}, Label: {label}")
        return local_features_A, all_seq_features_A, all_seq_features_B, label

    def __len__(self):
        return len(self.protein_list_A)

    

    def get_index_label_mapping(self):
        return self.index_list, self.label_list