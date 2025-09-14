import torch
import numpy as np
import sys
import argparse as agp
import random
import os
import time
import torch.utils.data.sampler as sampler
from TASK2_CNN_model import CNNAttentionModel
from TASK2_data_generator import dataSet
import pickle
import timeit
import torch.nn as nn
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, accuracy_score, matthews_corrcoef
import pandas as pd
from numpy import int64 as np_int64
from lookahead import Lookahead
from torch_optimizer import RAdam





def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def metrics(correct_labels, predicted_labels, predicted_scores):
    ACC = accuracy_score(correct_labels, predicted_labels)
    AUC = roc_auc_score(correct_labels, predicted_scores)
    CM = confusion_matrix(correct_labels, predicted_labels)
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    Rec = TP / (TP + FN)
    Pre = TP / (TP + FP)
    F1 = 2 * Pre * Rec / (Pre + Rec)
    MCC = matthews_corrcoef(correct_labels, predicted_labels)
    precision, recall, _ = precision_recall_curve(correct_labels, predicted_scores)
    PRC = auc(recall, precision)
    return ACC, AUC, Rec, Pre, F1, MCC, PRC

def stack_fn(batch):
    local_features, all_seq_features_A, all_seq_features_B, labels = [], [], [], []

    for local, seq_A, seq_B, label in batch:
        local_features.append(local)
        all_seq_features_A.append(seq_A)
        all_seq_features_B.append(seq_B)
        labels.append(label)

    N = len(labels)
    local_dim = 1024
    protein_dim = 1024

    locals_len = max(local.shape[0] for local in local_features)
    proteins_len_A = max(protein_A.shape[0] for protein_A in all_seq_features_A)
    proteins_len_B = max(protein_B.shape[0] for protein_B in all_seq_features_B)

    locals_new = np.zeros((N, locals_len, local_dim))
    proteins_new_A = np.zeros((N, proteins_len_A, protein_dim))
    proteins_new_B = np.zeros((N, proteins_len_B, protein_dim))
    labels_new = np.zeros(N, dtype=np_int64)

    for i, (local, protein_A, protein_B, label) in enumerate(zip(local_features, all_seq_features_A, all_seq_features_B, labels)):
        locals_new[i, :local.shape[0], :] = local
        proteins_new_A[i, :protein_A.shape[0], :] = protein_A
        proteins_new_B[i, :protein_B.shape[0], :] = protein_B
        labels_new[i] = label

    locals_new = np.stack(locals_new)
    proteins_new_A = np.stack(proteins_new_A)
    proteins_new_B = np.stack(proteins_new_B)
    labels_new = np.stack(labels_new)

    return locals_new, proteins_new_A, proteins_new_B, labels_new

def main(seed):
    init_seeds(seed)

    """ Load preprocessed data """
    with open('./data_cache/TASK2_all_train_samples.pkl', 'rb') as f:
        balanced_train_list = pickle.load(f)

    with open('./data_cache/TASK2_all_test_samples.pkl', 'rb') as f:
        test_list = pickle.load(f)

    train_samples = sampler.SubsetRandomSampler(balanced_train_list)
    dev_samples = sampler.SubsetRandomSampler(test_list)

    batch_size = 128

    train_loader = torch.utils.data.DataLoader(all_dataSet, batch_size=batch_size,
                                               sampler=train_samples,
                                               num_workers=0, collate_fn=stack_fn, drop_last=False)

    valid_loader = torch.utils.data.DataLoader(all_dataSet, batch_size=batch_size,
                                               sampler=dev_samples,
                                               num_workers=0, collate_fn=stack_fn, drop_last=False)

    """ Create model, trainer, and tester """
    local_dim = 1024
    hid_dim = 1024
    kernel_size = 7
    dropout = 0.1
    lr = 5e-4
    weight_decay = 1e-4
    iteration = 50


    model = CNNAttentionModel(local_dim, hid_dim, 3, kernel_size, dropout, device)
    model.to(device)

    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]


    optimizer_inner = RAdam([{'params': weight_p, 'weight_decay': weight_decay}, 
                         {'params': bias_p, 'weight_decay': 0}], lr=lr)
    optimizer = Lookahead(optimizer_inner, k=5, alpha=0.5)


    os.makedirs(('./output/result'), exist_ok=True)
    os.makedirs(('./output/model'), exist_ok=True)
    file_AUCs = './output/result/TASK2_CNN.txt'
    file_model = './output/model/TASK2_CNN'

    AUCs = ('Epoch\tTime1(sec)\tTime2(sec)\tLoss_train\tACC_dev\tAUC_dev\tRec_dev\tPre_dev\tF1_dev\tMCC_dev\tPRC_dev')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    print('Training...')
    print(AUCs)

    max_MCC_dev = 0

    Epsion = 0.2
    for epoch in range(1, iteration + 1):
        start = timeit.default_timer()


        model.train()
        loss_total = 0

        for batch_idx, batch in enumerate(train_loader):
            local_features_A, all_seq_features_A, all_seq_features_B, labels = batch


            local_features_A = torch.tensor(local_features_A, dtype=torch.float32).to(device)
            all_seq_features_A = torch.tensor(all_seq_features_A, dtype=torch.float32).to(device)
            all_seq_features_B = torch.tensor(all_seq_features_B, dtype=torch.float32).to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)


            weights = []
            batch_labels = labels.cpu().tolist()
            for label in range(2):
                weights.append(1 - (batch_labels.count(label) / len(batch_labels)) + Epsion)
            class_weights = torch.FloatTensor(weights).to(device)


            criterion = nn.CrossEntropyLoss(weight=class_weights)

            optimizer.zero_grad()

            outputs = model(local_features_A,all_seq_features_A, all_seq_features_B, batch_idx=batch_idx)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            loss_total += loss.item()

        end1 = timeit.default_timer()
        time1 = end1 - start

        model.eval()
        correct_labels_valid, predicted_labels_valid, predicted_scores_valid = [], [], []

        with torch.no_grad():
            for batch in valid_loader:
                local_features_A, all_seq_features_A, all_seq_features_B, labels = batch

                local_features_A = torch.tensor(local_features_A, dtype=torch.float32).to(device)
                all_seq_features_A = torch.tensor(all_seq_features_A, dtype=torch.float32).to(device)
                all_seq_features_B = torch.tensor(all_seq_features_B, dtype=torch.float32).to(device)
                labels = torch.tensor(labels, dtype=torch.long).to(device)

                outputs = model(local_features_A, all_seq_features_A, all_seq_features_B)
                predicted_scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()
                correct_labels = labels.cpu().numpy()

                correct_labels_valid.extend(correct_labels)
                predicted_labels_valid.extend(predicted_labels)
                predicted_scores_valid.extend(predicted_scores)


        ACC_dev, AUC_dev, Rec_dev, Pre_dev, F1_dev, MCC_dev, PRC_dev = metrics(correct_labels_valid, predicted_labels_valid, predicted_scores_valid)

        end2 = timeit.default_timer()
        time2 = end2 - end1

        AUCs = [epoch, time1, time2, loss_total, ACC_dev, AUC_dev, Rec_dev, Pre_dev, F1_dev, MCC_dev, PRC_dev]
        with open(file_AUCs, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

        if MCC_dev > max_MCC_dev:
            last_improve = epoch
            print('Last improved: %s' % last_improve)
            torch.save(model.state_dict(), file_model)
            max_MCC_dev = MCC_dev

if __name__ == "__main__":
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    # Load preprocessed data
    all_encode_file_A = './data_cache/TASK2_TPhos_dataB_encode_data_all_ProteinAA.pkl'
    all_encode_file_B = './data_cache/TASK2_TPhos_dataB_encode_data_all_ProteinBB.pkl'
    all_list_file_A = './data_cache/TASK2_TPhos_dataB_protein_list_all_AA.pkl'
    all_list_file_B = './data_cache/TASK2_TPhos_dataB_protein_list_all_BB.pkl'

    window_size = 15
    all_dataSet = dataSet(window_size, all_encode_file_A, all_encode_file_B, all_list_file_A, all_list_file_B)

    SEED = 1  #7
    main(SEED)
