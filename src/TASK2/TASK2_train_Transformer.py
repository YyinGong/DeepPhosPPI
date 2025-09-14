import torch
import numpy as np
import sys
import argparse as agp
import random
import os
import time
import torch.utils.data.sampler as sampler
from TASK2_Transformer_model import *  # dataSet
from TASK2_data_generator import *
import pickle
import timeit
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, \
    accuracy_score, matthews_corrcoef
import pandas as pd
from numpy import int64 as np_int64


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

    locals_len = 0
    proteins_len_A = 0
    proteins_len_B = 0
    N = len(labels)
    local_num = []
    protein_num_A = []
    protein_num_B = []

    local_dim = 1024
    protein_dim = 1024
    for local in local_features:
        local_num.append(local.shape[0])
        if local.shape[0] >= locals_len:
            locals_len = local.shape[0]
    for protein_A in all_seq_features_A:
        protein_num_A.append(protein_A.shape[0])
        if protein_A.shape[0] >= proteins_len_A:
            proteins_len_A = protein_A.shape[0]
    for protein_B in all_seq_features_B:
        protein_num_B.append(protein_B.shape[0])
        if protein_B.shape[0] >= proteins_len_B:
            proteins_len_B = protein_B.shape[0]

    locals_new = np.zeros((N, locals_len, local_dim))
    i = 0
    for local in local_features:
        a_len = local.shape[0]
        locals_new[i, :a_len, :] = local
        i += 1

    proteins_new_A = np.zeros((N, proteins_len_A, protein_dim))
    i = 0
    for protein_A in all_seq_features_A:
        a_len = protein_A.shape[0]
        proteins_new_A[i, :a_len, :] = protein_A
        i += 1

    proteins_new_B = np.zeros((N, proteins_len_B, protein_dim))
    i = 0
    for protein_B in all_seq_features_B:
        a_len = protein_B.shape[0]
        proteins_new_B[i, :a_len, :] = protein_B
        i += 1

    labels_new = np.zeros(N, dtype=np_int64)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    locals_new = np.stack(locals_new)
    proteins_new_A = np.stack(proteins_new_A)
    proteins_new_B = np.stack(proteins_new_B)
    labels_new = np.stack(labels_new)

    return locals_new, proteins_new_A, proteins_new_B, labels_new, local_num, protein_num_A, protein_num_B


def main(seed):
    init_seeds(seed)

    """Load preprocessed data."""

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

    """ create model ,trainer and tester """
    protein_dim = 1024
    local_dim = 1024
    hid_dim =  64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout =  0.1
    lr = 5e-3
    weight_decay = 1e-4
    decay_interval = 5
    lr_decay = 1.0
    iteration = 70
    kernel_size = 7

    encoder_A = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    encoder_B = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(local_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention,
                      PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder_A, encoder_B, decoder, device)

    model.to(device)

    trainer = Trainer(model, lr, weight_decay)
    tester = Tester(model)

    # Output files.
    os.makedirs(('./output/result'), exist_ok=True)
    os.makedirs(('./output/model'), exist_ok=True)
    file_AUCs = './output/result/TASK2-Transformer' + '.txt'
    file_model = './output/model/' + 'TASK2_Transformer'
    AUCs = ('Epoch\tTime1(sec)\tTime2(sec)\tLoss_train\tACC_dev\tAUC_dev\tRec_dev\tPre_dev\tF1_dev\tMCC_dev\tPRC_dev')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    # Start training.
    print('Training...')
    print(AUCs)

    max_MCC_dev = 0
    for epoch in range(1, iteration + 1):
        start = timeit.default_timer()
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(train_loader, device)

        end1 = timeit.default_timer()
        time1 = end1 - start

        correct_labels_valid, predicted_labels_valid, predicted_scores_valid = tester.test(valid_loader,
                                                                                           device)

        if torch.cuda.is_available():
            correct_labels_valid = torch.tensor(np.array(correct_labels_valid), dtype=torch.float64, device='cuda')
            predicted_labels_valid = torch.tensor(np.array(predicted_labels_valid), dtype=torch.float64, device='cuda')
            predicted_scores_valid = torch.tensor(np.array(predicted_scores_valid), dtype=torch.float64, device='cuda')

            correct_labels_valid_list = correct_labels_valid.cpu()
            predicted_labels_valid_list = predicted_labels_valid.cpu()
            predicted_scores_valid_list = predicted_scores_valid.cpu()

            ACC_dev, AUC_dev, Rec_dev, Pre_dev, F1_dev, MCC_dev, PRC_dev = metrics(
                correct_labels_valid_list, predicted_labels_valid_list, predicted_scores_valid_list)


        end2 = timeit.default_timer()
        time2 = end2 - end1
        AUCs = [epoch, time1, time2, loss_train, ACC_dev, AUC_dev, Rec_dev, Pre_dev, F1_dev, MCC_dev, PRC_dev]
        tester.save_AUCs(AUCs, file_AUCs)

        if MCC_dev > max_MCC_dev:
            last_improve = epoch
            print('last_improve: %s' % last_improve)
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


    #  """Load preprocessed data."""
    all_encode_file_A = './data_cache/TASK2_TPhos_dataB_encode_data_all_ProteinAA.pkl'
    all_encode_file_B = './data_cache/TASK2_TPhos_dataB_encode_data_all_ProteinBB.pkl'
    all_list_file_A = './data_cache/TASK2_TPhos_dataB_protein_list_all_AA.pkl'
    all_list_file_B = './data_cache/TASK2_TPhos_dataB_protein_list_all_BB.pkl'


    window_size = 15
    all_dataSet = dataSet(window_size, all_encode_file_A, all_encode_file_B, all_list_file_A,  all_list_file_B)

    SEED =  1
    main(SEED)


