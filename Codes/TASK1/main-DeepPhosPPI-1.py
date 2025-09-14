import torch
import numpy as np
import sys
import argparse as agp
import random
import os
import time
import torch.utils.data.sampler as sampler
from TASK1_model import *
from data_generator_new_TPhos import *
import pickle
import timeit
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, accuracy_score, matthews_corrcoef
import pandas as pd
from TASK1_model import todevice



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


def stack_fn_cnn_ddp(batch):
    indexs, all_seq_features, labels = [], [], []
    for i in batch:
        index, seq, label = cnn_dataSet[i]
        indexs.append(index)
        all_seq_features.append(seq)
        labels.append(label)

    locals_len = 0
    proteins_len = 0
    N = len(labels)

    local_dim = 1024
    protein_dim = 1024

    for protein in all_seq_features:
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]

    indexs_new = np.zeros(N, dtype=np.int64)
    i = 0
    for index in indexs:
        indexs_new[i] = index
        i += 1

    proteins_new = np.zeros((N, proteins_len, protein_dim))
    i = 0
    for protein in all_seq_features:
        a_len = protein.shape[0]
        proteins_new[i, :a_len, :] = protein
        i += 1

    labels_new = np.zeros(N, dtype=np.int64)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    indexs_new = np.stack(indexs_new)
    proteins_new = np.stack(proteins_new)
    labels_new = np.stack(labels_new)

    return proteins_new, indexs_new, labels_new

def stack_fn_cnn(batch):
    indexs, all_seq_features, labels = [], [], []
    for index, seq, label in batch:
        indexs.append(index)
        all_seq_features.append(seq)
        labels.append(label)

    locals_len = 0
    proteins_len = 0
    N = len(labels)

    local_dim = 1024
    protein_dim = 1024

    for protein in all_seq_features:
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]

    indexs_new = np.zeros(N, dtype=np.int64)
    i = 0
    for index in indexs:
        indexs_new[i] = index
        i += 1

    proteins_new = np.zeros((N, proteins_len, protein_dim))
    i = 0
    for protein in all_seq_features:
        # print(protein.shape)
        a_len = protein.shape[0]
        # print(a_len)
        proteins_new[i, :a_len, :] = protein
        i += 1

    labels_new = np.zeros(N, dtype=np.int64)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    indexs_new = np.stack(indexs_new)
    proteins_new = np.stack(proteins_new)
    labels_new = np.stack(labels_new)

    return proteins_new, indexs_new, labels_new


def main(seed):
    init_seeds(seed)

    """Load preprocessed data."""

    with open('./data_cache/TPhos_dataA_list.pkl', "rb") as fp:
        all_list = pickle.load(fp)


    with open('./data_cache/2_TPhos_train_samples.pkl', 'rb') as f:
        balanced_train_list = pickle.load(f)

    with open('./data_cache/2_TPhos_val_samples.pkl', 'rb') as f:
        balanced_validation_list = pickle.load(f)


    batch_size = 128

    train_samples = sampler.SubsetRandomSampler(balanced_train_list)
    dev_samples = sampler.SubsetRandomSampler(balanced_validation_list)

    train_loader = torch.utils.data.DataLoader(cnn_dataSet, batch_size=batch_size, sampler=train_samples, num_workers=0, collate_fn=stack_fn_cnn, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(cnn_dataSet, batch_size=batch_size, sampler=dev_samples, num_workers=0, collate_fn=stack_fn_cnn, drop_last=False)



    """ create model, trainer and tester """

    protein_dim = 1024
    local_dim = 1024
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    lr = 5e-3  #5e-4
    weight_decay = 1e-4
    decay_interval = 5
    lr_decay = 1.0
    iteration = 100
    kernel_size = 7

    encoder2 = Encoder2(protein_dim, hid_dim, n_layers, kernel_size, dropout, device, n_heads)
    decoder2 = Decoder2(local_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer2, SelfAttention2,
                      PositionwiseFeedforward2, dropout, device)
    model = Predictor2(encoder2, decoder2, device)

    model.to(device)

    trainer = Trainer2(model, lr, weight_decay)
    tester = Tester2(model)

    # Output files.
    os.makedirs(('./output/result'), exist_ok=True)
    os.makedirs(('./output/model'), exist_ok=True)
    file_AUCs = './output/result/TASK1_TPhosPPIS' + '.txt'
    file_model = './output/model/' + 'TASK1_TPhosPPIS'
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


        correct_labels_valid, predicted_labels_valid, predicted_scores_valid = tester.test(valid_loader, device)
        



        if torch.cuda.is_available():
            correct_labels_valid = torch.tensor(np.array(correct_labels_valid), dtype=torch.float64, device='cuda')
            predicted_labels_valid = torch.tensor(np.array(predicted_labels_valid), dtype=torch.float64, device='cuda')
            predicted_scores_valid = torch.tensor(np.array(predicted_scores_valid), dtype=torch.float64, device='cuda')


            ACC_dev, AUC_dev, Rec_dev, Pre_dev, F1_dev, MCC_dev, PRC_dev = metrics(correct_labels_valid.cpu().numpy(),
                                                                        predicted_labels_valid.cpu().numpy(),
                                                                        predicted_scores_valid.cpu().numpy())

        end2 = timeit.default_timer()
        time2 = end2 - end1
        AUCs = [epoch, time1, time2, loss_train, ACC_dev, AUC_dev, Rec_dev, Pre_dev, F1_dev, MCC_dev, PRC_dev]
        tester.save_AUCs(AUCs, file_AUCs)

        if MCC_dev > max_MCC_dev:
            last_improve = epoch
            print('last_improve: %s' % last_improve)
            if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
            #     if dist.get_rank() == 0:
            #         tester.save_model(model, file_model)
            # else:
                torch.save(model.state_dict(), file_model)
            max_MCC_dev = MCC_dev
        print('\t'.join(map(str, AUCs)))



if __name__ == "__main__":
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')


    """Load preprocessed data."""

    all_encode_file = './data_cache/TPhos_dataA_encode_data.pkl'
    all_label_file = './data_cache/TPhos_dataA_label.pkl'
    all_list_file = './data_cache/TPhos_dataA_list.pkl'

    window_size = 15
    cnn_dataSet = dataSet_cnn(window_size, all_encode_file, all_label_file, all_list_file)

    SEED = 1
    main(SEED)


