import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef, precision_recall_curve, auc
from TASK2_CNN_model import CNNAttentionModel
from TASK2_Transformer_model import *
from TASK2_data_generator import dataSet
import torch.utils.data.sampler as sampler
import pickle


def init_seeds(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(correct_labels, predicted_labels, predicted_scores):
    ACC = accuracy_score(correct_labels, predicted_labels)
    AUC = roc_auc_score(correct_labels, predicted_scores)
    CM = confusion_matrix(correct_labels, predicted_labels)
    TN, FP, FN, TP = CM.ravel()
    Rec = TP / (TP + FN)
    Pre = TP / (TP + FP)
    F1 = 2 * Pre * Rec / (Pre + Rec)
    MCC = matthews_corrcoef(correct_labels, predicted_labels)
    precision, recall, _ = precision_recall_curve(correct_labels, predicted_scores)
    PRC = auc(recall, precision)
    return ACC, AUC, Rec, Pre, F1, MCC, PRC

def stack_fn_cnn(batch):
    local_features, all_seq_features_A, all_seq_features_B, labels = [], [], [], []
    indices = []

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

    locals_new = torch.zeros((N, locals_len, local_dim), dtype=torch.float32)
    proteins_new_A = torch.zeros((N, proteins_len_A, protein_dim), dtype=torch.float32)
    proteins_new_B = torch.zeros((N, proteins_len_B, protein_dim), dtype=torch.float32)
    labels_new = torch.zeros(N, dtype=torch.long)

    for i, (local, protein_A, protein_B, label) in enumerate(zip(local_features, all_seq_features_A, all_seq_features_B, labels)):
        locals_new[i, :local.shape[0], :] = torch.tensor(local, dtype=torch.float32)
        proteins_new_A[i, :protein_A.shape[0], :] = torch.tensor(protein_A, dtype=torch.float32)
        proteins_new_B[i, :protein_B.shape[0], :] = torch.tensor(protein_B, dtype=torch.float32)
        labels_new[i] = torch.tensor(label, dtype=torch.long)

    return locals_new, proteins_new_A, proteins_new_B, labels_new

def stack_fn_transformer(batch):
    local_features, all_seq_features_A, all_seq_features_B, labels = [], [], [], []
    local_num, protein_num_A, protein_num_B = [], [], []
    indices = []

    for local, seq_A, seq_B, label in batch:
        local_features.append(local)
        all_seq_features_A.append(seq_A)
        all_seq_features_B.append(seq_B)
        labels.append(label)
        local_num.append(local.shape[0])
        protein_num_A.append(seq_A.shape[0])
        protein_num_B.append(seq_B.shape[0])



    N = len(labels)
    local_dim = 1024
    protein_dim = 1024

    locals_len = max(local.shape[0] for local in local_features)
    proteins_len_A = max(protein_A.shape[0] for protein_A in all_seq_features_A)
    proteins_len_B = max(protein_B.shape[0] for protein_B in all_seq_features_B)

    locals_new = torch.zeros((N, locals_len, local_dim), dtype=torch.float32)
    proteins_new_A = torch.zeros((N, proteins_len_A, protein_dim), dtype=torch.float32)
    proteins_new_B = torch.zeros((N, proteins_len_B, protein_dim), dtype=torch.float32)
    labels_new = np.zeros(N, dtype=np.int64)


    for i, (local, protein_A, protein_B, label) in enumerate(zip(local_features, all_seq_features_A, all_seq_features_B, labels)):
        locals_new[i, :local.shape[0], :] = torch.tensor(local, dtype=torch.float32)
        proteins_new_A[i, :protein_A.shape[0], :] = torch.tensor(protein_A, dtype=torch.float32)
        proteins_new_B[i, :protein_B.shape[0], :] = torch.tensor(protein_B, dtype=torch.float32)
        labels_new[i] = torch.tensor(label, dtype=torch.long)

    return locals_new, proteins_new_A, proteins_new_B, labels_new, local_num, protein_num_A, protein_num_B



def predict_with_cnn(cnn_model, test_loader, device):
    cnn_model.eval()
    correct_labels, predicted_labels, predicted_scores = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            local_features, all_seq_features_A, all_seq_features_B, labels = batch
            local_features = local_features.to(device)
            all_seq_features_A = all_seq_features_A.to(device)
            all_seq_features_B = all_seq_features_B.to(device)
            labels = labels.to(device)


            outputs = cnn_model(local_features, all_seq_features_A, all_seq_features_B)
            predicted_scores_batch = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            predicted_labels_batch = torch.argmax(outputs, dim=1).cpu().numpy()
            correct_labels_batch = labels.cpu().numpy()

            correct_labels.extend(correct_labels_batch)
            predicted_labels.extend(predicted_labels_batch)
            predicted_scores.extend(predicted_scores_batch)
    return correct_labels, predicted_labels, predicted_scores







def ensemble_predictions(cnn_scores, transformer_scores, alpha=0):
    ensemble_scores = [alpha * cnn + (1 - alpha) * transformer for cnn, transformer in zip(cnn_scores, transformer_scores)]
    ensemble_predictions = [1 if score > 0.5 else 0 for score in ensemble_scores]
    return ensemble_predictions, ensemble_scores




if __name__ == "__main__":
    init_seeds(seed=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    with open('./data_cache/TASK2_all_test_samples.pkl', 'rb') as f:
        test_list = pickle.load(f)

    batch_size = 128
    all_encode_file_A = './data_cache/TASK2_TPhos_dataB_encode_data_all_ProteinAA.pkl'
    all_encode_file_B = './data_cache/TASK2_TPhos_dataB_encode_data_all_ProteinBB.pkl'
    all_list_file_A = './data_cache/TASK2_TPhos_dataB_protein_list_all_AA.pkl'
    all_list_file_B = './data_cache/TASK2_TPhos_dataB_protein_list_all_BB.pkl'

    window_size = 15
    all_dataSet = dataSet(window_size, all_encode_file_A, all_encode_file_B, all_list_file_A, all_list_file_B)
    test_samples = sampler.SubsetRandomSampler(test_list)
    test_loader_cnn = torch.utils.data.DataLoader(
    all_dataSet,
    batch_size=batch_size,
    sampler=test_samples,
    num_workers=0,
    collate_fn=stack_fn_cnn)
    test_loader_transformer = torch.utils.data.DataLoader(
    all_dataSet,
    batch_size=batch_size,
    sampler=test_samples,
    num_workers=0,
    collate_fn=stack_fn_transformer)


    cnn_model = CNNAttentionModel(1024, 1024, 3, 7, 0.1, device)
    cnn_model.load_state_dict(torch.load('./output/model/TASK2_CNN', map_location=device))
    cnn_model.to(device)

    encoder_A = Encoder(1024, 64, 3, 7, 0.1, device)
    encoder_B = Encoder(1024, 64, 3, 7, 0.1, device)
    decoder = Decoder(1024, 64, 3, 8, 256, DecoderLayer, SelfAttention, PositionwiseFeedforward, 0.1, device)
    transformer_model = Predictor(encoder_A, encoder_B, decoder, device)
    transformer_model.load_state_dict(torch.load('./output/model/TASK2_TransformerPPIS', map_location=device))
    transformer_model.to(device)

    transformer_tester = Tester(transformer_model)

    transformer_correct, transformer_preds, transformer_scores = transformer_tester.test(test_loader_transformer, device)


    cnn_correct, cnn_preds, cnn_scores = predict_with_cnn(cnn_model, test_loader_cnn, device)


    print("CNN Model Metrics:")
    print(metrics(cnn_correct, cnn_preds, cnn_scores))

    print("Transformer Model Metrics:")
    print(metrics(transformer_correct, transformer_preds, transformer_scores))

    ensemble_preds, ensemble_scores = ensemble_predictions(cnn_scores, transformer_scores, alpha=0.5)
    print("Ensemble Metrics:")
    print(metrics(transformer_correct, ensemble_preds, ensemble_scores))




