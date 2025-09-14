import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Radam import *
from lookahead import Lookahead


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(logits, targets)
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)

        return x, attention


class Encoder(nn.Module):

    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout, device):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList(
            [nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2) for _ in
             range(self.n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)

        for conv in self.convs:
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)
        return conved


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.do(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        x = x.permute(0, 2, 1)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg_1 = trg
        trg, _ = self.sa(trg, trg, trg, trg_mask)
        trg = self.ln(trg_1 + self.do(trg))

        trg_2 = trg
        trg, attention = self.ea(trg, src, src, src_mask)
        trg = self.ln(trg_2 + self.do(trg))

        trg_3 = trg
        trg = self.ln(trg_3 + self.do(self.pf(trg)))
        return trg, attention


class Decoder(nn.Module):
    def __init__(self, local_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = local_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.ft = nn.Linear(local_dim, hid_dim)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 64)
        self.fc_3 = nn.Linear(64, 2)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.ft(trg)

        for layer in self.layers:
            trg, _ = layer(trg, src, trg_mask, src_mask)

        norm = torch.norm(trg, dim=2)
        norm = F.softmax(norm, dim=1)

        sum = torch.zeros((trg.shape[0], self.hid_dim)).to(self.device)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = trg[i, j,] * norm[i, j]
                sum[i,] += v

        label = F.relu(self.fc_1(sum))
        label = self.do(label)
        label = F.relu(self.fc_2(label))
        label = self.fc_3(label)
        return sum, _, label

Epsion = 0.2
class Predictor(nn.Module):
    def __init__(self, encoder_A, encoder_B, decoder, device):
        super().__init__()
        self.encoder_A = encoder_A
        self.encoder_B = encoder_B
        self.decoder = decoder
        self.device = device

    def make_masks(self, local_num, protein_num, local_max_len, protein_max_len):
        N = len(local_num)
        local_mask = torch.zeros((N, local_max_len))
        protein_mask_A = torch.zeros((N, protein_max_len))
        protein_mask_B = torch.zeros((N, protein_max_len))
        for i in range(N):
            local_mask[i, :local_num[i]] = 1
            protein_mask_A[i, :protein_num[i]] = 1
            protein_mask_B[i, :protein_num[i]] = 1
        local_mask = local_mask.unsqueeze(1).unsqueeze(3).to(self.device)
        protein_mask_A = protein_mask_A.unsqueeze(1).unsqueeze(2).to(self.device)
        protein_mask_B = protein_mask_B.unsqueeze(1).unsqueeze(2).to(self.device)
        return local_mask, protein_mask_A, protein_mask_B

    def forward(self, local, protein_A, protein_B, local_num, protein_num_A, protein_num_B):
        local_max_len = local.shape[1]
        protein_A_max_len = protein_A.shape[1]
        protein_B_max_len = protein_B.shape[1]

        local_mask, protein_A_mask, protein_B_mask = self.make_masks(local_num, protein_num_A, local_max_len, protein_A_max_len)

        enc_src_A = self.encoder_A(protein_A)
        enc_src_B = self.encoder_B(protein_B)

        sum, attention, out = self.decoder(local, enc_src_A, local_mask, protein_A_mask)

        return sum, attention, out

    def __call__(self, data, train=True):
        local, protein_A, protein_B, correct_interaction, local_num, protein_num_A, protein_num_B = data


        weights = []
        batch_labels = correct_interaction.data.cpu().tolist()
        for label in range(2):
            weights.append(1 - (batch_labels.count(label) / len(batch_labels)) + Epsion)
        class_weights = torch.FloatTensor(weights).to(self.device)

        loss_CrossEn = nn.CrossEntropyLoss(weight=class_weights)


        if train:
            sum, attention, predicted_interaction = self.forward(local, protein_A, protein_B, local_num, protein_num_A, protein_num_B)
            loss2 = loss_CrossEn(predicted_interaction, correct_interaction)
            return loss2

        else:
            sum, attention, predicted_interaction = self.forward(local, protein_A, protein_B, local_num, protein_num_A, protein_num_B)
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]
            return correct_labels, predicted_labels, predicted_scores


def todevice(locals, proteins_A, proteins_B, labels, local_num, protein_num_A, protein_num_B, device):
    locals_new = torch.Tensor(locals).to(device)
    proteins_A_new = torch.Tensor(proteins_A).to(device)
    proteins_B_new = torch.Tensor(proteins_B).to(device)
    labels_new = torch.from_numpy(labels).to(device)

    return locals_new, proteins_A_new, proteins_B_new, labels_new, local_num, protein_num_A, protein_num_B


class Trainer(object):
    def __init__(self, model, lr, weight_decay):
        self.model = model
        weight_p, bias_p = [], []

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        self.optimizer_inner = RAdam(
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)

    def train(self, dataloader, device):
        self.model.train()
        loss_total = 0
        self.optimizer.zero_grad()
        for batch_idx, (local, protein_A, protein_B, label, local_num, protein_num_A, protein_num_B) in enumerate(dataloader):
            print(batch_idx)
            data_pack = todevice(local, protein_A, protein_B, label, local_num, protein_num_A, protein_num_B, device)
            loss = self.model(data_pack)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataloader, device):
        self.model.eval()
        T, Y, S = [], [], []
        with torch.no_grad():
            for batch_idx, (local, protein_A, protein_B, label, local_num, protein_num_A, protein_num_B) in enumerate(dataloader):
                data_pack = todevice(local, protein_A, protein_B, label, local_num, protein_num_A, protein_num_B, device)
                correct_labels, predicted_labels, predicted_scores = self.model(data_pack, train=False)
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
        return T, Y, S

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

