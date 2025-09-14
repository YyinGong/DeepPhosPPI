import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Radam import RAdam
from lookahead import Lookahead


class SelfAttention2(nn.Module):
    """Simple Self-Attention mechanism."""
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)
        self.query = nn.Linear(hid_dim, hid_dim)
        self.key = nn.Linear(hid_dim, hid_dim)
        self.value = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = F.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        attended = torch.matmul(attention, V)
        return attended, attention


class Encoder2(nn.Module):
    """Protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout, device, n_heads):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2)
            for _ in range(self.n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)
        
        # Add Attention Layer
        self.attention = SelfAttention2(hid_dim, n_heads, dropout, device)

    def forward(self, protein):
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)

        for conv in self.convs:
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim=1)
            conved = conved + conv_input
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        conved = self.attention(conved)[0]
        conved = self.ln(conved)
        return conved


class PositionwiseFeedforward2(nn.Module):
    """Feedforward layer in Transformer."""
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.do(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        x = x.permute(0, 2, 1)
        return x


class DecoderLayer2(nn.Module):
    """Decoder Layer."""
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


class Decoder2(nn.Module):
    """Decoder with feedforward classification layers."""
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
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList([
            decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
            for _ in range(n_layers)
        ])
        self.ft = nn.Linear(local_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 64)
        self.fc_3 = nn.Linear(64, 2)

    def forward(self, trg, index):
        norm = torch.norm(trg, dim=2)
        norm = F.softmax(norm, dim=1)
        sum = torch.zeros((trg.shape[0], self.hid_dim)).to(self.device)
        for i in range(norm.shape[0]):
            v = trg[i, index[i], ]
            v = v * norm[i, index[i]]
            sum[i, ] += v
        label = F.relu(self.fc_1(sum))
        label = self.do(label)
        label = F.relu(self.fc_2(label))
        label = self.fc_3(label)
        return sum, label


class Predictor2(nn.Module):
    """Predictor combining Encoder and Decoder."""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, protein, index):
        enc_src = self.encoder(protein)
        sum, out = self.decoder.forward(enc_src, index)
        return sum, out

    def __call__(self, data, train=True):
        protein, index, correct_interaction = data
        Loss = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 5])).float().to(self.device))
        if train:
            sum, predicted_interaction = self.forward(protein, index)
            loss2 = Loss(predicted_interaction, correct_interaction)
            return loss2
        else:
            sum, predicted_interaction = self.forward(protein, index)
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]
            return correct_labels, predicted_labels, predicted_scores


class Trainer2:
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
        for batch_idx, (protein, index, label) in enumerate(dataloader):
            data_pack = todevice(protein, index, label, device)
            loss = self.model(data_pack)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_total += loss.item()
        return loss_total


class Tester2:
    def __init__(self, model):
        self.model = model

    def test(self, dataloader, device):
        self.model.eval()
        T, Y, S = [], [], []
        with torch.no_grad():
            for batch_idx, (protein, index, label) in enumerate(dataloader):
                data_pack = todevice(protein, index, label, device)
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

def todevice(proteins, index, labels, device):
    proteins_new = torch.Tensor(proteins).float().to(device)
    labels_new = torch.from_numpy(labels).long().to(device)
    index = index
    return (proteins_new, index, labels_new)
