import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torchvision import transforms, models
from torch.autograd import Variable
import numpy as np


__all__ = ['DSN', 'LSTM', 'DSNet', 'DSNetAF', 'CNN_LSTM']

class LSTM(nn.Module):
    def __init__(self, in_dim = 1024, hid_dim = 256, num_layers=1):         # in_dim = 입력에 대한 expected feature의 수 (입력 사이즈) / hid_dim = hidden state에서의 feature의 수 (은닉층의 사이즈)
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        #self.lstm = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim*2, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        p = F.sigmoid(self.fc(h))
        return p



class DSN(nn.Module):
    """Deep Summarization Network"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be eitorcher 'lstm' or 'gru'"
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim*2, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        p = F.sigmoid(self.fc(h))
        return p





class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.sqrt_d_k = math.sqrt(d_k)

    def forward(self, Q, K, V):
        attn = torch.bmm(Q, K.transpose(2, 1))
        attn = attn / self.sqrt_d_k

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        y = torch.bmm(attn, V)

        return y, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_head=8, num_feature=1024):
        super().__init__()
        self.num_head = num_head

        self.Q = nn.Linear(num_feature, num_feature, bias=False)
        self.K = nn.Linear(num_feature, num_feature, bias=False)
        self.V = nn.Linear(num_feature, num_feature, bias=False)

        self.d_k = num_feature // num_head
        self.attention = ScaledDotProductAttention(self.d_k)

        self.fc = nn.Sequential(
            nn.Linear(num_feature, num_feature, bias=False),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        _, seq_len, num_feature = x.shape  # [1, seq_len, 1024]
        K = self.K(x)  # [1, seq_len, 1024]
        Q = self.Q(x)  # [1, seq_len, 1024]
        V = self.V(x)  # [1, seq_len, 1024]

        K = K.view(1, seq_len, self.num_head, self.d_k).permute(
            2, 0, 1, 3).contiguous().view(self.num_head, seq_len, self.d_k)
        Q = Q.view(1, seq_len, self.num_head, self.d_k).permute(
            2, 0, 1, 3).contiguous().view(self.num_head, seq_len, self.d_k)
        V = V.view(1, seq_len, self.num_head, self.d_k).permute(
            2, 0, 1, 3).contiguous().view(self.num_head, seq_len, self.d_k)

        y, attn = self.attention(Q, K, V)  # [num_head, seq_len, d_k]
        y = y.view(1, self.num_head, seq_len, self.d_k).permute(
            0, 2, 1, 3).contiguous().view(1, seq_len, num_feature)

        y = self.fc(y)

        return y, attn

class AttentionExtractor(MultiHeadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *inputs):
        out, _ = super().forward(*inputs)
        return out

class GCNExtractor(nn.Module):
    def __init__(self, num_feature):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.gcn = GCNConv(num_feature, num_feature)

    def forward(self, x):
        x = x.squeeze(0)
        edge_indices, edge_weights = self.create_graph(x, keep_ratio=0.3)
        out = self.gcn(x, edge_indices, edge_weights)
        out = out.unsqueeze(0)
        return out

    @staticmethod
    def create_graph(x, keep_ratio=0.3):
        seq_len, _ = x.shape
        keep_top_k = int(keep_ratio * seq_len * seq_len)

        edge_weights = torch.matmul(x, x.t())
        edge_weights = edge_weights - torch.eye(seq_len, seq_len).to(x.device)
        edge_weights = edge_weights.view(-1)
        edge_weights, edge_indices = torch.topk(
            edge_weights, keep_top_k, sorted=False)

        edge_indices = edge_indices.unsqueeze(0)
        edge_indices = torch.cat(
            [edge_indices / seq_len, edge_indices % seq_len])

        return edge_indices, edge_weights

class LSTMExtractor(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *inputs):
        out, _ = super().forward(*inputs)
        return out

def build_base_model(base_type: str, num_feature: int, num_head: int) -> nn.Module:
    if base_type == 'linear':
        base_model = nn.Linear(num_feature, num_feature)
    elif base_type == 'lstm':
        base_model = LSTMExtractor(num_feature, num_feature)
    elif base_type == 'bilstm':
        base_model = LSTMExtractor(num_feature, num_feature // 2,
                                   bidirectional=True)
    elif base_type == 'gcn':
        base_model = GCNExtractor(num_feature)
    elif base_type == 'attention':
        base_model = AttentionExtractor(num_head, num_feature)
    else:
        raise ValueError(f'Invalid base model {base_type}')

    return base_model



class DSNet (nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, anchor_scales,
                 num_head):
        super().__init__()
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        self.base_model = build_base_model(base_model, num_feature, num_head)

        self.roi_poolings = [nn.AvgPool1d(scale, stride=1, padding=scale // 2)
                             for scale in anchor_scales]

        self.layer_norm = nn.LayerNorm(num_feature)
        self.fc1 = nn.Sequential(
            nn.Linear(num_feature, num_hidden),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden)
        )
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)

    def forward(self, x):
        _, seq_len, _ = x.shape
        out = self.base_model(x)
        out = out + x
        out = self.layer_norm(out)

        out = out.transpose(2, 1)
        pool_results = [roi_pooling(out) for roi_pooling in self.roi_poolings]
        out = torch.cat(pool_results, dim=0).permute(2, 0, 1)[:-1]

        out = self.fc1(out)

        pred_cls = self.fc_cls(out).sigmoid().view(seq_len, self.num_scales)
        pred_loc = self.fc_loc(out).view(seq_len, self.num_scales, 2)

        return out



class DSNetAF(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, num_head):
        super().__init__()
        self.base_model = build_base_model(base_model, num_feature, num_head)
        self.layer_norm = nn.LayerNorm(num_feature)

        self.fc1 = nn.Sequential(
            nn.Linear(num_feature, num_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden)
        )
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)
        self.fc_ctr = nn.Linear(num_hidden, 1)

    def forward(self, x):
        _, seq_len, _ = x.shape
        out = self.base_model(x)

        out = out + x
        out = self.layer_norm(out)

        out = self.fc1(out)

        pred_cls = self.fc_cls(out).sigmoid().view(seq_len)
        pred_loc = self.fc_loc(out).exp().view(seq_len, 2)

        pred_ctr = self.fc_ctr(out).sigmoid().view(seq_len)

        return out


class CNN_LSTM(nn.Module):
    def __init__(self, in_dim = 1024, hid_dim = 256, num_layers=1):
        super(CNN_LSTM, self).__init__()

        '''
        self.lstm_cell_1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
        self.lstm_cell_2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
        self.lstm_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell([self.lstm_cell_1, self.lstm_cell_2], state_is_tuple=True)'''

        self.lstm_cell_1 = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.lstm_cell_2 = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)


    def forward(self, x):
        l1 = self.lstm_cell_1(x)
        l2 = self.lstm_cell_2(l1)
        #p = self.lstm_cells(l2)

        return l2
