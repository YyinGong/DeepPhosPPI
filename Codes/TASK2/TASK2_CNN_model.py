import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAttentionModel(nn.Module):
    def __init__(self, local_dim, hid_dim, n_layers, kernel_size, dropout, device):
        super(CNNAttentionModel, self).__init__()
        self.device = device

        # 蛋白A局部特征的卷积层
        # 使用Conv1d对局部特征进行卷积，提取局部模式信息
        self.local_conv = nn.Conv1d(in_channels=local_dim, out_channels=hid_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.local_ln = nn.LayerNorm(hid_dim)  # 对卷积后的特征进行层归一化
        self.local_do = nn.Dropout(dropout)  # 使用Dropout防止过拟合

        # 蛋白A全局特征的卷积层
        # 对蛋白A的全局特征进行卷积，提取重要的全局模式信息
        self.global_conv_A = nn.Conv1d(in_channels=local_dim, out_channels=hid_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.global_ln_A = nn.LayerNorm(hid_dim)  # 对卷积后的特征进行层归一化
        self.global_do_A = nn.Dropout(dropout)  # 使用Dropout防止过拟合

        # 蛋白B全局特征的卷积层
        # 对蛋白B的全局特征进行卷积，提取重要模式信息
        self.global_conv_B = nn.Conv1d(in_channels=local_dim, out_channels=hid_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.global_ln_B = nn.LayerNorm(hid_dim)  # 对卷积后的特征进行层归一化
        self.global_do_B = nn.Dropout(dropout)  # 使用Dropout防止过拟合

        # 注意力机制，用于加权局部特征和全局特征
        # 输入为拼接后的局部特征和全局特征，输出一个标量注意力权重
        self.attention_fc = nn.Linear(hid_dim * 3, 1)  # 线性层用于计算加权系数

        # 分类网络部分
        # 使用三层全连接层，最终将特征分类为两个类别（增强/抑制）
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 64)
        self.fc_3 = nn.Linear(64, 2)

        # 对隐藏层的特征进行归一化，使用8个组来进行归一化
        self.gn = nn.GroupNorm(8, 256)

    def forward(self, local_features_A, all_seq_features_A, all_seq_features_B, batch_idx=None):
        """
        :param local_features_A: 蛋白A的局部特征 [batch size, local len, feature_dim]
        :param all_seq_features_B: 蛋白B的全局特征 [batch size, seq len, feature_dim]
        :param batch_idx: 当前批次的索引 (可选)
        """
        # if batch_idx is not None:
            # print(batch_idx)
        # print(f"local_features_A shape: {local_features_A.shape}")  # 应该是 [batch_size, local_len, local_dim]
        # print(f"all_seq_features_A shape: {all_seq_features_A.shape}")  # 应该是 [batch_size, seq_len, local_dim]
        # print(f"all_seq_features_B shape: {all_seq_features_B.shape}")  # 应该是 [
        print(batch_idx)
        # 处理蛋白A的局部特征
        # 调整维度顺序，以适应Conv1d的输入格式 (batch_size, feature_dim, seq_len)
        local_features_A = local_features_A.permute(0, 2, 1)  # [batch size, feature_dim, local len]
        local_features_A = self.local_conv(local_features_A)  # 对局部特征进行卷积 [batch size, hid_dim, local len]
        local_features_A = local_features_A.permute(0, 2, 1)  # 调整回原来的维度顺序 [batch size, local len, hid_dim]
        local_features_A = self.local_ln(local_features_A)  # 进行层归一化，帮助训练稳定性
        local_features_A = self.local_do(local_features_A)  # 使用Dropout防止过拟合

        # 聚合局部特征：沿着局部序列长度的维度进行求和，得到全局的局部特征表示
        sum_A_local = torch.sum(local_features_A, dim=1)  # [batch size, hid_dim]

        # 处理蛋白A的全局特征
        all_seq_features_A = all_seq_features_A.permute(0, 2, 1)  # [batch size, feature_dim, seq_len]
        all_seq_features_A = self.global_conv_A(all_seq_features_A)  # 对全局特征进行卷积 [batch size, hid_dim, seq_len]
        all_seq_features_A = all_seq_features_A.permute(0, 2, 1)  # 调整回原来的维度顺序 [batch size, seq_len, hid_dim]
        all_seq_features_A = self.global_ln_A(all_seq_features_A)  # 进行层归一化，帮助训练稳定性
        all_seq_features_A = self.global_do_A(all_seq_features_A)  # 使用Dropout防止过拟合

        # 聚合蛋白A的全局特征：沿着序列长度的维度进行求和，得到全局特征表示
        sum_A_global = torch.sum(all_seq_features_A, dim=1)  # [batch size, hid_dim]

        # 处理蛋白B的全局特征
        # 调整维度顺序，以适应Conv1d的输入格式 (batch_size, feature_dim, seq_len)
        all_seq_features_B = all_seq_features_B.permute(0, 2, 1)  # [batch size, feature_dim, seq len]
        all_seq_features_B = self.global_conv_B(all_seq_features_B)  # 对全局特征进行卷积 [batch size, hid_dim, seq len]
        all_seq_features_B = all_seq_features_B.permute(0, 2, 1)  # 调整回原来的维度顺序 [batch size, seq len, hid_dim]
        all_seq_features_B = self.global_ln_B(all_seq_features_B)  # 进行层归一化，帮助训练稳定性
        all_seq_features_B = self.global_do_B(all_seq_features_B)  # 使用Dropout防止过拟合

        # 聚合蛋白B的全局特征：沿着序列长度的维度进行求和，得到全局特征表示
        sum_B = torch.sum(all_seq_features_B, dim=1)  # [batch size, hid_dim]

        # 使用注意力机制来强调局部特征
        # 将蛋白A的局部特征、蛋白A的全局特征和蛋白B的全局特征进行拼接，形成 [batch size, 3 * hid_dim]
        combined_features = torch.cat((sum_A_local, sum_A_global, sum_B), dim=1)  # 拼接局部和全局特征
        attention_weights = torch.sigmoid(self.attention_fc(combined_features))  # 计算注意力权重，值域在 [0, 1]

        # 加权局部特征和全局特征
        # 根据注意力权重对蛋白A和蛋白B的特征进行加权，确保局部特征的重要性不会被稀释
        weighted_sum_A_local = sum_A_local * attention_weights  # 对局部特征加权
        weighted_sum_A_global = sum_A_global * attention_weights  # 对蛋白A的全局特征加权
        weighted_sum_B = sum_B * (1 - attention_weights)  # 对蛋白B的全局特征加权

        # 将加权的局部特征和全局特征结合
        combined = weighted_sum_A_local + weighted_sum_A_global + weighted_sum_B  # 最终得到结合后的特征 [batch size, hid_dim]

        # 通过分类网络进行最终预测
        # 使用三层全连接层进行分类，得到二分类的输出（增强/抑制）
        # label = F.relu(self.fc_1(combined))  # 第一层全连接并使用ReLU激活函数
        label = F.gelu(self.fc_1(combined))  # 改为 GELU
        label = self.gn(label)  # 使用GroupNorm进行归一化
        # label = F.relu(self.fc_2(label))  # 第二层全连接并使用ReLU激活函数
        label = F.gelu(self.fc_2(label))  # 改为 GELU
        label = self.fc_3(label)  # 最后一层全连接，得到二分类输出 [batch size, 2]

        return label
