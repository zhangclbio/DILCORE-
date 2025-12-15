import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn import MultiheadAttention
class ViewFusionModule(nn.Module):
    def __init__(self, fusion_dim, num_heads=4, dropout=0.1):
        super(ViewFusionModule, self).__init__()
        
        # 交叉注意力机制
        self.cross_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout)

        # 归一化层
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )

        # Gating 机制
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算 Self-Attention
        self_attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(self_attn_output + x)

        # 计算 Cross-Attention
        cross_attn_output, _ = self.cross_attn(x, x, x)
        x = self.norm2(cross_attn_output + x)

        # FFN + 残差连接
        ffn_out = self.ffn(x)
        gate = self.gate(ffn_out)
        x = x + gate * ffn_out  # 通过 gating 控制 FFN 对特征的贡献

        return x


# 关键修改
def InfoNCE_Loss(view_num=3, Mv_common_peculiar=[], model=None, queue=None):
    """
    采用 Dynamic Temperature 机制的 InfoNCE 损失：
    - 使用 self.temperature 作为可训练参数
    - 通过梯度更新动态调整 temperature
    """
    loss_total = 0.0
    count = 0
    
    # **获取动态温度**
    temperature = model.temperature.clamp(min=0.01, max=1.0)  # 限制 temperature 避免极端值
    
    for view in range(view_num):
        common_i = F.normalize(Mv_common_peculiar[view][0], dim=1)  # 当前视图的 common
        for view_other in range(view_num):
            if view == view_other:
                continue  # 跳过相同视图
            
            common_j = F.normalize(Mv_common_peculiar[view_other][0], dim=1)  # 其他视图的 common
            peculiar_neg = F.normalize(Mv_common_peculiar[view_other][1], dim=1)  # 负样本

            # **计算正样本相似度**
            pos_sim = torch.exp(F.cosine_similarity(common_i, common_j, dim=1) / temperature)

            # **计算负样本相似度**
            if queue is not None:
                neg_sim = torch.exp(F.cosine_similarity(common_i, queue, dim=1) / temperature)
            else:
                neg_sim = torch.exp(F.cosine_similarity(common_i, peculiar_neg, dim=1) / temperature)

            # **引入 Hard Negative Mining**
            neg_sim_weighted = torch.exp(-neg_sim * 0.5)  # 给予较难负样本更高权重

            # 计算损失
            loss = -torch.log(pos_sim / (pos_sim + torch.sum(neg_sim_weighted, dim=0)))
            loss_total += torch.mean(loss)
            count += 1

    return loss_total / count


def target_distribution(q):
    """
    计算目标分布 p，使得 q 更加尖锐，提高聚类质量。
    """
    weight = q ** 2 / torch.sum(q, dim=0, keepdim=True)  # 计算权重
    p = (weight.t() / torch.sum(weight, dim=1)).t()
    return p
class loss_funcation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, view_num, data_batch, out_list, latent_dist, lmda_list, batch_size, model):
        batch_size = data_batch[0].shape[1]
        Mv_common_peculiar = latent_dist['Mv_common_peculiar']
        mu, logvar = latent_dist['mu_logvar']

        rec_loss, KLD, I_loss, loss = 0.0, 0.0, 0.0, 0.0
        loss_dict = {}

        for i in range(view_num):
            rec_loss += torch.sum(torch.pow(data_batch[i] - out_list[i], 2))
            KLD += 0.5 * torch.sum(torch.exp(logvar[i]) + torch.pow(mu[i], 2) - 1. - logvar[i])

        rec_loss /= view_num
        KLD /= view_num

        # **使用动态温度**
        I_loss = InfoNCE_Loss(view_num, Mv_common_peculiar, model=model)

        loss = (lmda_list['rec_lmda'] * rec_loss + lmda_list['KLD_lmda'] * KLD + lmda_list['I_loss_lmda'] * I_loss)
        loss_dict['rec_loss'] = rec_loss
        loss_dict['KLD'] = KLD
        loss_dict['I_loss'] = I_loss
        loss_dict['loss'] = loss
        return loss, loss_dict



class DILCR(nn.Module):
    def __init__(self, in_dim=[], encoder_dim=[1024], feature_dim=512, peculiar_dim=128, common_dim=128,
                 mu_logvar_dim=10, cluster_var_dim=384, up_and_down_dim=512, cluster_num=5, view_num=3,
                 temperature=.67, device = 'cpu'):
        super(DILCR, self).__init__()
        self.device = device

        self.view_num = view_num
        self.mu_logvar_dim = mu_logvar_dim
        self.feature_dim = feature_dim
        self.common_dim = common_dim
        self.peculiar_dim = peculiar_dim
        self.in_dim = in_dim
        self.temperature = 0.07
        self.fusion_dim = self.common_dim * 3
        self.cluster_num = cluster_num
        self.alpha = 1.0
        self.cluster_var_dim = cluster_var_dim
        self.up_and_down_dim = up_and_down_dim
        self.use_up_and_down = up_and_down_dim
        # **动态温度**
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        Mv_encoder_MLP = []
        Mv_feature_peculiar = []
        Mv_peculiar_to_mu_logvar = []
        Mv_feature_to_common = []
        # decoder
        Mv_decoder_MLP = []
        for i in range(self.view_num):
            # 输入到特征
            encoder_MLP = []
            # 输入维度不一致
            encoder_MLP += [
                nn.Linear(in_dim[i], encoder_dim[0]),
                nn.GELU()
            ]
            # 最后一层MLP到特征
            encoder_MLP += [
                nn.Linear(encoder_dim[-1], feature_dim),
                nn.GELU()
            ]
            Mv_encoder_MLP.append(nn.Sequential(*encoder_MLP))
            # 特征到mu,logvar
            Mv_feature_peculiar.append(nn.Sequential(
                nn.Linear(self.feature_dim, self.peculiar_dim),
                nn.GELU()
            ))
            Mv_peculiar_to_mu_logvar.append(nn.Sequential(
                nn.Linear(self.peculiar_dim, self.mu_logvar_dim),
                nn.GELU()
            ))
            # 特征到common
            Mv_feature_to_common.append(nn.Sequential(
                nn.Linear(self.feature_dim, self.common_dim),
                nn.GELU()
            ))
        # 连接后的common 融合注意力机制
        trans_enc = nn.TransformerEncoderLayer(d_model=self.fusion_dim, nhead=1, dim_feedforward=1024,dropout=0.0)
        if self.use_up_and_down != 0:
            fusion_to_cluster = nn.Sequential(
                nn.Linear(self.fusion_dim, self.up_and_down_dim),
                # lusc
                nn.Linear(self.up_and_down_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.up_and_down_dim),
                nn.Linear(self.up_and_down_dim, self.cluster_var_dim)
            )
        else:
            fusion_to_cluster = nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim),
                nn.Linear(self.fusion_dim, self.cluster_var_dim),
            )
        for i in range(view_num):
            decoder_MLP = []
            self.peculiar_and_class_dim = mu_logvar_dim + self.cluster_var_dim
            decoder_MLP += [
                nn.Linear(self.peculiar_and_class_dim, encoder_dim[-1]),
                nn.GELU()
            ]
            decoder_MLP += [
                nn.Linear(encoder_dim[0], in_dim[i]),
                nn.Sigmoid()
            ]
            Mv_decoder_MLP.append(nn.Sequential(*decoder_MLP))

        self.Mv_in_to_feature = nn.ModuleList(Mv_encoder_MLP)
        self.Mv_feature_peculiar = nn.ModuleList(Mv_feature_peculiar)
        self.Mv_peculiar_to_mu_logvar = nn.ModuleList(Mv_peculiar_to_mu_logvar)
        self.Mv_feature_to_common = nn.ModuleList(Mv_feature_to_common)
        # self.Mv_common_to_fusion = nn.TransformerEncoder(trans_enc, num_layers=1)
        self.Mv_common_to_fusion = ViewFusionModule(fusion_dim=self.fusion_dim, num_heads=4)

        self.fusion_to_cluster = fusion_to_cluster
        self.Mv_decoder_MLP = nn.ModuleList(Mv_decoder_MLP)
        # 标签监督层
        # self.cluster_MLP = nn.Linear(self.cluster_num,self.cluster_num)
        self.cluster_layer = nn.Parameter(torch.zeros([self.cluster_num, self.cluster_var_dim], dtype = torch.float32))

    def encoder(self, X):
        '''
        :param X: 3 * b * d 三个视图 b是batch_size d是特征维度
        :return: mu,logvar,common
        '''
        mu = []
        logvar = []
        common = []
        Mv_common_peculiar = []
        for net_index in range(self.view_num):
            view_index = net_index
            feature = self.Mv_in_to_feature[net_index](X[view_index])
            peculiar = self.Mv_feature_peculiar[net_index](feature)
            mu.append(self.Mv_peculiar_to_mu_logvar[net_index](peculiar))
            logvar.append(self.Mv_peculiar_to_mu_logvar[net_index](peculiar))
            temp = self.Mv_feature_to_common[net_index](feature)  # 单个视图的common
            common.append(temp)
            Mv_common_peculiar.append([peculiar, temp])
        # # print(feature)
        Mv_common = torch.concat(common, dim=1)
        Mv_common = torch.unsqueeze(Mv_common, dim=1)
        fusion = self.Mv_common_to_fusion(Mv_common)
        fusion = fusion.reshape([Mv_common.shape[0], -1])
        cluster_var = self.fusion_to_cluster(fusion)

        return Mv_common_peculiar, fusion, cluster_var, mu, logvar

    def reparameterization(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(std.shape).to(self.device)
            return mu + std * eps
        else:
            # Reconstruction mode
            return mu

    def decoder(self, peculiar_and_common):
        out_list = []
        for i in range(self.view_num):
            temp = self.Mv_decoder_MLP[i](peculiar_and_common[i])
            out_list.append(temp)
        return out_list

    def forward(self, X):
        latent_dist = dict()
        Mv_common_peculiar, fusion, cluster_var, mu, logvar = self.encoder(X)

        # 计算 Z
        peculiar_and_common = []
        z = []
        for i in range(self.view_num):
            bn = nn.BatchNorm1d(mu[i].shape[1]).to(self.device)
            # mu[i] = bn(mu[i])
            # logvar[i] = bn(logvar[i])
            z.append(self.reparameterization(mu[i], logvar[i]))
            peculiar_and_common.append(torch.concat([z[i], cluster_var], dim=1))

        out_list = self.decoder(peculiar_and_common)

        # **优化的 Student’s t-distribution 计算 q**
        squared_distance = torch.sum(torch.pow(cluster_var.unsqueeze(1) - self.cluster_layer, 2), dim=2)
        q = 1.0 / (1.0 + squared_distance / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()  # 归一化

        # **优化的 p 计算，避免剧烈变化**
        p = target_distribution(q)

        # 存储
        latent_dist['mu_logvar'] = [mu, logvar]
        latent_dist['fusion'] = fusion
        latent_dist['Mv_common_peculiar'] = Mv_common_peculiar
        latent_dist['z'] = z
        latent_dist['cluster_var'] = cluster_var
        latent_dist['q'] = q
        latent_dist['p'] = p  # **优化后的 p**

        return out_list, latent_dist
