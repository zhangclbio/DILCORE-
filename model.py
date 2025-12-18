import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn import MultiheadAttention

class ViewFusionModule(nn.Module):
    def __init__(self, fusion_dim, num_heads=4, dropout=0.1):
        super(ViewFusionModule, self).__init__()
        
        # Cross attention mechanism
        self.cross_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout)

        # Normalization layers
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)

        # Feed Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Compute Self-Attention
        self_attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(self_attn_output + x)

        # Compute Cross-Attention
        cross_attn_output, _ = self.cross_attn(x, x, x)
        x = self.norm2(cross_attn_output + x)

        # FFN + residual connection
        ffn_out = self.ffn(x)
        gate = self.gate(ffn_out)
        x = x + gate * ffn_out  # Control FFN's contribution to features via gating

        return x


# Key modification
def InfoNCE_Loss(view_num=3, Mv_common_peculiar=[], model=None, queue=None):
    """
    InfoNCE loss with Dynamic Temperature mechanism:
    - Use self.temperature as a trainable parameter
    - Dynamically adjust temperature through gradient updates
    """
    loss_total = 0.0
    count = 0
    
    # **Get dynamic temperature**
    temperature = model.temperature.clamp(min=0.01, max=1.0)  # Restrict temperature to avoid extreme values
    
    for view in range(view_num):
        common_i = F.normalize(Mv_common_peculiar[view][0], dim=1)  # Common features of current view
        for view_other in range(view_num):
            if view == view_other:
                continue  # Skip the same view
            
            common_j = F.normalize(Mv_common_peculiar[view_other][0], dim=1)  # Common features of other views
            peculiar_neg = F.normalize(Mv_common_peculiar[view_other][1], dim=1)  # Negative samples

            # **Calculate positive sample similarity**
            pos_sim = torch.exp(F.cosine_similarity(common_i, common_j, dim=1) / temperature)

            # **Calculate negative sample similarity**
            if queue is not None:
                neg_sim = torch.exp(F.cosine_similarity(common_i, queue, dim=1) / temperature)
            else:
                neg_sim = torch.exp(F.cosine_similarity(common_i, peculiar_neg, dim=1) / temperature)

            # **Introduce Hard Negative Mining**
            neg_sim_weighted = torch.exp(-neg_sim * 0.5)  # Assign higher weights to harder negative samples

            # Calculate loss
            loss = -torch.log(pos_sim / (pos_sim + torch.sum(neg_sim_weighted, dim=0)))
            loss_total += torch.mean(loss)
            count += 1

    return loss_total / count


def target_distribution(q):
    """
    Calculate target distribution p to sharpen q and improve clustering quality.
    """
    weight = q ** 2 / torch.sum(q, dim=0, keepdim=True)  # Calculate weights
    p = (weight.t() / torch.sum(weight, dim=1)).t()
    return p

class LossFunction(nn.Module):  # Renamed for PEP8 compliance
    def __init__(self):
        super().__init__()

    def forward(self, view_num, data_batch, out_list, latent_dist, lmda_list, batch_size, model):
        batch_size = data_batch[0].shape[1]
        Mv_common_peculiar = latent_dist['Mv_common_peculiar']
        mu, logvar = latent_dist['mu_logvar']

        rec_loss, KLD, I_loss, total_loss = 0.0, 0.0, 0.0, 0.0  # Renamed 'loss' to 'total_loss' for clarity
        loss_dict = {}

        for i in range(view_num):
            rec_loss += torch.sum(torch.pow(data_batch[i] - out_list[i], 2))
            KLD += 0.5 * torch.sum(torch.exp(logvar[i]) + torch.pow(mu[i], 2) - 1. - logvar[i])

        rec_loss /= view_num
        KLD /= view_num

        # **Use dynamic temperature**
        I_loss = InfoNCE_Loss(view_num, Mv_common_peculiar, model=model)

        total_loss = (lmda_list['rec_lmda'] * rec_loss + lmda_list['KLD_lmda'] * KLD + lmda_list['I_loss_lmda'] * I_loss)
        loss_dict['rec_loss'] = rec_loss
        loss_dict['KLD'] = KLD
        loss_dict['I_loss'] = I_loss
        loss_dict['loss'] = total_loss
        return total_loss, loss_dict

class DILCR(nn.Module):
    def __init__(self, in_dim=[], encoder_dim=[1024], feature_dim=512, peculiar_dim=128, common_dim=128,
                 mu_logvar_dim=10, cluster_var_dim=384, up_and_down_dim=512, cluster_num=5, view_num=3,
                 temperature=.67, device='cpu'):
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
        
        # **Dynamic temperature**
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        
        # Encoder layers
        mv_encoder_mlp = []  # Renamed for lowercase convention (PEP8)
        mv_feature_peculiar = []
        mv_peculiar_to_mu_logvar = []
        mv_feature_to_common = []
        
        # Decoder layers
        mv_decoder_mlp = []

        for i in range(self.view_num):
            # Input to feature encoder
            encoder_mlp = []
            # Handle inconsistent input dimensions
            encoder_mlp += [
                nn.Linear(in_dim[i], encoder_dim[0]),
                nn.GELU()
            ]
            # Final MLP layer to feature dimension
            encoder_mlp += [
                nn.Linear(encoder_dim[-1], feature_dim),
                nn.GELU()
            ]
            mv_encoder_mlp.append(nn.Sequential(*encoder_mlp))
            
            # Feature to peculiar space
            mv_feature_peculiar.append(nn.Sequential(
                nn.Linear(self.feature_dim, self.peculiar_dim),
                nn.GELU()
            ))
            
            # Peculiar space to mu/logvar
            mv_peculiar_to_mu_logvar.append(nn.Sequential(
                nn.Linear(self.peculiar_dim, self.mu_logvar_dim),
                nn.GELU()
            ))
            
            # Feature to common space
            mv_feature_to_common.append(nn.Sequential(
                nn.Linear(self.feature_dim, self.common_dim),
                nn.GELU()
            ))

        # Transformer encoder for fused common features (attention mechanism)
        trans_enc = nn.TransformerEncoderLayer(d_model=self.fusion_dim, nhead=1, dim_feedforward=1024, dropout=0.0)
        
        # Fusion to cluster projection
        if self.use_up_and_down != 0:
            fusion_to_cluster = nn.Sequential(
                nn.Linear(self.fusion_dim, self.up_and_down_dim),
                # LUSC (Lung Squamous Cell Carcinoma) specific layer
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

        # Decoder initialization
        for i in range(view_num):
            decoder_mlp = []
            self.peculiar_and_class_dim = mu_logvar_dim + self.cluster_var_dim
            decoder_mlp += [
                nn.Linear(self.peculiar_and_class_dim, encoder_dim[-1]),
                nn.GELU()
            ]
            decoder_mlp += [
                nn.Linear(encoder_dim[0], in_dim[i]),
                nn.Sigmoid()
            ]
            mv_decoder_mlp.append(nn.Sequential(*decoder_mlp))

        # Register modules
        self.mv_in_to_feature = nn.ModuleList(mv_encoder_mlp)
        self.mv_feature_peculiar = nn.ModuleList(mv_feature_peculiar)
        self.mv_peculiar_to_mu_logvar = nn.ModuleList(mv_peculiar_to_mu_logvar)
        self.mv_feature_to_common = nn.ModuleList(mv_feature_to_common)
        self.mv_common_to_fusion = ViewFusionModule(fusion_dim=self.fusion_dim, num_heads=4)

        self.fusion_to_cluster = fusion_to_cluster
        self.mv_decoder_mlp = nn.ModuleList(mv_decoder_mlp)
        
        # Cluster layer (label supervision)
        self.cluster_layer = nn.Parameter(torch.zeros([self.cluster_num, self.cluster_var_dim], dtype=torch.float32))

    def encoder(self, X):
        '''
        :param X: 3 * b * d (three views, b = batch size, d = feature dimension)
        :return: mu, logvar, common features
        '''
        mu = []
        logvar = []
        common = []
        mv_common_peculiar = []  # Renamed for consistency

        for net_index in range(self.view_num):
            view_index = net_index
            feature = self.mv_in_to_feature[net_index](X[view_index])
            peculiar = self.mv_feature_peculiar[net_index](feature)
            
            mu.append(self.mv_peculiar_to_mu_logvar[net_index](peculiar))
            logvar.append(self.mv_peculiar_to_mu_logvar[net_index](peculiar))
            
            temp = self.mv_feature_to_common[net_index](feature)  # Common features of single view
            common.append(temp)
            mv_common_peculiar.append([peculiar, temp])

        # Concatenate common features across views
        mv_common = torch.concat(common, dim=1)
        mv_common = torch.unsqueeze(mv_common, dim=1)
        fusion = self.mv_common_to_fusion(mv_common)
        fusion = fusion.reshape([mv_common.shape[0], -1])
        cluster_var = self.fusion_to_cluster(fusion)

        return mv_common_peculiar, fusion, cluster_var, mu, logvar

    def reparameterization(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(std.shape).to(self.device)
            return mu + std * eps
        else:
            # Reconstruction mode (inference)
            return mu

    def decoder(self, peculiar_and_common):
        out_list = []
        for i in range(self.view_num):
            temp = self.mv_decoder_mlp[i](peculiar_and_common[i])
            out_list.append(temp)
        return out_list

    def forward(self, X):
        latent_dist = dict()
        mv_common_peculiar, fusion, cluster_var, mu, logvar = self.encoder(X)

        # Compute latent variable Z
        peculiar_and_common = []
        z = []
        for i in range(self.view_num):
            bn = nn.BatchNorm1d(mu[i].shape[1]).to(self.device)
            z.append(self.reparameterization(mu[i], logvar[i]))
            peculiar_and_common.append(torch.concat([z[i], cluster_var], dim=1))

        out_list = self.decoder(peculiar_and_common)

        # **Optimized Student’s t-distribution for q calculation**
        squared_distance = torch.sum(torch.pow(cluster_var.unsqueeze(1) - self.cluster_layer, 2), dim=2)
        q = 1.0 / (1.0 + squared_distance / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()  # Normalization

        # **Optimized p calculation to avoid drastic fluctuations**
        p = target_distribution(q)

        # Store latent variables
        latent_dist['mu_logvar'] = [mu, logvar]
        latent_dist['fusion'] = fusion
        latent_dist['Mv_common_peculiar'] = mv_common_peculiar
        latent_dist['z'] = z
        latent_dist['cluster_var'] = cluster_var
        latent_dist['q'] = q
        latent_dist['p'] = p  # **Optimized target distribution p**

        return out_list, latent_dist
