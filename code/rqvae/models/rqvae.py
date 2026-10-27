import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# import wandb
import random
import collections
from .layers import MLPLayers
from .rq import ResidualVectorQuantizer
torch.autograd.set_detect_anomaly(True)

class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=512,
                 num_emb_list=None,
                 e_dim=64,
                 e_dim_list=None,
                 layers=None,
                 dropout_prob=0.0,
                 norm_type='ln',
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 recon_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 margin_rate=None,
                 id_decay=0.9,
                 mu = 1.0,
                 n_clusters = 10,
                 choose_method = 'hard'
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.e_dim_list = e_dim_list
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.norm_type = norm_type
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.id_decay = id_decay
        self.margin_rate = margin_rate
        self.mu = mu
        self.n_clusters = n_clusters
        self.choose_method = choose_method

        if self.e_dim_list is not None:
            self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim_list[0]]
            e_dim = self.e_dim_list[0]
        else:
            self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,norm_type=self.norm_type)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim, self.e_dim_list,
                                          mu=self.mu, kmeans_init=self.kmeans_init,
                                          kmeans_iters=self.kmeans_iters,
                                          margin_rate=self.margin_rate)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                dropout=self.dropout_prob,norm_type=self.norm_type)

    def forward(self, q, p=None, n=None, mode='train'):
        # q_e = self.encoder(q)

        # if mode == 'train' or mode == 'val':
        #     with torch.no_grad():
        #         p_e = self.encoder(p)
        #         n_e = self.encoder(n)
        q_e, p_e, n_e = q, p, n

        q_q, q_loss, cl_loss, layer_loss, indices, sim_dict = self.rq(q_e, p_e, n_e, mode=mode)
            
        q_rec = self.decoder(q_q)

        recon_loss = self.compute_loss(q_rec, q)
        
        q_loss = self.quant_loss_weight * q_loss
        recon_loss = self.recon_loss_weight * recon_loss
        total_loss = cl_loss + recon_loss + q_loss
        return q, q_q, q_rec, q_loss, cl_loss, layer_loss, recon_loss, total_loss, sim_dict, indices

    def vq_initialization(self, x):
        # self.rq.vq_ini(self.encoder(x))
        self.rq.vq_ini(x)

    @torch.no_grad()
    def get_ids(self, x, mode='test'):
        # x_e = self.encoder(x)
        x_e = x
        x_quant, ids = self.rq(x_e, mode=mode)
        return x_quant, ids

    def compute_loss(self, q_rec, q):
        # Reconstruction loss（只对 query 用）
        if self.loss_type == 'mse':
            recon_loss = F.mse_loss(q_rec, q, reduction='mean')
        elif self.loss_type == 'l1':
            recon_loss = F.l1_loss(q_rec, q, reduction='mean')
        else:
            raise ValueError('Invalid loss type')

        return recon_loss
