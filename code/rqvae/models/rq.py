import torch
import torch.nn as nn
import torch.nn.functional as F

from .vq import VectorQuantizer
from .layers import codebook_encoders, codebook_decoders
torch.autograd.set_detect_anomaly(True)

class L2NormScale(nn.Module):
    def __init__(self, scale=5.0, dim=-1, eps=1e-6):
        super().__init__()
        self.scale = scale
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return self.scale * F.normalize(x, p=2, dim=self.dim, eps=self.eps)

class ResidualVectorQuantizer(nn.Module):

    def __init__(self, n_e_list, e_dim, e_dim_list, mu, 
                 kmeans_init = False, kmeans_iters = 100, margin_rate=None, 
                 distance_flag='distance'):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.e_dim_list = e_dim_list
        self.mu = mu
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.margin_rate = margin_rate
        self.distance_flag = distance_flag
        self._diff_codebook = e_dim_list is not None

        # self._diff_codebook = False
        self._norm = False

        if self._diff_codebook:
            self.vq_encoders = codebook_encoders(e_dim_list, bn=False, 
                                                activation='swish')
            self.vq_decoders = codebook_decoders(e_dim, e_dim_list, bn=False,
                                                activation='swish')

        if self._diff_codebook:
            self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim, mu=self.mu,
                                                            kmeans_init = self.kmeans_init,
                                                            kmeans_iters = self.kmeans_iters,
                                                            distance_flag=self.distance_flag)
                                            for n_e, e_dim in zip(n_e_list,e_dim_list) ])
            self.norms = nn.ModuleList([nn.LayerNorm(normalized_shape=codebook_dim) 
                                        for codebook_dim in e_dim_list])
            # self.norms = nn.ModuleList([L2NormScale(scale=5.0, dim=-1) 
            #                             for _ in range(self.num_quantizers)])
        else:
            self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim, mu=self.mu,
                                                            kmeans_init = self.kmeans_init,
                                                            kmeans_iters = self.kmeans_iters,
                                                            distance_flag=self.distance_flag)
                                            for n_e in n_e_list ])
            self.norms = nn.ModuleList([nn.LayerNorm(normalized_shape=self.e_dim) 
                                        for _ in range(self.num_quantizers)])
            # self.norms = nn.ModuleList([L2NormScale(scale=5.0, dim=-1) 
            #                             for _ in range(self.num_quantizers)])

    def get_codebook(self):
        all_codebook = {}
        for idx, quantizer in enumerate(self.vq_layers):
            codebook = quantizer.get_codebook()
            all_codebook[idx] = codebook  # 以索引为键，将每个量化器的码本存入字典
        return all_codebook
    
    def vq_ini(self, x):
        residual = x
        init_emb_norm = []
        init_res_norm = []
        init_res_norm.append(torch.norm(residual,p=2,dim=-1).mean().detach().cpu().item())
        
        for idx, (quantizer, norm) in enumerate(zip(self.vq_layers, self.norms)):
            
            if self._diff_codebook:
                emb_dim = self.e_dim_list[idx]
                if idx:
                    residual = self.vq_encoders[idx-1](residual)
                    # print(self.vq_encoders[idx].mlp[0].weight.shape)
            else:
                emb_dim = self.e_dim

            x_res = quantizer.vq_init(residual, emb_dim)
            residual = residual - x_res
            if self._norm:
                residual = norm(residual)
            
            print()

            init_emb_norm.append(torch.norm(quantizer.get_codebook(),p=2,dim=-1).mean().detach().cpu().item())
            print(torch.max(torch.norm(quantizer.get_codebook(),p=2,dim=-1)), torch.min(torch.norm(quantizer.get_codebook(),p=2,dim=-1)))
        
            init_res_norm.append(torch.norm(residual,p=2,dim=-1).mean().detach().cpu().item())
            print(torch.max(torch.norm(residual,p=2,dim=-1)), torch.min(torch.norm(residual,p=2,dim=-1)))
        
        print("init norm res in layers:",init_res_norm)
        print("init norm emb in layers:",init_emb_norm)

    def get_distance(self, M, N, cal='cos'):
        """
        Args:
            M: torch.Tensor, shape [bs, dim]
            N: torch.Tensor, shape [bs, k, dim]
            cal: str, 'cos' or other (for Euclidean)
        Returns:
            dis: torch.Tensor, shape [bs, k]
        """
        # if len(N.shape) == 2:
        #     N = N.unsqueeze(1)
        # print(M.shape, N.shape)
        if cal == 'cos':
            sim = F.cosine_similarity(M.unsqueeze(1), N, dim=-1)
            dis = - sim                                   # cosine "distance"
        else:
            M_exp = M.unsqueeze(1)                        # [bs, 1, dim]
            dis = torch.norm(M_exp - N, p=2, dim=2)       # [bs, k]

        return dis

    # def compute_loss(self, p_dis, n_dis, idx, temperature=0.07, ori_margin=0.2):
    #     # margin = ori_margin * self.margin_rate[idx]
    #     # margin = 0.05
    #     margin = 0.1

    #     # 合并正样本和负样本
    #     pos_sim = -p_dis
    #     pos_sim = pos_sim.mean(dim=-1)  # (bs,)

    #     neg_sim = -n_dis
    #     neg_weight = torch.softmax((neg_sim / temperature), dim=-1)
    #     neg_sim = (neg_sim * neg_weight).sum(dim=1)  # (bs,)

    #     triplet_loss = F.relu(margin - pos_sim + neg_sim).mean()
    #     if self.margin_rate[idx] == 0:
    #         ret = torch.zeros_like(triplet_loss)
    #     else:
    #         ret = triplet_loss * self.margin_rate[idx]
    #     return  ret

    def compute_loss(self, p_dis, n_dis, idx, margin=0.3):
        margin = margin

        # 距离 -> 相似度（负号）
        pos_sim = -p_dis  # shape (bs, k)
        neg_sim = -n_dis  # shape (bs, k)

        # 计算逐对 triplet loss：relu(margin - pos + neg)
        triplet_loss = F.relu(margin - pos_sim + neg_sim)  # shape (bs, k)
        # abs_loss = F.relu(neg_sim - 0.5)
        # 平均所有 triplet 对
        loss = triplet_loss.mean()# + abs_loss.mean() # scalar

        # degree_regular = pos_sim.mean() + neg_sim.mean()
        # loss+=degree_regular/10

        # 层级权重控制
        if self.margin_rate[idx] == 0:
            return torch.zeros_like(loss)
        else:
            return loss * self.margin_rate[idx]
        


    # def compute_loss(self, p_dis, n_dis, idx, temperature=0.1):
    #     """
    #     p_dis: (bs, ) or (bs, 1) → 正样本距离（越小越好）
    #     n_dis: (bs, neg_num)     → 负样本距离
    #     """

    #     # 把距离转为相似度（负号），除以温度
    #     pos_sim = -p_dis / temperature              # (bs, 1)
    #     neg_sim = -n_dis / temperature              # (bs, neg_num)
    #     # neg_sim, _ = torch.topk(neg_sim, k=10)

    #     # 拼接正负样本在 dim=1
    #     logits = torch.cat([pos_sim, neg_sim], dim=1)  # (bs, 1 + neg_num)

    #     # 创建标签（第0列是正样本）
    #     labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    #     # 使用 cross_entropy 作为 InfoNCE loss
    #     info_nce_loss = F.cross_entropy(logits, labels)

    #     # 加权（可选）
    #     if self.margin_rate[idx] == 0:
    #         return torch.zeros_like(info_nce_loss)
    #     else:
    #         return info_nce_loss * self.margin_rate[idx]


    def forward(self, x, pos=None, neg=None, mode='train'): 

        all_losses = []
        all_indices = []
        x_residuals = []

        x_q = []
        residual = x
        x_rec = 0

        sim_dict = {}
        test_quant_emb = []

        for idx, (quantizer, norm) in enumerate(zip(self.vq_layers, self.norms)):   

            if self._diff_codebook and idx:
                residual = self.vq_encoders[idx-1](residual)

            # x_residuals.append(residual)
            # test_quant_emb.append(residual)
            x_res, x_res_rec, loss, indices = quantizer(residual)
            
            residual = residual - x_res#.detach()

            x_residuals.append(residual)
            test_quant_emb.append(residual)

            if self._norm:
                residual = norm(residual)
            x_q.append(x_res_rec)
            # test_quant_emb.append(residual)

            all_losses.append(loss)
            all_indices.append(indices)

        if self._diff_codebook:
            x_rec = x_q[-1]
            for i in range(len(self.vq_decoders)):
                x_rec = self.vq_decoders[i](x_rec)
                if i < len(self.vq_decoders) - 1:
                    x_rec = x_rec + x_q[-(i + 2)]
        else:
            x_rec = torch.stack(x_q, dim=0).sum(dim=0)

        # 以样本为中心进行残差迭代
        #####################################################################
        if mode == 'train' or mode == 'val':
            all_id_losses = []
            residuals = [pos ,neg]
            print(-1, [torch.norm(x, p=2, dim=-1).mean().detach().cpu().numpy().item() for x in residuals])
            for idx, (quantizer, norm) in enumerate(zip(self.vq_layers, self.norms)):
                distances = []
                temp = []
                for i in range(len(residuals)):

                    if self._diff_codebook and idx:
                        residuals[i] = self.vq_encoders[idx-1](residuals[i])
                    temp.append(residuals[i])
                    # distance = self.get_distance(x_residuals[idx], residuals[i], "cos")
                    # distances.append(distance)

                    sample_quant, _, _, _ = quantizer(residuals[i]) 
                    residuals[i] = residuals[i] - sample_quant

                    distance = self.get_distance(x_residuals[idx], residuals[i], "cos")
                    distances.append(distance)

                    # temp.append(residuals[i])
                    if self._norm:
                        residuals[i] = norm(residuals[i])

                # if idx == (len(self.vq_layers) - 1):
                if idx == 0:
                    neg_sim, _ = torch.topk(-distances[1], k=5, dim=1)
                    sim_dict={
                        "pos_sim": -distances[0].mean().item(),
                        "neg_sim": neg_sim.mean().item()
                    }

                all_id_losses.append(self.compute_loss(*distances, idx))

                print([torch.norm(x, p=2, dim=-1).mean().detach().cpu().numpy().item() for x in temp])
                norms = [torch.norm(x, p=2, dim=-1).mean().detach().cpu().numpy().item() for x in residuals]
                print(idx, norms)
            emb_norm = [torch.norm(vq_layer.get_codebook(), p=2, dim=-1).mean().item() 
                            for vq_layer in self.vq_layers]
            print(emb_norm)
            sum_id_losses = torch.stack(all_id_losses).sum()
        else:
            sum_id_losses = torch.tensor(0.0, device=x.device, requires_grad=False)

        anchor_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        if mode == 'test':
            return test_quant_emb, all_indices
        return x_rec, anchor_losses, sum_id_losses, all_id_losses, all_indices, sim_dict
    

def print_grad(name):
    def hook(grad):
        print(f"[GRAD HOOK] {name} grad mean = {grad.abs().mean():.6e}, shape = {grad.shape}")
    return hook