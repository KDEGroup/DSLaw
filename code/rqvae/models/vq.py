import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm, gumbel_softmax
import random
torch.autograd.set_detect_anomaly(True)


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim, mu = 0.25,
                kmeans_init = False, kmeans_iters = 10,
                distance_flag='distance', affine=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.mu = mu
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.distance_flag = distance_flag
        self.affine = affine

        self.embedding = nn.Embedding(self.n_e, self.e_dim)

        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

        if self.affine:
            self.register_parameter("affine_scale", nn.Parameter(torch.ones(1, self.e_dim)))
            self.register_parameter("affine_bias", nn.Parameter(torch.zeros(1, self.e_dim)))

    def get_codebook(self):
        C = self.embedding.weight
        if self.affine:
            C = self.affine_bias + self.affine_scale * C
        return C

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        if self.affine:
            z_q = self.affine_bias + self.affine_scale * z_q
        return z_q

    def init_emb(self, data, emb_dim):
        centers, _ = self.constrained_km(data, self.n_e)
        self.embedding.weight.data.copy_(centers)
        self.initted = True
    
    def constrained_km(self, data, n_clusters=10):
        from k_means_constrained import KMeansConstrained 
        x = data.cpu().detach().numpy()

        size_min = min(len(data) // (n_clusters * 2), 150) # 50 for the very first time, 10 the latter
        clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=size_min * 4, max_iter=10, n_init=10,
                                n_jobs=10, verbose=False) # 'size_min * 4' for the very first time, 'n_clusters * 4' for the latter
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()
        return t_centers, t_labels
                    
    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances
    
    def vq_init(self, x, emb_dim):
        latent = x.view(-1, self.e_dim)
        if not self.initted:
            self.init_emb(latent, emb_dim)
        
        
        if self.distance_flag == 'distance':
            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
                2 * torch.matmul(latent, self.embedding.weight.t())
        else:    
        # Calculate Cosine Similarity 
            d = latent@self.embedding.weight.t()

        # else:
        if self.distance_flag == 'distance':
            indices = torch.argmin(d, dim=-1)
        else:    
            indices = torch.argmax(d, dim=-1)
        x_q = self.embedding(indices).view(x.shape)

        return x_q

    def forward(self, x):
        # Flatten input
        latent = x.view(-1, self.e_dim)
        C = self.get_codebook()

        # if not self.initted:
        #     self.init_emb(latent)
        
        if self.distance_flag == 'distance':
            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(C**2, dim=1, keepdim=True).t()- \
                2 * torch.matmul(latent, C.t())
        else:    
        # Calculate Cosine Similarity 
            d = latent@C.t()
        
        if self.distance_flag == 'distance':
            indices = torch.argmin(d, dim=-1)
        else:    
            indices = torch.argmax(d, dim=-1)
        x_q = F.embedding(indices, C).view(x.shape)

        # x_q.register_hook(print_grad(f"x_q"))
        # x.register_hook(print_grad(f"x"))

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())

        loss = self.mu * codebook_loss + (1 - self.mu) * commitment_loss

        # x_q_rec = x + (x_q - x).detach()
        x_q_rec = x_q + x - x.detach()

        indices = indices.view(x.shape[:-1])

        return x_q, x_q_rec, loss, indices
    
def print_grad(name):
    def hook(grad):
        print(f"[GRAD HOOK] {name} grad mean = {grad.abs().mean():.6e}, shape = {grad.shape}")
    return hook
