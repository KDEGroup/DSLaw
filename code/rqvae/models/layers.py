import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from sklearn.cluster import KMeans
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

class MLPLayers(nn.Module):

    def __init__(
        self, layers, dropout=0.0, activation="leakyrelu", norm_type='bn'
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.norm_type = norm_type  # 'bn', 'ln', or None

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            # mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            # Add BatchNorm or LayerNorm based on norm_type
            if idx != (len(self.layers)-2):
                if self.norm_type == 'bn':
                    mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
                elif self.norm_type == 'ln':
                    mlp_modules.append(nn.LayerNorm(output_size))
            else:
                mlp_modules.append(nn.LayerNorm(output_size))

            # Add activation function
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None and idx != (len(self.layers)-2):
                mlp_modules.append(activation_func)
            # Add Dropout after the activation function, but skip it for the last layer
            if self.dropout > 0 and idx != len(self.layers) - 2:  # Skip dropout for the last layer
                mlp_modules.append(nn.Dropout(p=self.dropout))

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if isinstance(module, nn.ReLU):
                nn.init.kaiming_normal_(module.weight.data, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        size = input_feature.shape
        input_feature = input_feature.view(-1, size[-1])
        output_feature = self.mlp_layers(input_feature)
        return output_feature.view(*(size[:-1]), self.layers[-1])

class codebook_MLP(nn.Module):
    def __init__(self, input_size, output_size, bn=False, activation='tanh'):
        super(codebook_MLP, self).__init__()
        hidden_dim = 16
        
        layers = []
        layers.append(nn.Dropout(.8))
        layers.append(nn.Linear(input_size, hidden_dim, bias=False))
        layers.append(nn.Linear(hidden_dim, output_size, bias=False))
        layers.append(activation_layer(activation))
        
        self.mlp = nn.Sequential(*layers)
        
        self.apply(self.init_weights)  # 初始化
        self.ln = nn.LayerNorm(input_size)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.mlp(x)
        return self.ln(x + out)

def codebook_encoders(e_dimlist, bn=True, activation='sigmoid'):
    activation = activation
    mlp_modules = []
    for input_size, output_size in zip(e_dimlist[:-1], e_dimlist[1:]):
        mlp_modules.append(codebook_MLP(input_size, output_size, bn=bn, activation=activation))
    return nn.ModuleList(mlp_modules)

def codebook_decoders(encode_dim, e_dimlist, bn=True, activation='sigmoid'):
    e_dimlist = e_dimlist[::-1] + [encode_dim]
    activation = activation
    mlp_modules = []
    for input_size, output_size in zip(e_dimlist[:-1], e_dimlist[1:]):
        mlp_modules.append(codebook_MLP(input_size, output_size, bn=bn, activation=activation))
    return nn.ModuleList(mlp_modules)

import torch.nn as nn

def activation_layer(activation_name="relu", emb_dim=None):
    if activation_name is None or activation_name.lower() == "none":
        return None

    activation_dict = {
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "swish": nn.SiLU,   # Swish ≈ SiLU (x * sigmoid(x))
    }

    activation_name = activation_name.lower()
    if activation_name in activation_dict:
        return activation_dict[activation_name]()
    else:
        raise NotImplementedError(f"Activation function {activation_name} is not implemented")

def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
):
    B, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    x = samples.cpu().detach().numpy()

    cluster = KMeans(n_clusters = num_clusters, max_iter = num_iters).fit(x)

    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers).to(device)

    return tensor_centers

def gumbel_softmax(logits, tau=1.0):
        # Gumbel noise generation
        gumbel_noise = -torch.log(torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)  # Gumbel noise
        logits = (logits + gumbel_noise) / tau  # Add Gumbel noise and scale by tau
        
        # Apply softmax to get probabilities
        prob_dist = F.softmax(logits, dim=-1)
        return prob_dist

# @torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    Q = torch.exp(-distances / epsilon)

    B = Q.shape[0] # number of samples to assign
    K = Q.shape[1] # how many centroids per block (usually set to 256)

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    # print(Q.sum())
    for it in range(sinkhorn_iterations):

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K


    Q = Q * B # the colomns must sum to 1 so that Q is an assignment
    return Q

