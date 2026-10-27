
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
from torch_geometric.nn import GCNConv

class MultiSliceGAT(torch.nn.Module):
    def __init__(self, in_dims, hidden_dim, out_dim, option_num, charge_num, charge_th, id_th,
                 train_emb, train_ids, train_charge_dict, train_id_dict,
                 val_emb, val_ids, val_charge_dict, val_id_dict,
                 train_white, val_white,
                 charge_emb, id_emb, mer_type, heads, device):
        super().__init__()
        self.k = len(option_num)
        self.mer_type = mer_type
        self.mode = None
        self.device = device
        self.train_white = torch.tensor(train_white).to(self.device)
        self.val_white = torch.tensor(val_white).to(self.device)

        # self.gats = torch.nn.ModuleList([
        #     torch.nn.Sequential(
        #         GATConv(in_dims[0], hidden_dim, heads=heads, dropout=0.2),
        #         torch.nn.ELU(),
        #         GATConv(hidden_dim * heads, in_dims[0], heads=1, dropout=0.2)
        #     ) for i in range(self.k)
        # ])

        self.gat_shared1 = GATConv(in_dims[0], hidden_dim, heads=heads, concat=True)
        # self.gat_shared_mid1 = GATConv(hidden_dim*heads, hidden_dim, heads=heads, concat=True)
        # self.gat_shared_mid2 = GATConv(hidden_dim*heads, hidden_dim, heads=heads, concat=True)
        self.gat_shared2 = GATConv(hidden_dim*heads, in_dims[0], heads=1, concat=False)

        # self.gat_shared = GATConv(in_dims[0], in_dims[0], heads=1, concat=False)

        self.act = torch.nn.ELU()

        self.charge_emb = charge_emb
        self.new_nodes = id_emb

        train_graph, self.train_num = self.build_graph_list(train_emb, train_ids, self.charge_emb, train_charge_dict, charge_th)
        # self.graph_info(train_graph)
        self.train_graphs, self.inverse_train_ids = self.create_edges_from_dict(train_graph, train_id_dict, id_th)
        # self.graph_info(self.train_graphs)
        val_graph, self.val_num = self.build_graph_list(val_emb, val_ids, self.charge_emb, val_charge_dict, charge_th)
        # self.graph_info(val_graph)
        self.val_graphs, self.inverse_val_ids = self.create_edges_from_dict(val_graph, val_id_dict, id_th)
        # self.graph_info(self.val_graphs)

        self.id2real_id_train = {v: k for k, v in enumerate(train_ids)}
        self.id2real_id_val = {v: k for k, v in enumerate(val_ids)}

    def build_graph_list(self, emb, ids, charge_emb, inverse_charge_dict, charge_th):
        n, k = emb.shape[0], emb.shape[1]
        assert len(ids) == n, "Length of ids must match n in np_array"


        ori_ids = []
        node_types = []
        edge_index = []


        for j in range(n):
            ori_ids.append(ids[j])
            node_types.append(0)


        for key, linked_ids in inverse_charge_dict.items():
            # print(inverse_charge_dict)
            ori_ids.append(key)
            node_types.append(1)
            keyword_idx = len(ori_ids) - 1

            for lid in linked_ids:
                if int(lid) in ids:
                    src = int(np.where(ids == int(lid))[0][0])
                    edge_index.append([src, keyword_idx])
                    edge_index.append([keyword_idx, src])
                else:
                    print(key, int(lid))
                    raise KeyError

        graph_list = []
        for i in range(k):
            emb_i = np.array([emb[j, i] for j in range(n)], dtype=np.float32)  # shape: (n, d_i)
            emb_i = torch.tensor(emb_i, dtype=torch.float32) if isinstance(emb_i, np.ndarray) else emb_i
            emb_i = emb_i.to(self.device)
            charge_i = np.array([charge_emb[j, i] for j in range(len(charge_emb))], dtype=np.float32)  # shape: (n, d_i)
            # print(emb_i.shape, charge_i.shape)
            charge_sim = cosine_similarity(charge_i)
            # print(charge_sim.shape)
            charge_i = torch.tensor(charge_i).to(self.device)

            feature = torch.cat([emb_i, charge_i], dim=0)  # (n + m, d_i)

            single_edge_index= edge_index.copy()
            cnt = 0
            m = charge_sim.shape[0]


            if charge_th<=1.0:
                for i in range(m):
                    for j in range(i + 1, m):
                        if charge_sim[i, j] > charge_th:
                            cnt+=1
                            idx_i = n + i
                            idx_j = n + j
                            single_edge_index.append([idx_i, idx_j])
                            single_edge_index.append([idx_j, idx_i])

            else:
                triu_i, triu_j = np.triu_indices(m, k=1)
                sim_values = charge_sim[triu_i, triu_j]

                topk = int(charge_th)
                topk_idx = np.argpartition(-sim_values, topk)[:topk]
                for i, j in zip(triu_i[topk_idx], triu_j[topk_idx]):
                    cnt+=1
                    idx_i = n + i
                    idx_j = n + j
                    single_edge_index.append([idx_i, idx_j])
                    single_edge_index.append([idx_j, idx_i])

            node_type_tensor = torch.tensor(node_types, dtype=torch.long)
            edge_index_tensor = torch.tensor(single_edge_index, dtype=torch.long).t().contiguous()
            data = Data(
                x=feature,
                edge_index=edge_index_tensor,
                ori_id=ori_ids,
                node_type=node_type_tensor
            )
            graph_list.append(data)

        return graph_list, n

    def create_edges_from_dict(self, graphs, id_dict, id_th):
        new_graphs = []
        num = graphs[0].num_nodes

        for idx, graph in enumerate(graphs):
            build_dict = id_dict[idx]
            new_node = self.new_nodes[idx]

            # graph.x = torch.cat([graph.x.to(self.device), 
            #                      torch.stack(list(new_node)).to(self.device)], dim=0)
            graph.x = torch.cat([graph.x.to(self.device), 
                                 torch.tensor(new_node).to(self.device)], dim=0)

            src, dst = [], []
            for key, value in build_dict.items():
                for v in value:
                    graphnode = graph.ori_id.index(v)

                    src.append(num + key)
                    dst.append(graphnode)

                    src.append(graphnode)
                    dst.append(num + key)

            id_sim = cosine_similarity(new_node)
            cnt = 0
            m = id_sim.shape[0]

            if id_th<=1.0:
                for i in range(m):
                    for j in range(i + 1, m):
                        if id_sim[i, j] > id_th:
                            cnt+=1
                            idx_i = num + i
                            idx_j = num + j

                            src.append(idx_i)
                            dst.append(idx_j)

                            src.append(idx_j)
                            dst.append(idx_i)

            else:

                triu_i, triu_j = np.triu_indices(m, k=1)
                sim_values = id_sim[triu_i, triu_j]

                topk = int(id_th)
                topk_idx = np.argpartition(-sim_values, topk)[:topk]

                for i, j in zip(triu_i[topk_idx], triu_j[topk_idx]):
                    cnt+=1
                    idx_i = num + i
                    idx_j = num + j

                    src.append(idx_i)
                    dst.append(idx_j)

                    src.append(idx_j)
                    dst.append(idx_i)

            graph.edge_index = torch.cat([
                graph.edge_index.to(self.device),
                torch.tensor([src, dst], dtype=torch.long).to(self.device)
            ], dim=1)

            # #############
            # graph.x = graph.x.to(self.device)
            # graph.edge_index = graph.edge_index.to(self.device)
            

            new_graphs.append(graph)
        ids = graphs[0].ori_id
        inverse_ids = {id_:i for i, id_ in enumerate(ids)}

        return new_graphs, inverse_ids

    def graph_info(self, graphs):
        # 打印图信息
        for i, graph in enumerate(graphs):
            print(f"\n graph{i}:")
            print(graph.num_nodes)
            print(graph.num_edges)
            print(graph.x.shape)
            print(graph.ori_id[:10])
            print(list(zip(*graph.edge_index))[:5]) 

    def compute_loss(self, emb, pos_dict, neg_dict, loss_ids, margin=0.2, pos_num=100, neg_num=100, mode = None):

        total_loss = 0.0
        cnt = 0
        if mode == 'train':
            inverse_ids = self.id2real_id_train
        elif mode == 'val':
            inverse_ids = self.id2real_id_val
        else:
            raise KeyError("mode should be 'train' or 'val'")
        
        
        for idx, real_id in enumerate(loss_ids):
            anchor = emb[idx]
        # for idx, real_id in enumerate(loss_ids):
        #     anchor = emb[:, idx, :]
        
            pos_ids = pos_dict[real_id]
            neg_ids = neg_dict[real_id]

            if not pos_ids or not neg_ids:
                continue

            pos_ids = random.choices(pos_ids, k=pos_num)
            neg_ids = random.sample(neg_ids, k=neg_num)
            # print(len(pos_ids))
            # print(len(neg_ids))
            # breakpoint()
            pos_node = [inverse_ids[x] for x in pos_ids]
            neg_node = [inverse_ids[x] for x in neg_ids]
            pos_emb = emb[pos_node]
            neg_emb = emb[neg_node]

            # info_nce
            pos_sim = torch.cosine_similarity(anchor.unsqueeze(0), pos_emb, dim=-1)  # shape [P]
            neg_sim = torch.cosine_similarity(anchor.unsqueeze(0), neg_emb, dim=-1)  # shape [N]
            
            sim = torch.cat([pos_sim, neg_sim], dim=-1)  # [P + N]
            logits = sim / 0.07  # [P + N]
            pos_logits = logits[:len(pos_sim)]  # 前 P 个为正样本
            denom = torch.logsumexp(logits, dim=0)  # scalar
            log_probs = pos_logits - denom  # shape: [P]
            loss = -log_probs.mean()

            # pos_sim = torch.cosine_similarity(anchor.unsqueeze(1), pos_emb, dim=-1)  # shape [P]
            # neg_sim = torch.cosine_similarity(anchor.unsqueeze(1), neg_emb, dim=-1)  # shape [N]
            
            # logits = torch.cat([pos_sim, neg_sim], dim=1) / 0.07  # shape: (a, 2b)
            # pos_logits = logits[:, :pos_sim.shape[1]]  # shape: (a, b)
            # logsumexp = torch.logsumexp(logits, dim=1, keepdim=True)  # shape: (a, 1)
            # log_probs = pos_logits - logsumexp
            # per_anchor_loss = -log_probs.mean(dim=1)  # shape: (a,)
            # loss = per_anchor_loss.sum()  # scalar

            total_loss += loss
            cnt += 1

        return total_loss/cnt
    
    def degree_regular_loss(self, emb):
        degree_regular = 0
        X_norm = F.normalize(emb, p=2, dim=1)
        degree_regular = degree_regular + torch.matmul(X_norm, X_norm.T).mean()
        # for e in emb:
        #     degree_regular = degree_regular + F.cosine_similarity(e.unsqueeze(0), emb.unsqueeze(1), dim=-1).mean()

        print(degree_regular)
        return degree_regular

    def forward(self, graphs, case_num, mode):
        assert len(graphs) == self.k
        outputs = []

        # for i in range(self.k):
        #     if i == 0:
        #         x = graphs[i].x
        #     else:
        #         x = graphs[i].x + outputs[-1]
        #     edge_index = graphs[i].edge_index
        #     out = self.gats[i][0](x, edge_index)
        #     out = self.gats[i][1](out)
        #     out = self.gats[i][2](out, edge_index)
        #     outputs.append(out)
        
        for i in range(self.k):
            if i == 0:
                x = graphs[i].x
            else:
                x = graphs[i].x + outputs[-1]
            edge_index = graphs[i].edge_index
            # out = self.gat_shared(x, edge_index)
            out = self.gat_shared1(x, edge_index)
            out = self.act(out)
            # out = self.gat_shared_mid1(out, edge_index)
            # out = self.act(out)
            # out = self.gat_shared_mid2(out, edge_index)
            # out = self.act(out)
            out = self.gat_shared2(out, edge_index)
            outputs.append(out)
        outputs = torch.stack(outputs)
        ret_emb = outputs[-1, :case_num]
        # ret_emb = outputs[-1][:case_num]

        # if mode == 'train_graph':
        #     ret_emb = ret_emb + self.train_white
        # elif mode == 'val_graph':
        #     ret_emb = ret_emb + self.val_white
            
        # if self.mer_type == 'cat':
        #     ret_emb = torch.cat(outputs, dim=1)
        # elif self.mer_type == 'sum':
        #     ret_emb = torch.stack(outputs).sum(dim=0)  # shape: [num_nodes, out_dim]
        #     ret_emb = ret_emb + (graphs[-1].x)[:case_num]
        # else:
        #     raise ValueError(f"Invalid mer_type: {self.mer_type}. Choose 'cat' or 'sum'.")
        # ret_emb = self.dp(ret_emb)
        # ret_emb = self.fc(ret_emb)

        return ret_emb, outputs[:, :case_num, :]

