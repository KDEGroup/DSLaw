import logging
import csv
import numpy as np
import torch
import random
from time import time
from torch import optim
from tqdm import tqdm

import torch.nn.functional as F
from utils import ensure_dir,set_color,get_local_time
import os
from datasets import Dataset_init
from models.vq import VectorQuantizer
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)

class Trainer(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.save_step = min(args.save_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)

        self.neg_num = args.neg_num
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir, saved_model_dir)
        ensure_dir(self.ckpt_dir)
        self.train_record_pth = os.path.join(self.ckpt_dir, 'train_loss.csv')
        self.val_record_pth = os.path.join(self.ckpt_dir, 'val_loss.csv')

        self.best_loss = np.inf
        self.model = self.model.to(self.device)

        self.opt = self._build_optimizer()
    
    def _build_optimizer(self):
        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def constrained_km(self, data, n_clusters=10):
        from k_means_constrained import KMeansConstrained 
        # x = data.cpu().detach().numpy()
        # data = self.embedding.weight.cpu().detach().numpy()
        x = data
        size_min = min(len(data) // (n_clusters * 2), 12)
        clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=n_clusters * 6, max_iter=10, n_init=10,
                                n_jobs=10, verbose=False)
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()

        return t_centers, t_labels
    
    def vq_init(self):
        self.model.eval()

        original_data = Dataset_init(self.args.train_data_path)
        init_loader = DataLoader(original_data,num_workers=self.args.num_workers,
                             batch_size=len(original_data), shuffle=True,
                             pin_memory=True)

        iter_data = tqdm(
                    init_loader,
                    total=len(init_loader),
                    ncols=100,
                    desc=set_color(f"Initialization of vq","pink"),
                    )
        
        # Train
        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1]
            text = data.to(self.device)
            # print(data.shape)
            self.model.vq_initialization(text)    

    def accumulate_sim(self, accumulated_sim_dict, sim_dict):
        accumulated_sim_dict["pos_sim"] += sim_dict["pos_sim"]
        accumulated_sim_dict["neg_sim"] += sim_dict["neg_sim"]

    def _train_epoch(self, train_data, pos_dict, neg_dict, emb_dict, epoch_idx):

        self.model.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_cl_loss = 0.0
        total_layer_loss = [0.0] * len(self.args.margin_rate)
        total_quant_loss = 0.0

        codebook_ind = []
        cnt = 0
        accumulated_sim_dict = {
            "pos_sim": 0.0,
            "neg_sim": 0.0
        }

        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )

        for batch_idx, data in enumerate(iter_data):
            cnt += 1
            data, ind = data[0], data[1].squeeze()

            degree_regular = F.cosine_similarity(data.unsqueeze(0), data.unsqueeze(1), dim=2)
            degree_regular = degree_regular.mean()

            data, pos, neg = self._sample(data, pos_dict, self.neg_num, neg_dict, self.neg_num, ind, emb_dict)

            data = data.to(self.device)
            pos = pos.to(self.device)
            neg = neg.to(self.device)

            x_weighted, x_q, x_rec, q_loss, cl_loss, layer_loss, recon_loss, loss, sim_dict, indices = self.model(
                data, pos, neg, mode="train")
            
            # cal codebook frequency
            codebook_ind.append(indices)
            self.accumulate_sim(accumulated_sim_dict, sim_dict)
            
            self._check_nan(loss)

            # 模型参数更新
            self.opt.zero_grad()
            loss = loss+degree_regular
            loss.backward()
            self.opt.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_cl_loss +=cl_loss.item()
            for i, (a, b) in enumerate(zip(total_layer_loss, layer_loss)):
                total_layer_loss[i] = a + b
            total_quant_loss += q_loss.item()
        # self.opt.step()
        codebook_ind = torch.cat(codebook_ind, dim=0)
        sim_dict = {
            "pos_sim": accumulated_sim_dict["pos_sim"] / cnt,
            "neg_sim": accumulated_sim_dict["neg_sim"] / cnt
        }

        total_loss = total_loss / cnt
        total_recon_loss = total_recon_loss / cnt
        total_cl_loss = total_cl_loss / cnt
        total_layer_loss = [total_layer_loss[i] / cnt for i in range(len(total_layer_loss))]
        total_quant_loss = total_quant_loss / cnt

        return total_loss, total_recon_loss, total_cl_loss, total_layer_loss, total_quant_loss, sim_dict, codebook_ind

    @torch.no_grad()
    def _valid_epoch(self, valid_data, pos_dict, neg_dict, emb_dict, epoch_idx):
        self.model.eval()

        total_loss = 0
        total_recon_loss = 0
        total_cl_loss = 0
        total_layer_loss = [0.0] * len(self.args.margin_rate)
        total_quant_loss = 0

        codebook_ind = []
        cnt = 0
        accumulated_sim_dict = {
            "pos_sim": 0.0,
            "neg_sim": 0.0
        }

        iter_data = tqdm(
            valid_data,
            total=len(valid_data),
            ncols=100,
            desc=set_color(f"Valid {epoch_idx}", "blue"),
        )

        for batch_idx, data in enumerate(iter_data):
            cnt += 1
            data, ind = data[0], data[1].squeeze()

            data, pos, neg = self._sample(data, pos_dict, self.neg_num, neg_dict, self.neg_num, ind, emb_dict)                    

            data = data.to(self.device)
            pos = pos.to(self.device)
            neg = neg.to(self.device)
            _, _, _, q_loss, cl_loss, layer_loss, recon_loss, loss, sim_dict, indices  = self.model(
                data, pos, neg, mode="val")
            
            # cal codebook frequency
            codebook_ind.append(indices)
            self.accumulate_sim(accumulated_sim_dict, sim_dict)

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_cl_loss += cl_loss.item()
            for i, (a, b) in enumerate(zip(total_layer_loss, layer_loss)):
                total_layer_loss[i] = a + b
            total_quant_loss += q_loss.item()
        
        codebook_ind = torch.cat(codebook_ind, dim=0)
        sim_dict = {
            "pos_sim": accumulated_sim_dict["pos_sim"] / cnt,
            "neg_sim": accumulated_sim_dict["neg_sim"] / cnt
        }

        total_loss = total_loss / cnt
        total_recon_loss = total_recon_loss / cnt
        total_cl_loss = total_cl_loss / cnt
        total_layer_loss = [total_layer_loss[i] / cnt for i in range(len(total_layer_loss))]
        total_quant_loss = total_quant_loss / cnt

        return total_loss, total_recon_loss, total_cl_loss, total_layer_loss, total_quant_loss, sim_dict, codebook_ind

    def _save_checkpoint(self, epoch, loss, val_loss=None, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_loss_%.4f_model.pth' % (epoch, loss))
        state = {
            "args": self.args,
            "epoch": epoch,
            "loss": loss,
            "val_loss": val_loss,
            "best_loss": self.best_loss,
            "state_dict": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            # "scheduler": self.scheduler.state_dict()
            # "opt_main": self.opt_main.state_dict(),
            # "opt_codebook": self.opt_codebook.state_dict(),
            # "scheduler_main": self.scheduler_main.state_dict(),
            # "scheduler_codebook": self.scheduler_codebook.state_dict()
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

    def _generate_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss, cl_loss, quant_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("recon loss", "blue") + ": %.4f" % recon_loss
        train_loss_output +=", "
        train_loss_output += set_color("cl loss", "blue") + ": %.4f" % cl_loss
        train_loss_output +=", "
        train_loss_output += set_color("quant loss", "blue") + ": %.4f" % quant_loss

        return train_loss_output + "]"
    
    def _generate_loss_output_val(self, epoch_idx, s_time, e_time, loss, recon_loss, cl_loss, quant_loss):
        train_loss_output = (
            set_color("epoch %d validating", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("val loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("recon loss", "blue") + ": %.4f" % recon_loss
        train_loss_output +=", "
        train_loss_output += set_color("cl loss", "blue") + ": %.4f" % cl_loss
        train_loss_output +=", "
        train_loss_output += set_color("quant loss", "blue") + ": %.4f" % quant_loss

        return train_loss_output + "]"
    
    def _sample(self, d, pos_dict, p_num, neg_dict, n_num, ind, emb_dict):
        # 创建一个列表用于存放结果
        data, pos, neg = [], [], []
        
        # 遍历每个索引
        for s_d, idx in zip(d, ind):
            idx_str = str(int(idx.item()))  # 将索引转换为字符串
            if idx_str not in pos_dict:
                raise KeyError(f"Index '{idx_str}' not found in sample_dict.")
            
            pos_sample_list = pos_dict[idx_str]
            neg_sample_list = neg_dict[idx_str]
            if len(pos_sample_list) == 0:
                continue

            # pos_sample = random.sample(pos_sample_list, p_num)
            pos_sample = random.choices(pos_sample_list, k=p_num)
            neg_sample = neg_sample_list if len(neg_sample_list) == n_num else random.sample(neg_sample_list, n_num)

            pos_sample_block = torch.from_numpy(emb_dict[pos_sample])
            neg_sample_block = torch.from_numpy(emb_dict[neg_sample])

            pos.append(pos_sample_block)
            neg.append(neg_sample_block)
            data.append(s_d)

        return torch.stack(data), torch.stack(pos), torch.stack(neg)

    def log_loss_to_csv(self, csv_path, epoch, loss_dict):
        file_exists = os.path.exists(csv_path)

        # 字段名称：epoch + 所有 loss 字段
        fieldnames = ['epoch'] + list(loss_dict.keys())

        with open(csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writeheader()

            # 构造写入行
            row = {'epoch': epoch}
            row.update({k: float(v) for k, v in loss_dict.items()})
            writer.writerow(row)

    def calculate_codebook_usage(self, indices, num_codebooks):
        bs, layer_num = indices.size()
        num_codebooks = max(self.args.num_emb_list)

        # 创建一个 [layer_num, num_codebooks] 的全零矩阵
        usage_freq = torch.zeros(layer_num, num_codebooks, device=indices.device)

        # 遍历每一层，统计每个码本索引的出现次数
        for layer in range(layer_num):
            layer_indices = indices[:, layer]  # 获取当前层的所有码本索引
            for idx in range(num_codebooks):
                usage_freq[layer, idx] = (layer_indices == idx).sum().float()

        return usage_freq.int()
    
    def fit(self, data, pos_dict, neg_dict, val_data, val_pos_dict, val_neg_dict, emb_dict):
        
        self.vq_init()
        for epoch_idx in range(self.epochs):
            self.model.train()
            # train
            training_start_time = time()
            train_loss, train_recon_loss, train_cl_loss, train_layer_loss, train_quant_loss, train_sim_dict, train_codebook_ind = self._train_epoch(data, pos_dict, neg_dict, emb_dict, epoch_idx)
            training_end_time = time()

            train_loss_output = self._generate_loss_output(
                epoch_idx, training_start_time, training_end_time, 
                train_loss, train_recon_loss, train_cl_loss, train_quant_loss
            )

            self.logger.info(train_loss_output)
            
            # cal codebook fre
            codebook_fre = self.calculate_codebook_usage(train_codebook_ind, self.args.num_emb_list[0])
            print(codebook_fre)

            # loss保存
            loss_dict = {
                "total_loss": train_loss,
                "recon_loss": train_recon_loss,
                "cl_loss": train_cl_loss,
                "quant_loss": train_quant_loss,
                "codebook_fre": calculate_non_zero_ratio(codebook_fre)
            }

            for i, layer_loss in enumerate(train_layer_loss):
                loss_dict[f"layer{i}"] = layer_loss

            save_dict = loss_dict | train_sim_dict

            self.log_loss_to_csv(self.train_record_pth, epoch_idx, save_dict)

            # validate
            if (epoch_idx + 1) % self.eval_step == 0:
                self.model.eval()
                val_start_time = time()
                val_loss, val_recon_loss, val_cl_loss, val_layer_loss, val_quant_loss, val_sim_dict, val_codebook_ind = self._valid_epoch(val_data, val_pos_dict, val_neg_dict, emb_dict, epoch_idx)
                
                val_end_time = time()
                val_loss_output = self._generate_loss_output_val(
                epoch_idx, val_start_time, val_end_time, 
                val_loss, val_recon_loss, val_cl_loss, val_quant_loss
                )
                
                self.logger.info(val_loss_output)

                # cal codebook fre
                codebook_fre = self.calculate_codebook_usage(val_codebook_ind, self.args.num_emb_list[0])
                print(codebook_fre)

                for i,layer_loss in enumerate(train_layer_loss):
                    loss_dict[f"layer{i}"] = layer_loss

                val_loss_dict = {
                    "total_loss": val_loss,
                    "recon_loss": val_recon_loss,
                    "cl_loss": val_cl_loss,
                    "quant_loss": val_quant_loss,
                    "codebook_fre": calculate_non_zero_ratio(codebook_fre)
                }
                
                for i, layer_loss in enumerate(val_layer_loss):
                    val_loss_dict[f"layer{i}"] = layer_loss

                val_dict = val_loss_dict | val_sim_dict
                self.log_loss_to_csv(self.val_record_pth, epoch_idx, val_dict)

                self.best_loss = min(self.best_loss, val_loss)

            if (epoch_idx + 1) % self.save_step == 0:
                self._save_checkpoint(epoch_idx, train_loss, val_loss)

            

        return self.best_loss


def print_grad(name):
    def hook(grad):
        print(f"[GRAD HOOK] {name} grad mean = {grad.abs().mean():.6e}, shape = {grad.shape}")
    return hook

def calculate_non_zero_ratio(tensor):
    # 计算张量中 0 的个数
    zero_count = torch.sum(tensor == 0).item()

    # 计算总元素个数
    total_elements = tensor.numel()

    # 计算 0 的比例
    zero_ratio = zero_count / total_elements

    # 返回 1 减去 0 的比例
    return 1 - zero_ratio


