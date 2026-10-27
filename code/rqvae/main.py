import argparse
import random
import torch
import numpy as np
from time import time
import logging
# import wandb
from torch.utils.data import DataLoader
import json

from datasets import EmbDataset, PairedDataset
from models.rqvae import RQVAE
from trainer import  Trainer
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")

    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('--eval_step', type=int, default=1000, help='eval step')
    parser.add_argument('--save_step', type=int, default=1000, help='save step')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')

    # 训练集相关路径
    parser.add_argument("--train_data_path", type=str, default="../data", help="Path to the raw training data.")
    parser.add_argument('--train_ids_path', type=str, default="../data", help="Path to the training sample ID list.")
    parser.add_argument('--train_pos_dict_path', type=str, default="../data", help="Path to the positive samples dictionary for training.")
    parser.add_argument('--train_neg_dict_path', type=str, default="../data", help="Path to the negative samples dictionary for training.")

    # 验证集相关路径
    parser.add_argument("--valid_data_path", type=str, default="../data", help="Path to the raw validation data.")
    parser.add_argument('--valid_ids_path', type=str, default="../data", help="Path to the validation sample ID list.")
    parser.add_argument('--valid_pos_dict_path', type=str, default="../data", help="Path to the positive samples dictionary for validation.")
    parser.add_argument('--valid_neg_dict_path', type=str, default="../data", help="Path to the negative samples dictionary for validation.")

    # 嵌入字典路径
    parser.add_argument('--emb_path', type=str, default="../data", help="Path to the precomputed embedding dictionary.")

    # 训练参数设置
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--norm_type", type=str, default='bn', help="use ln or bn or None")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--neg_num", type=int, default=50, help="num of negtive samples")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument("--device", type=str, default="cuda:4", help="gpu or cpu")

    #模型参数设置
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256,256], help='emb num of every vq')
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')
    parser.add_argument('--e_dim', type=int, default=64, help='vq codebook embedding size')
    parser.add_argument('--e_dim_list', type=int, nargs='+', default=None, help='vq codebook embedding size list')
    parser.add_argument('--quant_loss_weight', type=float, default=0.5, help='vq quantion loss weight')
    parser.add_argument('--recon_loss_weight', type=float, default=0.1, help='vq recon loss weight')
    parser.add_argument('--margin_rate', type=float, nargs='+', default=[], help='margin rate for each quantizer')
    parser.add_argument('--mu', type=float, default=0.25, help='cl loss weight')
    parser.add_argument('--n_clusters', type=int, default=10, help='n_clusters')
    parser.add_argument('--choose_method', type=str, default="hard", help='sample_strategy')
    
    parser.add_argument("--ckpt_dir", type=str, default="./rqvae_checkpoint", help="output directory for model")
    
    return parser.parse_args()


if __name__ == '__main__':
    """fix the random seed"""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()

    # print(args)
    logging.basicConfig(level=logging.DEBUG)
    
    # with open(args.emb_dict_path, 'r') as file:
    #     emb_dict = json.load(file)
    emb_dict = np.load(args.emb_path)

    # build dataset
    train_data = np.load(args.train_data_path)
    with open(args.train_ids_path, 'r') as file:
        train_id = np.array(json.load(file)).reshape(-1, 1, 1)
    
    print("Data shape after removing pos:", train_data.shape)
    # print(train_id.shape)
    train_data = EmbDataset(train_data)
    train_id = EmbDataset(train_id)
    train_pair_data = PairedDataset(train_data, train_id)

    with open(args.train_pos_dict_path, 'r') as file:
        train_pos_dict = json.load(file)
    with open(args.train_neg_dict_path, 'r') as file:
        train_neg_dict = json.load(file)

    train_loader = DataLoader(train_pair_data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)


    # valid dataset
    valid_data = np.load(args.valid_data_path)
    with open(args.valid_ids_path, 'r') as file:
        valid_id = np.array(json.load(file)).reshape(-1, 1, 1)

    valid_data = EmbDataset(valid_data)
    valid_id = EmbDataset(valid_id)
    # print(valid_data.shape)
    # print(valid_id.shape)
    valid_pair_data = PairedDataset(valid_data, valid_id)
    
    with open(args.valid_pos_dict_path, 'r') as file:
        valid_pos_dict = json.load(file)
    with open(args.valid_neg_dict_path, 'r') as file:
        valid_neg_dict = json.load(file)
    
    valid_loader = DataLoader(valid_pair_data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)

    model = RQVAE(in_dim=train_data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  e_dim_list=args.e_dim_list,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  norm_type=args.norm_type,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  recon_loss_weight=args.recon_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  margin_rate=args.margin_rate,
                  mu=args.mu,
                  n_clusters= args.n_clusters,
                  choose_method=args.choose_method
                  )
    # print(model)
    
    
    # for batch in valid_loader:
    #     inputs, pos, id = batch  # 这两个是 tensor
    #     print(inputs.shape)      # (batch_size, 3, 768)
    #     print(pos.shape)         # (batch_size, 768)
    #     print(id.shape)
    #     break

    trainer = Trainer(args, model)
    best_loss = trainer.fit(train_loader, train_pos_dict, train_neg_dict, 
                            valid_loader, valid_pos_dict, valid_neg_dict, emb_dict)
    print("Best Loss", best_loss)




