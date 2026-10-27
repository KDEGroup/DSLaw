import numpy as np
import json
import torch
# import pandas as pd
import argparse
import random

from models import MultiSliceGAT
from trainer import GATTrainer
from utils import inverse_dict, invert_charges, load_dict

parser = argparse.ArgumentParser()

# 通用设置
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument('--in_dim', type=int, nargs='+', default=[])
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--out_dim", type=int, default=32)
parser.add_argument("--heads", type=int, default=1)
parser.add_argument("--option_num", type=int, nargs='+', default=[])
parser.add_argument("--mer_type", type=str, choices=["sum", "cat"], default="sum")
parser.add_argument("--emb_type", type=str, choices=["residual", "raw"], default="residual")
parser.add_argument("--charge_th", type=float, default=0.6)
parser.add_argument("--id_th", type=float, default=0.6)

# 数据路径
parser.add_argument("--data_ids", type=str, default="ELAM_data_ids.npy")
parser.add_argument("--residual_emb", type=str, default="ELAM_data_quant.npy")
parser.add_argument("--raw_emb", type=str, default="../dataset/ELAM/ELAM_emb.npy")
parser.add_argument("--charge_emb", type=str, default="ELAM_charge_quant.npy")
parser.add_argument("--id_emb", type=str, default="ELAM_codebook.npy")

parser.add_argument("--train_id_path", type=str, default="../dataset/ELAM/train/train_candidate_ids.json")
parser.add_argument("--val_id_path", type=str, default="../dataset/ELAM/test/test_candidate_ids.json")
parser.add_argument("--charge_id_path", type=str, default="../dataset/ELAM/charges/ELAM_charge_id.json")
parser.add_argument("--charge_case_path", type=str, default="../dataset/ELAM/charges/ELAM_charge_case.json")

parser.add_argument("--train_pos_path", type=str, default="../dataset/ELAM/train/train_relative.json")
parser.add_argument("--train_neg_path", type=str, default="../dataset/ELAM/train/train_hard_neg.json")
parser.add_argument("--val_pos_path", type=str, default="../dataset/ELAM/test/test_relative.json")
parser.add_argument("--val_neg_path", type=str, default="../dataset/ELAM/test/test_hard_neg.json")

args = parser.parse_args()
if args.cuda>=0:
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

    
"""fix the random seed"""
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

option_num = args.option_num
in_dims = args.in_dim
hidden_dim = args.hidden_dim
out_dim = args.out_dim
epochs = args.epochs
heads = args.heads

mer_type = args.mer_type
emb_type = args.emb_type

data_ids = np.load(args.data_ids)

if emb_type == 'residual':
    data_emb = np.load(args.residual_emb, allow_pickle=True)
    expanded_array = np.expand_dims(data_emb, axis=1)
    # data_emb = np.tile(expanded_array, (1, 6, 1))
    print(data_emb.shape)
# elif emb_type == 'raw':
#     data_emb = np.load(args.raw_emb)
#     data_emb = np.tile(data_emb[:, np.newaxis, :], (1, data_ids.shape[1], 1))

with open(args.train_id_path, "r") as f:
    train_ids = np.array(json.load(f))
with open(args.val_id_path, "r") as f:
    val_ids = np.array(json.load(f))

only_val_ids = val_ids[~np.isin(val_ids, train_ids)]

charge_emb = np.load(args.charge_emb, allow_pickle=True)
id_emb = np.load(args.id_emb, allow_pickle=True)

# 构建图数据
train_rqid = np.array([data_ids[i] for i in train_ids])
train_emb = np.array([data_emb[i] for i in train_ids])
val_rqid = np.array([data_ids[i] for i in val_ids])
val_emb = np.array([data_emb[i] for i in val_ids])

inverse_train_dict, _ = inverse_dict(option_num, train_ids, train_rqid)
inverse_val_dict, _ = inverse_dict(option_num, val_ids, val_rqid)

with open(args.charge_id_path, "r") as f:
    charge_num = len(json.load(f))

train_charges = invert_charges(args.charge_case_path, train_ids)
val_charges = invert_charges(args.charge_case_path, val_ids)

white_data = np.load(args.raw_emb)

model = MultiSliceGAT(
    in_dims, hidden_dim, out_dim, option_num, charge_num, args.charge_th, args.id_th,
    train_emb, train_ids, train_charges, inverse_train_dict,
    val_emb, val_ids, val_charges, inverse_val_dict,
    train_white=white_data[train_ids], val_white=white_data[val_ids],
    charge_emb=charge_emb, id_emb=id_emb, mer_type=mer_type, heads=heads, device=device
).to(device)

train_pos_dict = load_dict(args.train_pos_path)
train_neg_dict = load_dict(args.train_neg_path)
val_pos_dict = load_dict(args.val_pos_path)
val_neg_dict = load_dict(args.val_neg_path)

trainer = GATTrainer(
    model=model,
    epochs=epochs,
    eval_step=25,
    save_step=25,
    device=device,
    lr=args.lr,
    train_pos_dict=train_pos_dict,
    train_neg_dict=train_neg_dict,
    train_ids=train_ids,
    val_pos_dict=val_pos_dict,
    val_neg_dict=val_neg_dict,
    val_ids=only_val_ids,
    args = args
)

trainer.train()