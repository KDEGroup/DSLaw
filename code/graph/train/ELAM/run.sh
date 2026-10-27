cuda=$1
if [ -z "$1" ]; then
    echo "Error: Device number must be provided as the first argument."
    exit 1
fi

#!/bin/bash

# 执行 Python 主程序
python ../../main.py \
  --cuda $cuda \
  --epochs 1000 \
  --lr 1e-3 \
  --in_dim 512 512 512 512 512 512 \
  --option_num 64 64 64 64 64 64 \
  --hidden_dim 16 \
  --out_dim 64 \
  --heads 4\
  --charge_th 1500\
  --id_th 1500\
  \
  --mer_type cat \
  --emb_type residual \
  --data_ids "../../ELAM_data_ids.npy" \
  --residual_emb "../../ELAM_data_quant.npy" \
  --raw_emb "ELAM/ELAM_white_emb.npy" \
  --charge_emb "../../ELAM_charge_quant.npy" \
  --id_emb "../../ELAM_codebook.npy" \
  \
  --train_id_path "../../../dataset/ELAM/train/train_candidate_ids.json" \
  --val_id_path "../../../dataset/ELAM/test/test_candidate_ids.json" \
  --charge_id_path "../../../dataset/ELAM/charges/ELAM_charge_id.json" \
  --charge_case_path "../../../dataset/ELAM/charges/ELAM_charge_case.json" \
  --train_pos_path "../../../dataset/ELAM/train/train_relative.json" \
  --train_neg_path "../../../dataset/ELAM/train/train_hard_neg.json" \
  --val_pos_path "../../../dataset/ELAM/test/test_relative.json" \
  --val_neg_path "../../../dataset/ELAM/test/test_hard_neg.json"

  # --residual_emb "../../ELAM_data_quant.npy" \
