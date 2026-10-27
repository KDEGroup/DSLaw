cuda=$1
if [ -z "$1" ]; then
    echo "Error: Device number must be provided as the first argument."
    exit 1
fi

python ../main.py \
    --lr 1e-3 \
    --weight_decay 5e-3 \
    --epochs 500 \
    --eval_step 1 \
    --save_step 50 \
    --learner AdamW \
    --batch_size 512 \
    \
    --train_data_path ../../dataset/COLIEE/train/train_candidate_emb.npy \
    --train_ids_path ../../dataset/COLIEE/train/train_candidate_ids.json \
    --train_pos_dict_path ../../dataset/COLIEE/train/train_relative.json \
    --train_neg_dict_path ../../dataset/COLIEE/train/train_hard_neg.json \
    \
    --valid_data_path ../../dataset/COLIEE/test/test_candidate_emb.npy \
    --valid_ids_path ../../dataset/COLIEE/test/test_ids.json \
    --valid_pos_dict_path ../../dataset/COLIEE/test/test_relative.json \
    --valid_neg_dict_path ../../dataset/COLIEE/test/test_hard_neg.json \
    \
    --emb_path COLIEE/COLIEE_white_emb.npy \
    \
    --e_dim 512 \
    --neg_num 50 \
    --quant_loss_weight 1.0 \
    --recon_loss_weight 0.0 \
    --mu 0.5  \
    --num_emb_list 64 64 64 64 \
    --margin_rate 1 2 4 8 \
    --layers 128 \
    --dropout_prob 0.6 \
    --device "cuda:$cuda" \
    --norm_type "ln" \
    --e_dim_list 512 512 512 512 \