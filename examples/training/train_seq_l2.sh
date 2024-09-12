# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash
# pre-train on one locale dataset with feature embedding and text embedding

###############################################################################################
### Please modify the following variables according to your device and mission requirements ###
###############################################################################################
LOCAL_ROOT="/home/v-leiyuxuan/working_dir3/UniRec"  # path to UniRec

ALL_DATA_ROOT="/home/v-leiyuxuan/working_dir3/UniRec/data"
wandb_file="/home/v-leiyuxuan/working_dir3/UniRec/examples/training/wandb.yaml"
###############################################################################################


# default parameters for local run
MY_DIR=$LOCAL_ROOT
OUTPUT_ROOT="$LOCAL_ROOT/output"


MODEL_NAME='AttHist' # [AvgHist, AttHist, MF, SVDPlusPlus, GRU, SASRec, ConvFormer, MultiVAE]
DATA_TYPE='SeqRecDataset' #AERecDataset BaseDataset SeqRecDataset
DATASET_NAME="unirec_unorder_movies_v2"
verbose=2
learning_rate=0.001
epochs=100
weight_decay=0 #1e-6
dropout_prob=0
loss_type='softmax' # [bce, bpr, softmax, ccl, fullsoftmax]
distance_type='l2' # [cosine, mlp, dot]
n_sample_neg_train=19  #400
max_seq_len=40
history_mask_mode='unorder'  #'autoregressive'
embedding_size=128

cd $MY_DIR
export PYTHONPATH=$PWD


ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
mkdir -p $ALL_RESULTS_ROOT
### train ###################################
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$ALL_RESULTS_ROOT"/train_l2" \
    --learning_rate=$learning_rate \
    --dropout_prob=$dropout_prob \
    --use_pre_item_emb=0 \
    --loss_type=$loss_type \
    --max_seq_len=$max_seq_len \
    --has_user_bias=0 \
    --has_item_bias=0 \
    --epochs=$epochs  \
    --batch_size=2048 \
    --n_sample_neg_train=$n_sample_neg_train \
    --n_sample_neg_valid=99 \
    --valid_protocol='one_vs_k' \
    --test_protocol='one_vs_all' \
    --grad_clip_value=10.0 \
    --weight_decay=$weight_decay \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq"  \
    --history_mask_mode=$history_mask_mode \
    --metrics="['hit@10;20;50;100', 'ndcg@10;20;50;100','mrr@10;20;50;100']" \
    --key_metric="ndcg@10" \
    --shuffle_train=1 \
    --seed=2024 \
    --early_stop=5 \
    --embedding_size=$embedding_size \
    --num_workers=6 \
    --num_workers_test=0 \
    --verbose=$verbose \
    --neg_by_pop_alpha=0 \
    --distance_type=$distance_type \
    --scheduler_factor=0.1 \
    --tau=1.0 \
    --use_text_emb=0 \
    --text_emb_path=$text_emb_path \
    --text_emb_size=1024 \
    --use_features=0 \
    --features_filepath=$features_filepath  \
    --features_shape='[3489, 99]' \
    --use_wandb=0 \
    --wandb_file=$wandb_file