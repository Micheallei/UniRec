# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash
# infer embeddings for given users or items
# root
###############################################################################################
### Please modify the following variables according to your device and mission requirements ###
###############################################################################################
LOCAL_ROOT="/home/v-leiyuxuan/working_dir3/UniRec"  # path to UniRec

ALL_DATA_ROOT="/home/v-leiyuxuan/working_dir3/UniRec/data"

model_file="/home/v-leiyuxuan/working_dir3/UniRec/output/unirec_unorder_movies_v2/AttHist/train_l2/checkpoint_2024-09-12_120850_21/AttHist.pth"
output_path="/home/v-leiyuxuan/working_dir3/UniRec/output/unirec_unorder_movies_v2/AttHist/train_l2/checkpoint_2024-09-12_120850_21/"
###############################################################################################


# default parameters for local run
MY_DIR=$LOCAL_ROOT

DATASET_NAME="unirec_unorder_movies_v2"

cd $MY_DIR
export PYTHONPATH=$PWD

### infer valid user embedding ###################################
# CUDA_VISIBLE_DEVICES='0,1' torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29400 unirec/main/infer_embedding.py \
CUDA_VISIBLE_DEVICES=0 python unirec/main/infer_embedding.py \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq" \
    --model_file=$model_file \
    --test_batch_size=1024 \
    --output_emb_file=$output_path"valid_user_embedding.txt" \
    --node_type="user" \
    --last_item=0

# ### infer all item embeddings ###################################
# # CUDA_VISIBLE_DEVICES='0,1' torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29400 unirec/main/infer_embedding.py \
# CUDA_VISIBLE_DEVICES=0 python unirec/main/infer_embedding.py \
#     --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
#     --user_history_filename="user_history" \
#     --user_history_file_format="user-item_seq" \
#     --model_file=$model_file \
#     --test_batch_size=40960 \
#     --output_emb_file=$output_path"item_embedding.txt" \
#     --node_type="item"