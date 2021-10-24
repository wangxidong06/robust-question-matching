#!/usr/bin/bash
$unset CUDA_VISIBLE_DEVICES
python train.py \
       --train_set /home/wangxidong/robust-question-matching/data/train.txt \
       --dev_set /home/wangxidong/robust-question-matching/data/dev.txt \
       --save_dir /home/wangxidong/robust-question-matching/ERNIE-Gram/checkpoints/checkpoint-10-24_14:06 \
       --device gpu \
       --eval_step 100 \
       --train_batch_size 32 \
       --learning_rate 2E-5 \
       --rdrop_coef 0.0
