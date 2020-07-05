#!/bin/bash

sparse_meta_learning(){
python -u run_miniimagenet.py --inner-batch 10 \
--inner-iters 8 \
--meta-step 1 \
--meta-batch 5 \
--meta-iters 100000 \
--eval-batch 5 \
--eval-iters 50 \
--learning-rate $9 \
--meta-step-final 0 \
--train-shots 15 \
--gpu-id $1 \
--shots $2 \
--checkpoint $3 \
--num_filters $4 \
--compress-rate $5 \
--sparse-iter $6 \
--sparse-interval $7 \
--sparse-mode $8 \
--init-rate $10 \
--img-size $11 \
--ratio $12 \
--DATA_DIR $13 \
--DATASET tieredimagenet
--transductive
}

run_exp(){
(sparse_meta_learning 0 1 ./IHT_TieredExperiments/IHT_rate0.3_64_m51/Conv_ratio0.75_sparse20K_interva20K_0.001 64 0.3 20000 20000 IHT 0.001 0.1 84 0.75, ./data/tiered-imagenet/)
}

run_exp