#!/bin/bash

sparse_meta_learning(){
python -u run_miniimagenet.py --inner-batch 10 \
--inner-iters 8 \
--meta-step 1 \
--meta-batch 5 \
--meta-iters 100000 \
--eval-batch 5 \
--eval-iters 50 \
--meta-step-final 0 \
--train-shots 15 \
--gpu-id $1 \
--shots $2 \
--checkpoint $3 \
--num_filters $4 \
--compress-rate $5 \
--sparse-iter $6 \
--retrain-iter $7 \
--sparse-mode $8 \
--learning-rate $9 \
--img-size $10 \
--DATASET $11 \
--transductive
}

run_exp(){
(sparse_meta_learning 0 1 ./DSD_Experiments/DSD_64_m51/Conv_rate0.3_sparse_30K_90K 64 0.3 30000 90000 DSD 0.001 84, miniimagenet)
}

run_exp