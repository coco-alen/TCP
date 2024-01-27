CUDA_VISIBLE_DEVICES=2,3 python train.py \
    --id tcp_moblienet0.35_gru_100 \
    --epochs 60 \
    --lr 0.0001 \
    --val_every 1 \
    --batch_size 64 \
    --gpus 2