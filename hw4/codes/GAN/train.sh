CUDA_VISIBLE_DEVICES=2 python -u main.py \
    --do_train \
    --latent_dim 200 \
    --generator_hidden_dim 100 \
    --discriminator_hidden_dim 16 \
    --learning_rate 1e-3 \
    | tee log.log


