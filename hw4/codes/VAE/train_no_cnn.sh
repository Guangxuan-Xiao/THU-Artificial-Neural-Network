for latent_dim in 200; do
    CUDA_VISIBLE_DEVICES=1 python -u main.py \
        --do_train \
        --num_training_steps 25000 \
        --latent_dim $latent_dim \
        --no_cnn \
	| tee log/${latent_dim}_nocnn.log
done
