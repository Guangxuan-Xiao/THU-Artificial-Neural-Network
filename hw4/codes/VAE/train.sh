mkdir -p log
rm -rf runs/ results/
for latent_dim in 3 10 100 200; do
    CUDA_VISIBLE_DEVICES=1 python -u main.py \
        --do_train \
        --num_training_steps 25000 \
        --latent_dim $latent_dim \
	| tee log/$latent_dim.log
done
