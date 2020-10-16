mkdir -p logs
mkdir -p plots
for drop in {0..10}; do
MODEL_NAME=mlp_d${drop}0
CUDA_VISIBLE_DEVICES=0 python -u main.py --learning_rate 1e-3 --drop_rate 0.$drop --num_epochs 30 --model_name $MODEL_NAME --batch_norm | tee logs/$MODEL_NAME.log
done