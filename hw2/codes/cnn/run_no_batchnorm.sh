MODEL_NAME=cnn_d60_no_bn
CUDA_VISIBLE_DEVICES=0 python -u main.py --learning_rate 1e-3 --drop_rate 0.6 --num_epochs 30 --model_name $MODEL_NAME | tee logs/$MODEL_NAME.log