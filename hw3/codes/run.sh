mkdir -p ../logs/
CELL=LSTM
CUDA_VISIBLE_DEVICES=7 python -u main.py \
    --decode_strategy random \
    --max_probability 1 \
    --learning_rate 0.01 \
    --cell $CELL \
    | tee ../logs/${CELL}.log