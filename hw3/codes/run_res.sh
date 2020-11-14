mkdir -p ../logs/
mkdir -p ../plots/
mkdir -p ../outputs/
CELLS="LSTM GRU"
LAYERS="2"
UNITS="128"
for LAYER in $LAYERS; do
    for UNIT in $UNITS; do
        for CELL in $CELLS; do
            NAME=${CELL}_l${LAYER}_${UNIT}_res
            rm -rf ../runs/$NAME
            CUDA_VISIBLE_DEVICES=5 python -u main.py \
                --name ${NAME} \
                --decode_strategy top-p \
                --max_probability 0.8 \
                --temperature 0.8 \
                --learning_rate 1e-3 \
                --units ${UNIT} \
                --cell $CELL \
                --batch_size 64 \
                --layers $LAYER \
                --num_epochs 30 \
                --residual |
                tee ../logs/${NAME}_train.log

            CUDA_VISIBLE_DEVICES=5 python -u main.py \
                --name ${NAME} \
                --test ${NAME} \
                --decode_strategy top-p \
                --max_probability 0.8 \
                --temperature 0.8 \
                --learning_rate 1e-3 \
                --units ${UNIT} \
                --cell $CELL \
                --batch_size 128 \
                --layers $LAYER \
                --num_epochs 30 \
                --residual |
                tee ../logs/${NAME}_test.log
        done
    done
done
