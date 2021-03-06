mkdir -p ../logs/
mkdir -p ../plots/
mkdir -p ../outputs/
CELLS="GRU LSTM RNN"
LAYERS="2 1"
UNITS="128 64"
for UNIT in $UNITS; do
    for CELL in $CELLS; do
        for LAYER in $LAYERS; do
            NAME=${CELL}_l${LAYER}_${UNIT}
            rm -rf ../runs/$NAME
            CUDA_VISIBLE_DEVICES=7 python -u main.py \
                --name ${NAME} \
                --decode_strategy top-p \
                --max_probability 0.8 \
                --temperature 0.8 \
                --learning_rate 1e-3 \
                --units ${UNIT} \
                --cell $CELL \
                --batch_size 64 \
                --layers $LAYER \
                --num_epochs 30 |
                tee ../logs/${NAME}_train.log

            CUDA_VISIBLE_DEVICES=7 python -u main.py \
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
                --num_epochs 30 |
                tee ../logs/${NAME}_test.log
        done
    done
done
