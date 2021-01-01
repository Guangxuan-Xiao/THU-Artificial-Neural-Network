mkdir -p ../logs/
mkdir -p ../plots/
mkdir -p ../outputs/
CELLS="GRU"
LAYERS="2"
UNITS="300"
for LAYER in $LAYERS; do
    for UNIT in $UNITS; do
        for CELL in $CELLS; do
            NAME=${CELL}_l${LAYER}_${UNIT}_final
            # CUDA_VISIBLE_DEVICES=5 python -u main.py \
            #     --name ${NAME} \
            #     --test ${NAME} \
            #     --decode_strategy top-p \
            #     --max_probability 0.8 \
            #     --temperature 0.8 \
            #     --learning_rate 1e-3 \
            #     --units ${UNIT} \
            #     --cell $CELL \
            #     --batch_size 128 \
            #     --layers $LAYER \
            #     --num_epochs 30 \
            #     --residual \
            #     --layer_norm |
            #     tee ../logs/${NAME}_test_t08p08.log

            # CUDA_VISIBLE_DEVICES=5 python -u main.py \
            #     --name ${NAME} \
            #     --test ${NAME} \
            #     --decode_strategy top-p \
            #     --max_probability 0.8 \
            #     --temperature 1 \
            #     --learning_rate 1e-3 \
            #     --units ${UNIT} \
            #     --cell $CELL \
            #     --batch_size 128 \
            #     --layers $LAYER \
            #     --num_epochs 30 \
            #     --residual \
            #     --layer_norm |
            #     tee ../logs/${NAME}_test_t10p08.log

            CUDA_VISIBLE_DEVICES=5 python -u main.py \
                --name ${NAME} \
                --test ${NAME} \
                --decode_strategy random \
                --temperature 0.8 \
                --learning_rate 1e-3 \
                --units ${UNIT} \
                --cell $CELL \
                --batch_size 128 \
                --layers $LAYER \
                --num_epochs 30 \
                --residual \
                --layer_norm |
                tee ../logs/${NAME}_test_t08.log

            # CUDA_VISIBLE_DEVICES=5 python -u main.py \
            #     --name ${NAME} \
            #     --test ${NAME} \
            #     --decode_strategy random \
            #     --temperature 1 \
            #     --learning_rate 1e-3 \
            #     --units ${UNIT} \
            #     --cell $CELL \
            #     --batch_size 128 \
            #     --layers $LAYER \
            #     --num_epochs 30 \
            #     --residual \
            #     --layer_norm |
            #     tee ../logs/${NAME}_test_t10.log
        done
    done
done
