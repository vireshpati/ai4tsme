#!/bin/bash
# Script to train a transformer model on SC10 (Speech Commands) dataset

# Set up environment variables with relative paths
DATA="./speech_commands_data/speech_commands"
SAVE="./checkpoints/sc10"
FAIRSEQ_PATH="./fairseq"

# Create save directory
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

# Define model configuration
MODEL="transformer_sc_raw_base"

# Train the model
python -u ${FAIRSEQ_PATH}/train.py ${DATA} \
    --seed 1 --ddp-backend c10d --find-unused-parameters \
    --arch ${MODEL} --task speech_commands --encoder-normalize-before \
    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.01 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --dropout 0.0 --attention-dropout 0.0 --activation-dropout 0.0 --weight-decay 0.01 \
    --batch-size 5 --update-freq 4 --sentence-avg --max-update 250000 \
    --lr-scheduler polynomial_decay --warmup-updates 10000 \
    --power 1.0 --total-num-update 250000 --end-learning-rate 0.0 \
    --keep-last-epochs 1 --required-batch-size-multiple 1 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --sentence-class-num 10 --max-positions 16000 --sc-dropped-rate 0.0 \
    --fp16