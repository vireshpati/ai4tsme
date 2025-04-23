#!/bin/bash
# Script to evaluate a transformer model on SC10 (Speech Commands) dataset

# Set up environment variables with relative paths
DATA="./speech_commands_data/speech_commands"
SAVE="./checkpoints/sc10"
FAIRSEQ_PATH="./fairseq"
CHECKPOINT="${SAVE}/checkpoint_best.pt"

# Define model configuration
MODEL="transformer_sc_raw_small"

# Validate the model on the test set
python -u ${FAIRSEQ_PATH}/fairseq_cli/validate.py ${DATA} \
    --path ${CHECKPOINT} \
    --valid-subset test \
    --task speech_commands \
    --criterion lra_cross_entropy \
    --batch-size 5 --log-format simple \
    --fp16 \
    --quiet