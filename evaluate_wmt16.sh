#!/bin/bash
# Evaluate a transformer model on WMT16 English-German newstest2014

set -e

# Check if arguments are provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [beam_size] [lenpen]"
    echo "  checkpoint_path: Path to model checkpoint (or 'pretrained' to use downloaded model)"
    echo "  beam_size: Beam search size (default: 5)"
    echo "  lenpen: Length penalty (default: 0.6)"
    exit 1
fi

# Variables
CHECKPOINT=$1
BEAM_SIZE=${2:-5}
LENPEN=${3:-0.6}
RESULTS_DIR="results/wmt16"
GEN_FILE="$RESULTS_DIR/gen_output.txt"

# Create results directory
mkdir -p $RESULTS_DIR

# Set checkpoint path and data path based on input
if [ "$CHECKPOINT" = "pretrained" ]; then
    CHECKPOINT="data-wmt16/wmt16.en-de.joined-dict.transformer/model.pt"
    DATA_BIN="data-wmt16/wmt16.en-de.joined-dict.newstest2014"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Pretrained model not found. Run download_wmt16_data.sh first."
        exit 1
    fi
    echo "Using pretrained WMT16 model and test data for evaluation"
else
    # For trained models, use our binarized test data
    DATA_BIN="data-bin/wmt16ende_bpe32k"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Checkpoint file not found: $CHECKPOINT"
        exit 1
    fi
    echo "Using provided checkpoint for evaluation: $CHECKPOINT"
    echo "Using binarized test data from preprocessing"
fi

# Check if data exists
if [ ! -d "$DATA_BIN" ]; then
    echo "Test data not found at $DATA_BIN. Run download_wmt16_data.sh first."
    exit 1
fi

echo "Generating translations with beam size $BEAM_SIZE and length penalty $LENPEN..."

# Generate translations
fairseq-generate $DATA_BIN \
    --path $CHECKPOINT \
    --beam $BEAM_SIZE --lenpen $LENPEN \
    --batch-size 128 \
    --remove-bpe \
    > $GEN_FILE

# Extract BLEU score
cat $GEN_FILE | grep -P "^D-" | sort -V | cut -f3- > $RESULTS_DIR/gen.out
cat $GEN_FILE | grep -P "^T-" | sort -V | cut -f2- > $RESULTS_DIR/ref.out

# Use sacrebleu for evaluation
echo "Computing BLEU score with sacrebleu..."
cat $RESULTS_DIR/gen.out | sacrebleu $RESULTS_DIR/ref.out

echo "Evaluation complete. Results saved to $RESULTS_DIR"