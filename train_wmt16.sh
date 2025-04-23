  #!/bin/bash
  # train_wmt16.sh
  # Train a transformer model on WMT16 English-German dataset

  set -e

  # Make sure data is downloaded and processed
  BINARY_DATA="data-bin/wmt16ende_bpe32k"
  if [ ! -d "$BINARY_DATA" ]; then
      echo "WMT16 processed data not found. Run binarize_wmt16.sh first."
      exit 1
  fi

  # Variables
  DATA_BIN="data-bin/wmt16ende_bpe32k"
  CHECKPOINT_DIR="checkpoints/wmt16_en_de"
  NUM_GPUS=${1:-1}  # Number of GPUs (default: 1)
  UPDATE_FREQ=$((32 / $NUM_GPUS))  # Adjust update frequency based on GPU count

  # Create checkpoint directory
  mkdir -p $CHECKPOINT_DIR

  echo "Training WMT16 EN-DE transformer with $NUM_GPUS GPUs..."

  # Train transformer model
  fairseq-train $DATA_BIN \
      --arch transformer_vaswani_wmt_en_de_big \
      --share-decoder-input-output-embed \
      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
      --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
      --dropout 0.3 --weight-decay 0.0001 \
      --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
      --max-tokens 4096 \
      --save-dir $CHECKPOINT_DIR \
      --update-freq $UPDATE_FREQ \
      --max-epoch 20 \
      --save-interval 1 \
      --save-interval-updates 1000 \
      --keep-interval-updates 5 \
      --log-interval 100 \
      --patience 10 \
      --fp16 \
      --seed 42 \
      --log-format json \
      --tensorboard-logdir $CHECKPOINT_DIR/tensorboard \
      --eval-bleu \
      --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
      --eval-bleu-detok moses \
      --eval-bleu-remove-bpe \
      --eval-bleu-print-samples \
      --best-checkpoint-metric bleu \
      --maximize-best-checkpoint-metric

  echo "Training complete. Checkpoint saved to $CHECKPOINT_DIR"