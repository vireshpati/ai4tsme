#!/bin/bash
# Binarize WMT16 English-German data for fairseq

set -e

# Check if data exists
if [ ! -f "data-wmt16/train.tok.bpe.32000.en" ]; then
    echo "WMT16 preprocessed data not found. Run download_wmt16_data.sh first."
    exit 1
fi

# Create output directory
mkdir -p data-bin/wmt16ende_bpe32k

echo "Binarizing WMT16 EN-DE data for fairseq..."

# Run fairseq preprocessing
fairseq-preprocess \
  --source-lang en --target-lang de \
  --trainpref data-wmt16/train.tok.bpe.32000 \
  --validpref data-wmt16/newstest2013.tok.bpe.32000 \
  --testpref data-wmt16/newstest2014.tok.bpe.32000 \
  --destdir data-bin/wmt16ende_bpe32k \
  --workers 8 \
  --joined-dictionary

echo "Binarization complete. Binary data saved to data-bin/wmt16ende_bpe32k"