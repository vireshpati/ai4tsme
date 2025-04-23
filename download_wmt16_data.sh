#!/bin/bash
# Download and prepare WMT16 English-German dataset for machine translation
# This script does three things:
# 1. Downloads the pretrained WMT16 EN-DE transformer model
# 2. Downloads the preprocessed test data for evaluation
# 3. Preprocesses raw training data using a local Perl installation

set -e

# Use local Perl installation if available
if [ -d "$HOME/perl/bin" ]; then
    echo "Using local Perl installation..."
    export PATH="$HOME/perl/bin:$PATH"
    export PERL5LIB="$HOME/perl/lib:$PERL5LIB"
else
    echo "WARNING: Local Perl installation not found."
    echo "If you encounter Perl module issues, run install_local_perl.sh first."
fi

# Create directories
mkdir -p data-wmt16
cd data-wmt16

echo "Downloading WMT16 English-German data..."

# 1. Download pretrained model for evaluation
if [ ! -d "wmt16.en-de.joined-dict.transformer" ]; then
    echo "Downloading pretrained WMT16 EN-DE transformer model..."
    wget https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2
    tar -xjf wmt16.en-de.joined-dict.transformer.tar.bz2
fi

# 2. Download preprocessed test data
if [ ! -d "wmt16.en-de.joined-dict.newstest2014" ]; then
    echo "Downloading preprocessed WMT16 EN-DE test data..."
    wget https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2
    tar -xjf wmt16.en-de.joined-dict.newstest2014.tar.bz2
fi

cd ..  # Return to main directory

# 3. Set up local Perl environment to fix permission issues
echo "Setting up local Perl environment..."
# Create local Perl directory
mkdir -p $HOME/perl5/lib
# Set PERL5LIB to use local directory
export PERL5LIB=$HOME/perl5/lib

# Install required Perl modules locally - only if needed
if command -v cpanm &> /dev/null; then
    echo "cpanm already installed"
else
    echo "Installing cpanm..."
    curl -L https://cpanmin.us | perl - --self-contained --local-lib=$HOME/perl5 App::cpanminus
fi

# Add Perl bin directory to PATH
export PATH="$HOME/perl5/bin:$PATH"

# Install warnings module locally if it doesn't exist
if [ ! -d "$HOME/perl5/lib/perl5/warnings" ]; then
    echo "Installing required Perl modules locally..."
    $HOME/perl5/bin/cpanm --local-lib=$HOME/perl5 warnings
fi

# Install dependencies
echo "Checking for required dependencies..."
pip install --quiet sacrebleu subword-nmt

# Define output directories
RAW_DATA="data-wmt16/wmt16ende"
BINARY_DATA="data-bin/wmt16ende_bpe32k"

if [ ! -d "$RAW_DATA" ]; then
    echo "Downloading and preprocessing WMT16 EN-DE training data..."
    
    # Set environment for the prepare script
    export OUTPUT_DIR="wmt16ende"
    
    # Run prepare-wmt16en2de.sh with modified environment
    echo "Running prepare-wmt16en2de.sh with local Perl environment..."
    bash prepare-wmt16en2de.sh
    
    # Move the output to our data directory if needed
    if [ -d "wmt16ende" ] && [ ! -d "$RAW_DATA" ]; then
        mkdir -p "$RAW_DATA"
        cp -r wmt16ende/* "$RAW_DATA/"
    fi
fi

# 4. Binarize the data for fairseq
if [ ! -d "$BINARY_DATA" ] && [ -d "$RAW_DATA" ]; then
    echo "Binarizing the data for fairseq..."
    
    # Create the binary data directory
    mkdir -p $BINARY_DATA
    
    # Run fairseq preprocessing
    fairseq-preprocess \
        --source-lang en --target-lang de \
        --trainpref $RAW_DATA/train.tok.clean.bpe.32000 \
        --validpref $RAW_DATA/newstest2013.tok.bpe.32000 \
        --testpref $RAW_DATA/newstest2014.tok.bpe.32000 \
        --destdir $BINARY_DATA \
        --nwordssrc 32768 --nwordstgt 32768 \
        --joined-dictionary \
        --workers 8
fi

echo "WMT16 data and model preparation complete."
echo "You can now:"
echo "1. Evaluate the pretrained model: bash evaluate_wmt16.sh pretrained"
echo "2. Train your own model: bash train_wmt16.sh"