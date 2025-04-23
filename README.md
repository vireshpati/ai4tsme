# Status 4/22/2025

We have reproduce BLEU 29.5 on WMT pre-trained model. We have completed several epochs of training and seen BLEU consistently increase to > 20.

We have essentially validated train and eval for WMT.

Currently, not enough memory is allocated to be able to validate SC10, so the scripts runs but with OOM on forward pass.

# Transformer Evaluation for WMT16 and SC10

This repository contains code for training and evaluating transformer models on two tasks:
1. **WMT16 English-German Translation**: A neural machine translation task using the standard WMT16 dataset
2. **SC10 (Speech Commands)**: A speech classification task using the Speech Commands dataset

The implementation is based on [fairseq](https://github.com/facebookresearch/fairseq) and includes custom modifications for handling both text and audio data. The approach for SC10 is inspired by the [Mega](https://github.com/facebookresearch/mega) paper's experiments.

## Project Structure

- **fairseq/**: Modified version of the fairseq library with implementations for WMT16 and SC10
- **train_wmt16.sh**: Script to train a transformer model on WMT16 En-De
- **evaluate_wmt16.sh**: Script to evaluate a trained WMT16 model
- **train_sc10.sh**: Script to train a transformer model on SC10
- **evaluate_sc10.sh**: Script to evaluate a trained SC10 model
- **download_wmt16_data.sh**: Script to download and preprocess WMT16 data
- **prepare-wmt16en2de.sh**: Script to prepare WMT16 En-De data
- **download_sc10_data.sh**: Script to download SC10 raw data
- **download_sc10_preprocessed.sh**: Script to download pre-processed (by google) SC10 data
- **binarize_wmt16.sh**: Script to binarize WMT16 data for faster training
- **speech_commands_data/**: Directory for the SC10 dataset
- **wmt16_data/**: Directory for the WMT16 dataset

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/transformer-evaluation.git
cd transformer-evaluation
```

2. Set up a virtual environment:
```bash
conda create -n fairseq_env python=3.9
conda activate fairseq_env
```

3. Install fairseq requirements:
```bash
cd fairseq
pip install -e .
cd ..
```

## Data Preparation

### For WMT16 English-German Translation

1. Download and prepare the WMT16 data:
```bash
bash download_wmt16_data.sh
bash prepare-wmt16en2de-2.sh
```

2. Binarize the data for faster training:
```bash
bash binarize_wmt16.sh
```

### For SC10 Speech Classification

Option 1: Download pre-processed data (recommended):
```bash
bash download_sc10_preprocessed.sh
```

Option 2: Download raw data :
```bash
bash download_sc10_data.sh
```

## Scripts and Usage

### WMT16 Translation Task

1. **download_wmt16_data.sh**:
   - Downloads raw WMT16 English-German parallel corpus
   - Includes training, validation, and test sets
   - Usage: `bash download_wmt16_data.sh`

2. **prepare-wmt16en2de-2.sh**: 
   - Preprocesses WMT16 data (tokenization, cleaning)
   - Applies BPE (Byte Pair Encoding)
   - Builds vocabularies
   - Usage: `bash prepare-wmt16en2de-2.sh`
   - Taken from mega.

3. **binarize_wmt16.sh**:
   - Converts text data to binary format for faster loading
   - Creates vocabulary files
   - Usage: `bash binarize_wmt16.sh`

4. **train_wmt16.sh**:
   - Trains a transformer model on WMT16 data
   - Uses standard fairseq transformer architecture
   - Usage: `bash train_wmt16.sh`
   - Parameters include learning rate, batch size, etc.

5. **evaluate_wmt16.sh**:
   - Evaluates trained model on test set
   - Calculates BLEU score using sacrebleu
   - Usage: `bash evaluate_wmt16.sh`

### SC10 Speech Classification Task

1. **download_sc10_data.sh**:
   - Downloads raw Speech Commands dataset
   - Usage: `bash download_sc10_data.sh`

2. **download_sc10_preprocessed.sh**:
   - Downloads pre-processed Speech Commands data
   - Saves time by using already processed data
   - Usage: `bash download_sc10_preprocessed.sh`

3. **train_sc10.sh**:
   - Trains a transformer model on SC10 data
   - Uses custom transformer implementation for audio
   - Usage: `bash train_sc10.sh`
   - Parameters include model architecture, batch size, etc.

4. **evaluate_sc10.sh**:
   - Evaluates trained model on SC10 test set
   - Reports accuracy metrics
   - Usage: `bash evaluate_sc10.sh`

## Modified Files in fairseq

### For WMT16 Translation Task
The WMT16 task uses the standard fairseq implementation without significant modifications.

### For SC10 Speech Classification Task

1. **fairseq/fairseq/tasks/speech_command.py**:
   - Implements `SpeechCommandsTask` class for handling SC10 data
   - Registers the task with fairseq's task registry
   - Defines data loading and model building for SC10

2. **fairseq/fairseq/models/transformer_sc_raw.py**:
   - Implements `TransformerSCRawModel` for SC10
   - Defines encoder architecture that works with raw audio
   - Includes multiple model sizes (base, small, big)
   - Handles audio data with variable-length sequences

3. **fairseq/fairseq/criterions/lra_cross_entropy.py**:
   - Implements `LRACrossEntropyCriterion` for classification tasks
   - Reports accuracy metrics during training and evaluation

4. **fairseq/fairseq/data/audio/speech_commands_dataset.py**:
   - Implements `SpeechCommandsDataset` for SC10
   - Handles loading and preprocessing of audio data
   - Manages dataset splits (train, valid, test)

5. **fairseq/fairseq/data/__init__.py**:
   - Modified to expose `SpeechCommandsDataset`
   - Includes necessary import statements

## Training

### Training WMT16 Translation

```bash
bash train_wmt16.sh
```
The script includes parameters for:
- Transformer architecture and size
- Optimizer settings (Adam with specific learning rate schedule)
- Token generation and beam search settings
- Checkpointing and logging configurations

### Training SC10 Classification

```bash
bash train_sc10.sh
```

The default configuration uses:
- Model: transformer_sc_raw_small (4 layers, 48-dim embeddings)
- Batch size: 5 with update frequency 4 (effective batch size 20)
- Mixed precision (FP16) for memory efficiency
- Adam optimizer with learning rate 0.01
- 250,000 training updates

## Evaluation

### Evaluating WMT16 Translation

```bash
bash evaluate_wmt16.sh
```

This script:
- Generates translations for the test set
- Computes BLEU score using sacrebleu
- Outputs translation quality metrics

### Evaluating SC10 Classification

```bash
bash evaluate_sc10.sh
```

This script:
- Evaluates the best checkpoint on the test set
- Reports accuracy metrics
- Analyzes model performance on speech classification


## Acknowledgments

- This implementation is based on the [fairseq](https://github.com/facebookresearch/fairseq) library by Facebook AI Research
- The SC10 approach is inspired by the [Mega](https://github.com/facebookresearch/mega) paper's experiments
- WMT16 data processing is adapted from standard fairseq examples
