#!/bin/bash
# Download and prepare Speech Commands 10 dataset

set -e

# Create data directory
mkdir -p data-sc10
cd data-sc10

echo "Downloading Speech Commands dataset..."

# Download dataset if not already downloaded
if [ ! -f "speech_commands_v0.02.tar.gz" ]; then
    echo "Downloading Speech Commands v0.02 dataset..."
    curl -O http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
    echo "Extracting dataset..."
    tar -xzf speech_commands_v0.02.tar.gz
fi

# Download data2vec pretrained model for finetuning
if [ ! -d "data2vec_model" ]; then
    mkdir -p data2vec_model
    echo "Downloading pretrained data2vec audio model..."
    curl -o data2vec_model/data2vec_audio_base.pt https://dl.fbaipublicfiles.com/fairseq/data2vec/audio_base_ls.pt
fi

# Create label file for SC10 (10 command classes)
cat > sc10_labels.txt << EOL
yes
no
up
down
left
right
on
off
stop
go
EOL

echo "Creating data manifest..."

# Create Python script to generate manifest files
cat > create_manifest.py << EOL
import os
import json
import glob
import random
from pathlib import Path

# Set paths
data_root = os.path.abspath(".")
audio_path = os.path.join(data_root, "_background_noise_")
sc10_labels = [l.strip() for l in open("sc10_labels.txt").readlines()]

# Create manifest directory
os.makedirs("manifests", exist_ok=True)

# Function to create manifest
def create_manifest(split):
    manifest_path = os.path.join(data_root, "manifests", f"{split}.tsv")
    
    with open(manifest_path, "w") as f:
        # Write header
        f.write("id\taudio\tn_frames\tlabel\n")
        
        # Process each label
        for label in sc10_labels:
            label_dir = os.path.join(data_root, label)
            files = glob.glob(os.path.join(label_dir, "*.wav"))
            
            # Determine which files to include in this split
            random.seed(42)  # For reproducibility
            random.shuffle(files)
            
            if split == "train":
                split_files = files[:int(0.7*len(files))]
            elif split == "valid":
                split_files = files[int(0.7*len(files)):int(0.85*len(files))]
            else:  # test
                split_files = files[int(0.85*len(files)):]
            
            # Add files to manifest
            for file_path in split_files:
                file_id = os.path.basename(file_path).replace(".wav", "")
                # Using dummy n_frames value of 16000 (1 second at 16kHz)
                f.write(f"{file_id}\t{os.path.abspath(file_path)}\t16000\t{label}\n")
    
    print(f"Created {split} manifest at {manifest_path}")

# Create manifests for each split
create_manifest("train")
create_manifest("valid")
create_manifest("test")

# Create dictionary file
with open("manifests/dict.txt", "w") as f:
    for i, label in enumerate(sc10_labels):
        f.write(f"{label} {i+1}\n")
    
print("Created dictionary at manifests/dict.txt")
EOL

# Run the manifest creation script
python create_manifest.py

echo "Speech Commands 10 dataset preparation complete."
echo "Manifests generated in data-sc10/manifests/"