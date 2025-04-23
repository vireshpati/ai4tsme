#!/bin/bash
# Download preprocessed Speech Commands dataset from MEGA and set up for fairseq

set -e

# Create data directory
mkdir -p speech_commands_data
cd speech_commands_data

echo "Downloading preprocessed Speech Commands dataset..."

# Download dataset if not already downloaded
if [ ! -f "speech_commands.zip" ]; then
    echo "Downloading preprocessed Speech Commands dataset..."
    curl -O https://dl.fbaipublicfiles.com/mega/data/speech_commands.zip
    echo "Extracting dataset..."
    unzip -q speech_commands.zip
fi

echo "Setting up fairseq scaffolding for SC10 classification..."

# Create SC10 labels dictionary file at correct location
cat > speech_commands/dict.labels.txt << EOL
yes 1
no 2
up 3
down 4
left 5
right 6
on 7
off 8
stop 9
go 10
EOL

# Copy this to the standard dictionary location 
cp speech_commands/dict.labels.txt speech_commands/dict.txt

# Create labels file needed by fairseq
echo "labels" > speech_commands/labels

# Make sure all necessary splits exist
for split in train valid test; do
    touch speech_commands/${split}.${split}
done

cd ..  # Return to the original directory

echo "Preprocessed Speech Commands dataset ready at speech_commands_data/speech_commands/"
echo "All fairseq-required files have been set up for training"