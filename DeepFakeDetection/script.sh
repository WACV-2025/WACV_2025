#!/bin/bash

# Clone the DeepfakeDetection repository
git clone https://github.com/lira-centre/DeepfakeDetection.git -q
cd DeepfakeDetection

# Define directories
DATA_DIR="DeepfakeDetection/Data"
FAKE_DIR="$DATA_DIR/Fake/class1"
ORIGINAL_DIR="$DATA_DIR/Original/class1"

# Define source image paths
SOURCE_FAKE_IMG="fake.jpg"
SOURCE_ORIGINAL_IMG="Original.jpg"

#---------------------------------------------------------------------#
# Update this part of the script if you already have the folders in the required format.
# Format:
# data
# ├── Fake
# │   └── class1
# │       └── images
# └── Original
#     └── class1
#         └── images

# Define the number of copies
NUM_COPIES=10
# Create directory structure
mkdir -p "$FAKE_DIR"
mkdir -p "$ORIGINAL_DIR"

# Copy and rename images
for i in $(seq 1 $NUM_COPIES); do
    cp "$SOURCE_FAKE_IMG" "$FAKE_DIR/image$i.jpg"
    cp "$SOURCE_ORIGINAL_IMG" "$ORIGINAL_DIR/image$i.jpg"
done
#---------------------------------------------------------------------#

echo "Directory structure created and images copied/renamed successfully."

echo

# Update the Feature_extraction_pretrained.py script to set the correct data_dir path
sed -i "s|data_dir = .*|data_dir = '$DATA_DIR'|" Feature_extraction_pretrained.py

echo "Updated Feature_extraction_pretrained.py with the correct data_dir path."

echo

# Wait for all background jobs to finish (if any were running)
wait

echo "All datasets downloaded and extracted successfully."

echo

# Run the feature extraction and classifier scripts
echo "Running feature extraction using pretrained models..."
python Feature_extraction_pretrained.py

echo

echo "Running xDNN classifier..."
python xDNN_run.py

echo

echo "All tasks completed successfully."

echo
