#!/bin/bash

# Define the URL of the repository to clone
REPO_URL="https://github.com/zeyangsha/De-Fake.git"

# Define the name of the requirements file and the conda environment
REQUIREMENTS_FILE="requirements.txt"
CONDA_ENV_NAME="defake"

# Create a requirements.txt file with the necessary dependencies
cat <<EOF > $REQUIREMENTS_FILE
tqdm
Pillow
scikit-learn
ftfy
regex
git+https://github.com/openai/CLIP.git
fairscale==0.4.4
pycocoevalcap
natsort
timm
transformers==4.15.0
EOF

echo "Cloning the repository..."
echo "----------------------------------------"

# Clone the repository from the defined URL
# git clone $REPO_URL

# Change to the directory of the cloned repository
cd De-Fake || { echo "Failed to change directory to De-Fake"; exit 1; }

echo "Creating and activating the conda environment..."
echo "----------------------------------------"

# Create a new conda environment with the specified name and Python version
conda create --name $CONDA_ENV_NAME python=3.8.5 -y

# Activate the newly created conda environment
source activate $CONDA_ENV_NAME

echo "Installing packages from requirements.txt..."
echo "----------------------------------------"

# Install the Python packages specified in requirements.txt
# pip install -r $REQUIREMENTS_FILE -q

# Uncomment the following lines to download additional files if needed
# echo "Downloading files from Google Drive..."
# gdown https://drive.google.com/uc?id=1qI7x5iodaCFq0S61LKw4wWjql7cYou_4
# gdown https://drive.google.com/uc?id=1SuenxJP10VwArC6zW0SHMUGObMRqQhBD

echo "Processing images..."
echo "----------------------------------------"

# Define paths to the folders containing real and fake images
REAL_IMAGE_FOLDER="/home/shreyas/Desktop/Shreyas/Projects/De_Fake/images/real"  # Update this path as needed
FAKE_IMAGE_FOLDER="/home/shreyas/Desktop/Shreyas/Projects/De_Fake/images/fake"  # Update this path as needed

# Create directories to store metrics for real and fake images
mkdir -p metrics_real
mkdir -p metrics_fake

# Run the 'test.py' script on the real images and store metrics in the 'metrics_real' directory
echo "Running 'test.py' on real images..."
Metric_DIR="real"
python3 test.py --image_folder "$REAL_IMAGE_FOLDER" --metric_dir "$Metric_DIR"

# Run the 'test.py' script on the fake images and store metrics in the 'metrics_fake' directory
echo "Running 'test.py' on fake images..."
Metric_DIR="fake"
python3 test.py --image_folder "$FAKE_IMAGE_FOLDER" --metric_dir "$Metric_DIR"

echo "Calculating and averaging metrics..."

echo "Script execution completed."
echo "----------------------------------------"
