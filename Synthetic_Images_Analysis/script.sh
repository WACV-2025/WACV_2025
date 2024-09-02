#!/bin/bash

echo "========== Starting the Script =========="
echo " "
echo "========== Installing Dependencies =========="
pip install -r requirements.txt -q

# Clone the repository
echo 
echo "Cloning the SyntheticImagesAnalysis repository..."
git clone https://github.com/grip-unina/SyntheticImagesAnalysis.git && echo "Repository cloned." || echo "Failed to clone the repository."
cd SyntheticImagesAnalysis

echo 
echo "========== Repository Setup Done =========="

# Download weights using gdown
echo "Downloading weights..."
gdown 1OwKKuf0gLxtZFpIbZes4yYsjvKTzZtfh -O DenoiserWeight/ && echo "Weights downloaded." || echo "Failed to download weights."

echo
echo "========== Weights Download Done =========="

# Create the output folder if it doesn't exist
echo "Creating outputs folder..."
mkdir -p outputs && echo "Folder outputs created."

echo
echo "========== Output Folder Setup Done =========="
# Run the image generation script
echo "Running the image generation script..."
python generate_images.py --files_path ../images/ --out_dir outputs --out_name results && echo "Image generation completed." || echo "Image generation failed."

echo
echo "========== Image Generation Done =========="

# Run the spectra generation script
echo "Running the spectra generation script..."
python generate_spectra.py --files_path ../images/ --out_dir outputs --out_name spectral && echo "Spectra generation completed." || echo "Spectra generation failed."

echo
echo "========== Spectra Generation Done =========="
echo
echo "========== Script Completed =========="
