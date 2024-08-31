# Deepfake Detection

This repository provides code for feature extraction and classification in deepfake detection. Below are the instructions for setting up and running the analysis using custom images.

## Data Format

Ensure your data follows this structure:
```
data
├── Fake
│   └── class1
│       └── images
└── Original
    └── class1
        └── images
```

- **Fake Images**: Place your fake images in `data/Fake/class1/images`.
- **Original Images**: Place your original images in `data/Original/class1/images`.

The provided setup script will create these directories if they do not already exist and copy your images into them.

## Setup and Execution

1. **Clone the Repository**
   ```bash
   git clone https://github.com/lira-centre/DeepfakeDetection.git
   cd DeepfakeDetection
   ```

2. **Prepare the Data**
   Ensure `fake.jpg` and `Original.jpg` are accessible. Then run:
   ```bash
   chmod +x setup_and_run.sh
   ./setup_and_run.sh
   ```

   This script will:
   - Create the directory structure.
   - Copy and rename images to match the required format.
   - Update the data path in the Python scripts.

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Analysis**
   The setup script will automatically execute:
   ```bash
   python Feature_extraction_pretrained.py
   python xDNN_run.py
   ```

## Notes

- The setup script uses the repository code to process your custom images.
- Verify paths and dependencies are correctly configured.
