#!/bin/bash

# Script to run the preprocessing for removing person names from movie overviews
# This script handles environment activation and package installation with GPU support

echo "Starting movie metadata preprocessing with GPU support..."

# Activate conda environment
echo "Activating conda environment 'cu126'..."
conda activate cu126


# Download the spaCy model if not already installed
echo "Checking spaCy model..."
python -m spacy download en_core_web_sm

# Run the preprocessing script with GPU support
echo "Running preprocessing script with GPU acceleration..."
python /home/castle/Codes/Movie-Recommender/src/preprocess_metadata.py --gpu

echo "Preprocessing complete! The output file is:"
echo "/home/castle/Codes/Movie-Recommender/dataset/movies_metadata_name_removed.csv" 