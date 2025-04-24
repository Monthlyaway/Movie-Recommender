# Movie Recommender NER Preprocessing

This document explains how to run the preprocessing script to remove person names from movie plot overviews using Named Entity Recognition (NER) and how to use the processed data with the recommender system.

*For general usage instructions, see the [main README](README.md)*

## Problem Addressed

The TF-IDF approach for movie recommendations can be skewed by character/actor names in movie plots, as these names tend to be frequent and distinctive words within the plot. This can cause recommendations to be based more on shared character names rather than true plot similarities.

By removing person names from the overviews before calculating TF-IDF scores, we can focus the recommendations on plot concepts and themes rather than specific character names.

## Prerequisites

The preprocessing script requires:
- Python 3.6+
- spaCy library
- spaCy English model (en_core_web_sm)
- pandas
- tqdm (for progress bars)

All of these are included in the `requirements.txt` file in the project root.

## Running the Preprocessing Script

### Option 1: Using the Convenience Script (Recommended)

We've provided a shell script that handles conda environment activation and package installation:

```bash
# Make the script executable if not already
chmod +x /home/castle/Codes/Movie-Recommender/run_preprocessing.sh

# Run the script
/home/castle/Codes/Movie-Recommender/run_preprocessing.sh
```

### Option 2: Manual Execution

If you prefer to run the preprocessing manually:

```bash
# Activate conda environment (if you're using conda)
conda activate movie-recommender

# Install required packages (if not already installed)
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the preprocessing script
python src/preprocess_metadata.py
```

## Script Options

The preprocessing script can be customized with command-line arguments:

```bash
python src/preprocess_metadata.py --help
```

Available options:
- `--input`: Path to the input movies_metadata.csv file
- `--output`: Path to save the processed output file
- `--model`: SpaCy model to use for NER (default: en_core_web_sm)

Example with custom paths:
```bash
python src/preprocess_metadata.py \
  --input /path/to/custom/input.csv \
  --output /path/to/custom/output.csv
```

## Using the Processed Data with the Recommender

The main application has been updated to support using the name-removed version of the metadata:

```bash
# Run with original metadata
python src/main.py

# Run with name-removed metadata
python src/main.py --use-name-removed

# Use with simple recommender
python src/main.py --recommender simple --use-name-removed
```

## How It Works

The preprocessing script:
1. Loads the original movies_metadata.csv file
2. Uses spaCy's Named Entity Recognition to identify PERSON entities in each movie overview
3. Removes these entities from the text
4. Saves the modified data to a new CSV file with the same structure as the original

The modified recommender system works the same way as before, but now operates on plots without person names.

## Comparing Results

To compare recommendations with and without name removal:

```bash
python src/example_comparison.py
```

This will show you side-by-side recommendations for the same input movies using both the original and the name-removed datasets.

## Expected Improvements

By removing person names from the overviews:
- Recommendations should be more focused on plot elements, themes, and concepts
- Results should be less biased by shared character or actor names
- Movies with similar plots but different character names should be better matched
- Overall recommendation quality should improve for plot-based similarity 