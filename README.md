# Movie Recommender

A command-line movie recommendation system offering multiple recommendation strategies with a user-friendly interface.

## Overview

This project implements a flexible movie recommendation system with different recommendation approaches:

1. **Plot-Based Recommender**: Recommends movies similar to a given movie based on plot descriptions using TF-IDF and cosine similarity
2. **Simple Recommender**: Ranks movies by a weighted rating formula balancing average ratings and popularity (vote count)
3. **Name-Removed Variant**: Enhanced text processing that removes person names from movie plots using Named Entity Recognition

## Project Structure

```
Movie-Recommender/
├── dataset/                 # Dataset files (not included in repo)
│   ├── movies_metadata.csv  # Main movie metadata
│   ├── links_small.csv      # Movie ID mappings
│   └── ...                  # Other dataset files
├── src/                     # Source code
│   ├── __init__.py          # Package initialization
│   ├── main.py              # Main entry point
│   ├── data_loader.py       # Data loading utilities
│   ├── preprocess_metadata.py # NER preprocessing script
│   ├── example_comparison.py  # Comparison utility
│   ├── recommenders/        # Recommendation algorithms
│   │   ├── __init__.py
│   │   ├── plot_recommender.py   # Content-based recommendations
│   │   └── simple_recommender.py # Weighted rating recommendations
│   ├── ui/                  # User interface
│   │   ├── __init__.py
│   │   └── cli.py           # Command-line interface
│   └── utils/               # Utility functions
├── run_preprocessing.sh     # Script to run NER preprocessing
├── requirements.txt         # Required packages
├── SIMPLE_RECOMMENDER_README.md  # Details on simple recommender
├── README_NER_PREPROCESSING.md   # Details on name entity removal
└── README.md                # This file
```

## Requirements

The following packages are required to run the Movie Recommender:

```bash
# Install using pip
pip install -r requirements.txt

# Download the required spaCy model
python -m spacy download en_core_web_sm
```

Alternatively, you can install the packages individually:

```bash
pip install pandas numpy scikit-learn spacy rich tqdm colorama tabulate
python -m spacy download en_core_web_sm
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Movie-Recommender.git
cd Movie-Recommender
```

2. Install the required packages:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Download the MovieLens dataset from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) and extract it to the `dataset` directory.

## Usage

### Basic Usage

Run the main script with default settings (plot-based recommender):

```bash
python src/main.py
```

### Simple Recommender

Use the IMDB weighted rating formula to find top-rated movies:

```bash
python src/main.py --recommender simple
```

Optional parameters:
- `--percentile 0.85` - Set the vote count percentile (default: 0.90)

### Name Removal Preprocessing

To preprocess the dataset by removing person names from plots (improves content recommendations):

```bash
# Make the script executable
chmod +x run_preprocessing.sh

# Run the preprocessing
./run_preprocessing.sh
```

Then use the preprocessed data:

```bash
python src/main.py --use-name-removed
```

### Combining Features

You can combine different features, for example:

```bash
# Use simple recommender with name-removed dataset
python src/main.py --recommender simple --use-name-removed

# Use plot recommender with name-removed dataset and custom percentile
python src/main.py --use-name-removed --percentile 0.85
```

### Comparing Recommendations

To compare recommendations with and without named entity removal:

```bash
python src/example_comparison.py
```

## Detailed Documentation

For detailed information about specific components, refer to:

- [Simple Recommender Documentation](SIMPLE_RECOMMENDER_README.md) - Details about the weighted rating approach
- [NER Preprocessing Documentation](README_NER_PREPROCESSING.md) - Information about named entity removal

## Features

- **Multiple Recommendation Strategies**: Choose between content-based and weighted-rating approaches
- **Interactive CLI**: User-friendly command-line interface with rich formatting
- **Named Entity Recognition**: Advanced text preprocessing to improve content-based recommendations
- **Comprehensive Error Handling**: Graceful handling of missing data and edge cases
- **Detailed Documentation**: In-depth explanation of algorithms and implementation

## Dataset

This project uses the MovieLens dataset, which includes:
- Movie metadata (titles, release dates, genres, etc.)
- Plot overviews
- Movie ratings
- Links to external databases (IMDb, TMDB)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
