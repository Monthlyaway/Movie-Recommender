# Command Line Interface

This document describes how to use the Movie Recommender System via the command line interface.

## Basic Usage

```bash
python src/main.py [options]
```

## Available Commands

### Get Recommendations

```bash
python src/main.py --recommend --movie "Movie Title" --method [method_name] --count [number]
```

Parameters:
- `--movie`: Movie title to get recommendations for (in quotes)
- `--method`: Recommendation method to use (default: "content")
- `--count`: Number of recommendations to return (default: 10)

Available methods:
- `content`: Content-based recommendations using plot similarity
- `collab`: Collaborative filtering recommendations
- `assoc`: Association rule-based recommendations
- `rating`: Simple weighted rating recommendations
- `hybrid`: Combined approach using multiple methods

### Load and Preprocess Data

```bash
python src/main.py --preprocess [options]
```

Parameters:
- `--input`: Path to input data file (default: "data/movies_metadata.csv")
- `--output`: Path to save processed data (default: "data/processed_metadata.csv")
- `--ner`: Apply Named Entity Recognition to filter out person names (optional)

### Compare Methods

```bash
python src/main.py --compare --movie "Movie Title" --methods [method1] [method2] ...
```

Compare different recommendation methods for the same movie.

### Example Usage

```bash
# Get 5 plot-based recommendations for "The Matrix"
python src/main.py --recommend --movie "The Matrix" --method content --count 5

# Get association rule recommendations for "Toy Story"
python src/main.py --recommend --movie "Toy Story" --method assoc

# Compare all methods for "Jurassic Park"
python src/main.py --compare --movie "Jurassic Park" --methods content collab assoc rating hybrid

# Preprocess data with NER
python src/main.py --preprocess --ner
```

## Advanced Options

### Tuning Parameters

You can tune specific parameters for each method:

```bash
python src/main.py --recommend --movie "Movie Title" --method [method_name] --params key1=value1 key2=value2
```

Examples of parameters:
- For content-based: `max_features=5000 min_df=2`
- For collaborative filtering: `factors=100 iterations=20`
- For association rules: `min_support=0.1 min_confidence=0.5` 