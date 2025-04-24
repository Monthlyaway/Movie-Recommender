# Movie Recommender System

A comprehensive movie recommendation system that implements multiple recommendation algorithms.

## Installation

```bash
pip install -r requirements.txt
```

## Documentation

### Recommender Types

The system includes several recommendation approaches:

- [Weighted Rating](docs/recommenders/weighted_rating.md) - Simple popularity-based recommendations
- [Content-Based Recommender](docs/recommenders/content_based.md) - Uses TF-IDF and cosine similarity on movie plots
- [Association Rules](docs/recommenders/association_rules.md) - Pattern discovery using frequent itemsets

### Command Line Interface

For information on how to use the command-line interface, see the [CLI documentation](docs/cli.md).

## Project Structure

- `src/` - Source code for the recommender system
  - `recommenders/` - Implementation of various recommendation algorithms
  - `utils/` - Utility functions
  - `ui/` - User interface components
  - `data_loader.py` - Functions for loading and preprocessing data
  - `main.py` - Entry point for the application

