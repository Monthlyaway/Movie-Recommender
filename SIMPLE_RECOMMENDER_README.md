# Simple Movie Recommender

This document explains how to use the Simple Movie Recommender, which ranks movies based on the IMDB Weighted Rating Formula.

*For general usage instructions, see the [main README](README.md)*

## What is the Simple Movie Recommender?

The Simple Movie Recommender uses a balanced approach to rank movies by considering both their average ratings and the number of votes they received. This addresses the issue where movies with high ratings but few votes might be unreliably ranked compared to movies with slightly lower ratings but many votes.

### Behavior Examples:

- **High-vote movies** (v ≫ m): Rating relies more on the movie's own average (e.g., when v=1000, m=100, 90% of the weight comes from the movie's own rating)
- **Low-vote movies** (v ≪ m): Rating gravitates toward the global mean C, preventing extreme ratings from a few users from affecting rankings

### The IMDB Weighted Rating Formula

The formula used is:

$$
Score = \frac{v}{v + m} * R + \frac{m}{v + m} * C
$$

Where:
- `v`: Number of votes for the movie
- `m`: Minimum votes required to be listed (set to the 90th percentile by default)
- `R`: Average rating of the movie
- `C`: Mean vote across the whole dataset

This formula balances the movie's specific rating with the dataset's average rating, weighted by the number of votes.

## How to Use

### Command Line Arguments

To use the Simple Movie Recommender, run the main script with the `--recommender` argument:

```bash
python src/main.py --recommender simple
```

Additional options:
- `--percentile`: Set the vote count percentile threshold (default: 0.90)
  ```bash
  python src/main.py --recommender simple --percentile 0.85
  ```

- `--use-name-removed`: Use the version of metadata with person names removed from overviews
  ```bash
  python src/main.py --recommender simple --use-name-removed
  ```

### User Interface

Once running, the Simple Movie Recommender displays a clean interface:

1. Choose "View top rated movies" to see the top-rated movies according to the weighted formula
2. Enter the number of movies to display (default is 10)
3. View the results in a formatted table with IMDb links

## Implementation Details

The implementation is contained in `src/recommenders/simple_recommender.py` and includes:

```python
class SimpleRecommender:
    def __init__(self, vote_count_percentile=0.90):
        # Initialize parameters
        
    def fit(self, metadata_df):
        # Calculate mean vote (C) and minimum votes threshold (m)
        # Filter movies with enough votes
        # Calculate weighted scores
        # Sort movies by score
        
    def recommend(self, movie_title=None, top_n=10):
        # Return top N movies by weighted score
```

## How It Differs from Other Recommenders

The main differences between the Simple Movie Recommender and other recommenders are:

1. **No User Input Required**: Unlike the Plot Recommender which needs a movie title, the Simple Recommender works without any specific movie input.

2. **Universal Recommendations**: All users receive the same recommendations based on objective metrics rather than personalized content similarity.

3. **Pre-Calculation**: The scoring happens during fitting rather than at recommendation time, making recommendations very fast.

## Examples

Using the Simple Recommender with default settings often highlights well-known, critically acclaimed movies like "The Godfather," "The Shawshank Redemption," and others that have both high ratings and substantial vote counts.

## Future Improvements

Planned enhancements for the Simple Recommender:
- Filtering by genre, release year, etc.
- Adjusting the vote count threshold via the UI
- Including more movie metadata in the results
- Visualizations of the rating distribution
- Genre-weighted recommendations

## Related Features

- [Named Entity Removal](README_NER_PREPROCESSING.md): Can be combined with the Simple Recommender, though it has less impact than with the Plot Recommender.
- [Example Comparison Script](src/example_comparison.py): Shows differences between recommendation approaches. 