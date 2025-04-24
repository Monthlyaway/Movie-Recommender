# Simple Movie Recommender

This document explains how to use the Simple Movie Recommender, which ranks movies based on the IMDB Weighted Rating Formula.

## What is the Simple Movie Recommender?

The Simple Movie Recommender uses a balanced approach to rank movies by considering both their average ratings and the number of votes they received. This addresses the issue where movies with high ratings but few votes might be unreliably ranked compared to movies with slightly lower ratings but many votes.

高投票电影（v ≫ m）：评分更依赖自身（如v=1000, m=100时，权重90%来自电影自身评分）

低投票电影（v ≪ m）：评分向全局均值C收缩，避免少数用户的极端评分影响排名

### The IMDB Weighted Rating Formula

The formula used is:

```
Score = (v / (v + m)) * R + (m / (v + m)) * C
```

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

Once running, the Simple Movie Recommender displays a simple interface:

1. Choose "View top rated movies" to see the top-rated movies according to the weighted formula
2. Enter the number of movies to display (default is 10)
3. View the results in a formatted table

## Examples

```bash
# Basic usage with default settings (90th percentile)
python src/main.py --recommender simple

# View results with a more inclusive threshold (80th percentile)
python src/main.py --recommender simple --percentile 0.80

# Use with the name-removed dataset (if available)
python src/main.py --recommender simple --use-name-removed
```

## How It Differs from the Plot Recommender

The main differences between the Simple Movie Recommender and the Plot Recommender are:

1. **Recommendation Approach**:
   - Simple Recommender: Shows the same top movies to all users based on ratings and popularity
   - Plot Recommender: Recommends movies similar to a specific movie based on plot content

2. **User Input**:
   - Simple Recommender: No specific movie input required
   - Plot Recommender: Requires a specific movie title as input

3. **Use Case**:
   - Simple Recommender: Best for discovering generally well-received movies
   - Plot Recommender: Best for finding movies similar to ones you already know and like

## Technical Details

The implementation:
- Filters movies based on vote count to ensure statistical reliability
- Pre-calculates scores for all qualified movies
- Returns results in the same format as other recommenders (title, IMDB ID)
- Handles edge cases like missing data
- Provides diagnostics via the `get_details()` method

## Future Improvements

Planned enhancements for the Simple Recommender:
- Filtering by genre, release year, etc.
- Adjusting the vote count threshold at runtime
- Including more movie metadata in the results
- Visualizations of the rating distribution 