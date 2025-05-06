"""
IMDB Weighted Rating Formula Utility

This module provides functions for calculating the IMDB weighted rating formula,
which balances average ratings with number of votes to create a more reliable score.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def calculate_weighted_score_params(ratings_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate the global parameters needed for the IMDB weighted score formula:
    C = mean vote across the whole dataset
    m = minimum votes required to be listed (calculated as a percentile)

    Args:
        ratings_df: DataFrame containing 'vote_average' and 'vote_count' columns

    Returns:
        Tuple of (C, m) where:
        - C: mean rating across all movies
        - m: minimum votes threshold
    """
    # Check if dataset has required columns
    required_columns = ['vote_average', 'vote_count']
    missing_columns = [
        col for col in required_columns if col not in ratings_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Calculate the mean vote across all movies
    C = ratings_df['vote_average'].mean()

    # Default to 90th percentile for minimum votes required
    m = ratings_df['vote_count'].quantile(0.90)

    return C, m


def weighted_rating(row: pd.Series, C: float, m: float) -> float:
    """
    Calculate the weighted rating for a movie using the IMDB formula.

    Score = (v/(v+m) * R) + (m/(v+m) * C)
    where:
    - v: number of votes for the movie
    - m: minimum votes required to be listed
    - R: average rating of the movie
    - C: mean vote across the whole dataset

    Args:
        row: Series containing 'vote_count' and 'vote_average' for a movie
        C: Mean vote across all movies
        m: Minimum votes required

    Returns:
        float: Weighted score
    """
    v = row['vote_count']
    R = row['vote_average']
    return (v / (v + m) * R) + (m / (v + m) * C)


def calculate_all_weighted_scores(
    df: pd.DataFrame,
    vote_count_percentile: float = 0.90
) -> Dict[str, Any]:
    """
    Calculate weighted scores for all movies in a DataFrame.

    Args:
        df: DataFrame containing movie data with 'vote_average' and 'vote_count'
        vote_count_percentile: The percentile cutoff for minimum votes (default 90%)

    Returns:
        Dict containing:
        - 'C': global mean rating
        - 'm': minimum votes threshold
        - 'scores': Series of weighted scores indexed by movie index
    """
    # Calculate parameters
    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(vote_count_percentile)

    # Calculate scores for all movies
    scores = df.apply(lambda x: weighted_rating(x, C, m), axis=1)

    return {
        'C': C,
        'm': m,
        'scores': scores
    }


def calculate_normalized_weighted_scores(
    df: pd.DataFrame,
    vote_count_percentile: float = 0.90
) -> Dict[str, Any]:
    """
    Calculate weighted scores for all movies in a DataFrame and normalize them to 0-1 range.

    Args:
        df: DataFrame containing movie data with 'vote_average' and 'vote_count'
        vote_count_percentile: The percentile cutoff for minimum votes (default 90%)

    Returns:
        Dict containing:
        - 'C': global mean rating
        - 'm': minimum votes threshold
        - 'scores': Series of weighted scores indexed by movie index
        - 'normalized_scores': Series of normalized scores (0-1 range) indexed by movie index
    """
    # Get standard weighted scores
    result = calculate_all_weighted_scores(df, vote_count_percentile)
    
    # Normalize the scores to 0-1 range
    scores = result['scores']
    min_score = scores.min()
    max_score = scores.max()
    
    # Avoid division by zero
    score_range = max_score - min_score
    if score_range == 0:
        score_range = 1
    
    # Create normalized scores
    normalized_scores = (scores - min_score) / score_range
    
    # Add to result
    result['normalized_scores'] = normalized_scores
    result['min_score'] = min_score
    result['max_score'] = max_score
    
    return result
