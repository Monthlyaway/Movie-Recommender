"""
Simple Movie Recommender based on IMDB Weighted Rating formula.

This recommender ranks movies by a weighted rating formula that balances
the average rating and the number of votes, addressing the issue where movies
with high ratings but few votes might not be as reliable as movies with
slightly lower ratings but many votes.
"""

import pandas as pd
import numpy as np

class SimpleRecommender:
    """
    Recommends top movies based on a weighted rating formula that accounts for
    both average rating and number of votes.
    
    Uses the IMDB formula: Score = (v/(v+m) * R) + (m/(v+m) * C)
    where:
    - v: number of votes for the movie
    - m: minimum votes required to be listed
    - R: average rating of the movie
    - C: mean vote across the whole dataset
    """
    
    def __init__(self, vote_count_percentile=0.90):
        """
        Initialize the recommender.
        
        Args:
            vote_count_percentile: The percentile cutoff for minimum votes (default 90%)
        """
        self.vote_count_percentile = vote_count_percentile
        self.C = None  # Mean vote across all movies
        self.m = None  # Minimum votes required
        self.qualified_movies = None
        
    def fit(self, metadata_df):
        """
        Fit the recommender to the dataset.
        
        Args:
            metadata_df: DataFrame containing movie metadata with vote_average and vote_count
        """
        # Check if dataset has required columns
        required_columns = ['vote_average', 'vote_count', 'title']
        missing_columns = [col for col in required_columns if col not in metadata_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Calculate the mean vote across all movies
        self.C = metadata_df['vote_average'].mean()
        
        # Calculate the minimum votes required (90th percentile)
        self.m = metadata_df['vote_count'].quantile(self.vote_count_percentile)
        
        # Filter for movies with enough votes
        qualified = metadata_df.copy().loc[metadata_df['vote_count'] >= self.m]
        
        # Calculate the weighted score
        qualified['score'] = qualified.apply(self._weighted_rating, axis=1)
        
        # Sort by score
        self.qualified_movies = qualified.sort_values('score', ascending=False)
        
        print(f"Simple recommender fitted with {len(self.qualified_movies)} qualified movies.")
        print(f"Mean rating (C): {self.C:.2f}")
        print(f"Minimum votes (m): {self.m:.0f}")
        
        return self
    
    def _weighted_rating(self, x):
        """
        Calculate the weighted rating for a movie.
        
        Args:
            x: Series containing vote_count and vote_average for a movie
            
        Returns:
            float: Weighted score
        """
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + self.m) * R) + (self.m / (v + self.m) * self.C)
    
    def recommend(self, movie_title=None, top_n=10):
        """
        Return the top N movies by weighted score.
        
        Args:
            movie_title: Not used in this recommender, included for API consistency
            top_n: Number of movies to recommend
            
        Returns:
            List of (movie_title, imdb_id_full) tuples
        """
        if self.qualified_movies is None:
            print("Error: Recommender has not been fitted. Call fit() first.")
            return None
            
        # Return the top N movies
        top_movies = self.qualified_movies.head(top_n)
        
        # Format the output to match other recommenders
        # Return list of tuples with (title, imdb_id)
        results = []
        for _, movie in top_movies.iterrows():
            # Handle case where imdb_id_full might not exist
            imdb_id = movie.get('imdb_id_full', None)
            results.append((movie['title'], imdb_id))
            
        return results
    
    def get_details(self):
        """
        Returns details about the recommender setup.
        
        Returns:
            dict: Information about the recommender parameters
        """
        return {
            "name": "Simple Movie Recommender (IMDB Formula)",
            "type": "weighted_rating",
            "mean_vote_threshold": self.C,
            "minimum_vote_threshold": self.m,
            "vote_percentile": self.vote_count_percentile,
            "qualified_movies_count": len(self.qualified_movies) if self.qualified_movies is not None else 0
        }


if __name__ == "__main__":
    # Example usage
    import os
    import sys
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(project_root))
    
    try:
        from src.data_loader import load_metadata
    except ImportError:
        print("Error importing modules. Run this script from the project root directory.")
        sys.exit(1)
    
    print("Testing SimpleRecommender...")
    
    # Load data
    metadata_df = load_metadata()
    
    if not metadata_df.empty:
        # Create and fit the recommender
        recommender = SimpleRecommender()
        recommender.fit(metadata_df)
        
        # Get top 10 recommendations
        recommendations = recommender.recommend(top_n=10)
        
        # Print results
        print("\nTop 10 Movies by Weighted Rating:")
        print("-" * 60)
        for i, (title, imdb_id) in enumerate(recommendations):
            link = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else "N/A"
            print(f"{i+1}. {title} - {link}")
    else:
        print("Failed to load metadata.") 