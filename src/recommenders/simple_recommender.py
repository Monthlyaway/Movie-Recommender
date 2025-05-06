"""
Simple Movie Recommender based on IMDB Weighted Rating formula.

This recommender ranks movies by a weighted rating formula that balances
the average rating and the number of votes, addressing the issue where movies
with high ratings but few votes might not be as reliable as movies with
slightly lower ratings but many votes.
"""

import pandas as pd
import numpy as np
from src.utils.weighted_score import weighted_rating, calculate_all_weighted_scores
from src.data_loader import load_ratings

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
        self.ratings_df = None  # Store ratings data
        
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
            
        # Calculate weighted scores using the utility function
        weighted_scores_result = calculate_all_weighted_scores(
            metadata_df, 
            vote_count_percentile=self.vote_count_percentile
        )
        
        self.C = weighted_scores_result['C']
        self.m = weighted_scores_result['m']
        
        # Filter for movies with enough votes
        qualified = metadata_df.copy().loc[metadata_df['vote_count'] >= self.m]
        
        # Apply the weighted scores
        qualified['score'] = weighted_scores_result['scores'].loc[qualified.index]
        
        # Load ratings data
        print("Loading ratings data...")
        self.ratings_df = load_ratings()
        
        # Track the original rating scale
        self.original_rating_scale = 5  # Default assumption
        
        # Calculate average user rating per movie from ratings.csv
        if not self.ratings_df.empty:
            try:
                avg_ratings = self.ratings_df.groupby('movieId')['rating'].mean().reset_index()
                avg_ratings.rename(columns={'rating': 'avg_user_rating'}, inplace=True)
                
                # Check the scale of ratings
                max_rating = self.ratings_df['rating'].max()
                self.original_rating_scale = max_rating  # Store the original scale
                print(f"Maximum user rating in dataset: {max_rating}")
                
                # If ratings are on a scale other than 10, normalize to 1-10 scale
                if max_rating != 10 and max_rating > 0:
                    scale_factor = 10.0 / max_rating
                    print(f"Normalizing user ratings from 1-{max_rating} to 1-10 scale (factor: {scale_factor})")
                    avg_ratings['avg_user_rating'] = avg_ratings['avg_user_rating'] * scale_factor
                
                # Merge average user ratings with qualified movies
                qualified = pd.merge(qualified, avg_ratings, left_on='movieId', right_on='movieId', how='left')
                
                # Fill NaN values for movies that don't have user ratings
                qualified['avg_user_rating'] = qualified['avg_user_rating'].fillna(0)
                print(f"Added average user ratings for {qualified['avg_user_rating'].notna().sum()} movies")
            except Exception as e:
                print(f"Error processing ratings data: {e}")
                qualified['avg_user_rating'] = 0
        else:
            print("No ratings data available.")
            qualified['avg_user_rating'] = 0
        
        # Sort by score
        self.qualified_movies = qualified.sort_values('score', ascending=False)
        
        print(f"Simple recommender fitted with {len(self.qualified_movies)} qualified movies.")
        print(f"Mean rating (C): {self.C:.2f}")
        print(f"Minimum votes (m): {self.m:.0f}")
        
        return self
    
    def recommend(self, movie_title=None, top_n=10):
        """
        Return the top N movies by weighted score.
        
        Args:
            movie_title: Not used in this recommender, included for API consistency
            top_n: Number of movies to recommend
            
        Returns:
            Tuple of (recommendations list, parameters dict)
            - List of (movie_title, imdb_id_full, weighted_score, avg_user_rating) tuples
            - Dict containing C and m values
        """
        if self.qualified_movies is None:
            print("Error: Recommender has not been fitted. Call fit() first.")
            return None
            
        # Return the top N movies
        top_movies = self.qualified_movies.head(top_n)
        
        # Format the output to include weighted scores and user ratings
        results = []
        for _, movie in top_movies.iterrows():
            # Handle case where imdb_id_full might not exist
            imdb_id = movie.get('imdb_id_full', None)
            weighted_score = float(movie['score'])
            avg_user_rating = float(movie.get('avg_user_rating', 0))
            
            results.append((movie['title'], imdb_id, weighted_score, avg_user_rating))
        
        # Include C and m parameters in the return
        params = {
            "C": round(self.C, 2),  # global mean rating
            "m": int(self.m),       # minimum votes threshold
            "rating_scale": self.original_rating_scale
        }
            
        return results, params
    
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
        recommendations, params = recommender.recommend(top_n=10)
        
        # Print results
        print("\nTop 10 Movies by Weighted Rating:")
        print("-" * 60)
        print(f"{'Rank':<4} | {'Title':<40} | {'IMDb ID':<12} | {'Weighted Score':<15} | {'User Rating':<10}")
        print("-" * 100)
        for i, (title, imdb_id, weighted_score, avg_user_rating) in enumerate(recommendations):
            link = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else "N/A"
            print(f"{i+1:<4} | {title[:40]:<40} | {imdb_id or 'N/A':<12} | {weighted_score:.2f:<15} | {avg_user_rating:.1f}")
        print("\nParameters:")
        print(params)
    else:
        print("Failed to load metadata.") 