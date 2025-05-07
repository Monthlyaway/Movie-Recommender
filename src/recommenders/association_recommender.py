"""
Association Rule Mining Recommender System.

This recommender uses the FP-Growth algorithm to discover hidden patterns in user behavior,
identifying relationships like "users who liked Movie A also liked Movie B."
"""

import pandas as pd
import numpy as np
import random
from typing import List, Optional, Tuple, Dict, Union, Set
# Import the fuzzy search utility
from src.utils.fuzzy_search import find_top_fuzzy_matches
# Import weighted score utility
from src.utils.weighted_score import weighted_rating


class AssociationRecommender:
    """
    Recommends movies based on association rules mined from user behavior patterns.

    Uses the FP-Growth algorithm to find frequent patterns and generate rules like
    "users who liked Movie A also liked Movie B."
    """

    def __init__(self, min_support=0.06, min_confidence=0.3, min_lift=1.2, rating_threshold=6):
        """
        Initialize the recommender.

        Args:
            min_support: Minimum support threshold for FP-Growth (default: 0.06)
            min_confidence: Minimum confidence for rules (default: 0.3 or 30%)
            min_lift: Minimum lift for rules (default: 1.2)
            rating_threshold: Minimum rating to consider a movie as "liked" (default: 6)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.rating_threshold = rating_threshold
        self.rules = None
        self.movie_id_to_title = None
        self.title_to_movie_id = None
        self.frequent_itemsets = None
        self.metadata = None

    def fit(self, metadata_df, ratings_df):
        """
        Fit the recommender to the dataset by mining frequent itemsets.

        Args:
            metadata_df: DataFrame containing movie metadata with id and title
            ratings_df: DataFrame containing user ratings with userId, movieId, rating

        Returns:
            self: The fitted recommender or None if fitting fails
        """
        print("Fitting AssociationRecommender...")
        # Store metadata for later use
        self.metadata = metadata_df

        # 1. Create movie ID to title mapping
        print("Creating movie ID to title mapping...")
        self.movie_id_to_title = dict(
            zip(metadata_df['id'], metadata_df['title']))
        self.title_to_movie_id = {title: id for id,
                                  title in self.movie_id_to_title.items()}

        # 2. Filter ratings to only include "liked" movies
        print(f"Filtering ratings with threshold {self.rating_threshold}...")
        liked_ratings = ratings_df[ratings_df['rating']
                                   >= self.rating_threshold]

        # Check if we have enough ratings after filtering
        if len(liked_ratings) < 100:  # Arbitrary threshold, adjust as needed
            print(
                f"\nERROR: Not enough ratings meet the threshold of {self.rating_threshold}.")
            print(
                f"Only {len(liked_ratings)} ratings out of {len(ratings_df)} exceed this threshold.")
            print("Consider lowering the rating threshold or using a larger dataset.")
            return None

        # 3. Group ratings by user to create transactions
        print("Creating user transactions...")
        transactions = liked_ratings.groupby(
            'userId')['movieId'].apply(list).tolist()

        # 4. Encode transactions for FP-Growth
        print("Encoding transactions...")
        from mlxtend.preprocessing import TransactionEncoder
        encoder = TransactionEncoder()
        encoded_data = encoder.fit_transform(transactions)
        df_encoded = pd.DataFrame(encoded_data, columns=encoder.columns_)

        # 5. Run FP-Growth to find frequent itemsets
        print(f"Running FP-Growth with min_support={self.min_support}...")
        from mlxtend.frequent_patterns import fpgrowth
        self.frequent_itemsets = fpgrowth(
            df_encoded,
            min_support=self.min_support,
            use_colnames=True
        )

        # Check if any frequent itemsets were found
        if len(self.frequent_itemsets) == 0:
            print(
                f"\nERROR: No frequent itemsets found with min_support={self.min_support}.")
            print("This can happen when:")
            print(
                f"1. The rating threshold is too high (current: {self.rating_threshold})")
            print(
                f"2. The minimum support is too high (current: {self.min_support})")
            print("3. There aren't enough common patterns in the dataset")
            print("\nPossible solutions:")
            print("- Lower the rating threshold (--rating-threshold option)")
            print("- Lower the minimum support (--min-support option)")
            print("- Use a larger dataset")
            return None

        print(
            f"Association recommender fitted with {len(self.frequent_itemsets)} frequent itemsets.")
        print(f"Generated from {len(transactions)} user transactions.")

        return self

    def _get_imdb_id(self, movie_id):
        """
        Helper method to get the IMDb ID for a movie.

        Args:
            movie_id: The ID of the movie

        Returns:
            The IMDb ID string (tt-prefixed) or None if not found
        """
        if self.metadata is None:
            return None

        movie_row = self.metadata[self.metadata['id'] == movie_id]
        if movie_row.empty:
            return None

        return movie_row.iloc[0].get('imdb_id_full', None)

    def recommend(self, movie_title, top_n=10):
        """
        Recommend movies based on the given movie title.

        Args:
            movie_title: The title of the movie to get recommendations for
            top_n: Number of recommendations to return

        Returns:
            List of (movie_title, imdb_id_full) tuples or
            List of potential title matches (for fuzzy matching) or
            None if no match or rules found
        """
        if self.frequent_itemsets is None:
            print("Error: Recommender has not been fitted. Call fit() first.")
            return None

        # Find movie ID from title (with fuzzy matching support similar to PlotRecommender)
        movie_id = self.title_to_movie_id.get(movie_title)

        if movie_id is None:
            # Return potential fuzzy matches like PlotRecommender
            print(
                f"Exact title '{movie_title}' not found. Attempting fuzzy search...")
            available_titles = list(self.title_to_movie_id.keys())
            fuzzy_matches = find_top_fuzzy_matches(
                movie_title, available_titles, limit=5, score_cutoff=75)

            if fuzzy_matches:
                potential_matches = [match[0] for match in fuzzy_matches]
                print(f"Found potential fuzzy matches: {potential_matches}")
                return potential_matches  # Return list of potential titles
            else:
                print(
                    f"Error: Movie title '{movie_title}' not found, and no close match found via fuzzy search.")
                return None

        print(
            f"Finding co-occurrences for movie: '{movie_title}' (ID: {movie_id})...")

        # Find all frequent itemsets that contain the movie
        co_occurring_movies = {}
        for _, itemset in self.frequent_itemsets.iterrows():
            items = itemset['itemsets']
            # Must contain movie_id and at least one other item
            if movie_id in items and len(items) > 1:
                # Add all other movies in the itemset as potential recommendations
                for other_movie_id in items:
                    if other_movie_id != movie_id:
                        # Use support of the itemset as a factor for this movie's score
                        if other_movie_id in co_occurring_movies:
                            co_occurring_movies[other_movie_id] += itemset['support']
                        else:
                            co_occurring_movies[other_movie_id] = itemset['support']

        # Check if any co-occurring movies were found
        if not co_occurring_movies:
            print(f"No frequent itemsets containing movie: {movie_title}")
            return None

        # Get weighted scores for ranking
        recommendations = []
        for movie_id, support in co_occurring_movies.items():
            title = self.movie_id_to_title.get(movie_id)
            if title is None:
                continue  # Skip if title not found

            # Find movie metadata for weighted score calculation
            movie_metadata = self.metadata[self.metadata['id'] == movie_id]
            if not movie_metadata.empty:
                # Calculate weighted score if vote data is available
                if 'vote_average' in movie_metadata.columns and 'vote_count' in movie_metadata.columns:
                    row = movie_metadata.iloc[0]
                    # Use mean and minimum vote count from metadata for weighted rating
                    C = self.metadata['vote_average'].mean()
                    m = self.metadata['vote_count'].quantile(0.9)
                    w_score = weighted_rating(row, C, m)
                    # Combine support from frequent itemsets with weighted score
                    final_score = support * w_score
                else:
                    # If no vote data, just use support
                    final_score = support
            else:
                # If metadata not found, just use support
                final_score = support

            # Find IMDb ID from metadata
            imdb_id = self._get_imdb_id(movie_id)
            recommendations.append((title, imdb_id, final_score))

        # Sort by final score (highest first)
        recommendations.sort(key=lambda x: x[2], reverse=True)

        # Return top N recommendations (without the score)
        top_recommendations = [(title, imdb_id)
                               for title, imdb_id, _ in recommendations[:top_n]]

        print(
            f"Found {len(top_recommendations)} recommendations for '{movie_title}'")
        return top_recommendations

    def get_details(self):
        """
        Returns details about the recommender setup.

        Returns:
            dict: Information about the recommender parameters
        """
        return {
            "name": "Association Rule Mining Recommender",
            "type": "association_rules",
            "min_support": self.min_support,
            "min_confidence": self.min_confidence,
            "min_lift": self.min_lift,
            "rating_threshold": self.rating_threshold,
            "frequent_itemsets_count": len(self.frequent_itemsets) if self.frequent_itemsets is not None else 0,
            "approach": "Direct co-occurrence with weighted score ranking"
        }

    def get_random_rules(self, num_rules=5) -> List[Dict]:
        """
        Get random frequent itemsets with detailed information.

        Args:
            num_rules: Number of random itemsets to retrieve (default: 5)

        Returns:
            List of dictionaries containing itemset information or empty list if no itemsets
        """
        if self.frequent_itemsets is None or len(self.frequent_itemsets) == 0:
            return []

        # Get random sample of itemsets
        sample_size = min(num_rules, len(self.frequent_itemsets))
        sample_indices = random.sample(
            range(len(self.frequent_itemsets)), sample_size)
        sample_itemsets = self.frequent_itemsets.iloc[sample_indices]

        result = []
        for _, itemset in sample_itemsets.iterrows():
            # Convert IDs to movie titles for readability
            itemset_ids = list(itemset['itemsets'])

            itemset_titles = [self.movie_id_to_title.get(movie_id, f"Unknown ({movie_id})")
                              for movie_id in itemset_ids]

            # Create a dictionary with itemset information
            itemset_info = {
                'movies': itemset_titles,
                'support': itemset['support'],
                'movie_ids': itemset_ids  # Include IDs for potential lookups
            }

            result.append(itemset_info)

        return result


if __name__ == "__main__":
    # Example usage
    import os
    import sys

    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(project_root))

    try:
        from src.data_loader import load_metadata, load_ratings
    except ImportError:
        print("Error importing modules. Run this script from the project root directory.")
        sys.exit(1)

    print("Testing AssociationRecommender...")

    # Load data
    metadata_df = load_metadata()
    ratings_df = load_ratings()

    if not metadata_df.empty and not ratings_df.empty:
        # Create and fit the recommender
        recommender = AssociationRecommender(
            min_support=0.06,
            min_confidence=0.3,
            min_lift=1.2,
            rating_threshold=6
        )
        recommender.fit(metadata_df, ratings_df)

        # Get recommendations for a movie
        test_movie = "The Dark Knight"
        recommendations = recommender.recommend(test_movie, top_n=10)

        # Print results
        print("\nTop 10 Recommendations for", test_movie)
        print("-" * 60)
        if recommendations:
            for i, (title, imdb_id) in enumerate(recommendations):
                link = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else "N/A"
                print(f"{i+1}. {title} - {link}")
        else:
            print("No recommendations found.")
    else:
        print("Failed to load data.")
