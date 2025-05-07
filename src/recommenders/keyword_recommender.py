import pandas as pd
import numpy as np
import ast # For safely evaluating stringified literals
import math
from collections import Counter

# Attempt to import from utils, assuming it's in the PYTHONPATH
try:
    from src.utils.weighted_score import calculate_normalized_weighted_scores
except ImportError:
    # Fallback for cases where src is not directly in PYTHONPATH (e.g. running script directly)
    # This might happen if the script is run from a different directory or if PYTHONPATH isn't set up
    # for src. Adjust as necessary based on your project structure.
    import sys
    import os
    # Add the project root to sys.path to allow absolute imports from src
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.utils.weighted_score import calculate_normalized_weighted_scores


class KeywordRecommender:
    """
    Recommends movies based on user-provided keywords, combining keyword relevance
    with IMDB-style weighted scores.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, vote_count_percentile: float = 0.85):
        """
        Initializes the KeywordRecommender.

        Args:
            alpha: Weight for the keyword relevance score (KRS).
            beta: Weight for the normalized IMDB weighted score.
            vote_count_percentile: Percentile for minimum votes in IMDB score calculation.
        """
        self.alpha = alpha
        self.beta = beta
        self.vote_count_percentile = vote_count_percentile

        self.metadata_df = pd.DataFrame()
        self.idf_scores = {}  # Stores IDF score for each keyword: {keyword_name: idf_value}
        # Stores set of keywords for each movie ID: movie_id -> {kw1, kw2}
        # This will be a Series aligned with metadata_df's index (movie_id)
        self.movie_keyword_sets = pd.Series(dtype='object')
        # Stores normalized IMDB scores, aligned with metadata_df's index
        self.normalized_imdb_scores = pd.Series(dtype='float64')

        self._fitted = False # Flag to check if fit has been called

    def _parse_keyword_string(self, keyword_str: str) -> set:
        """
        Parses a stringified list of JSON objects (keywords) into a set of keyword names.

        Args:
            keyword_str: The string from the 'keywords' column of keywords.csv.
                         Example: "[{'id': 931, 'name': 'jealousy'}, ...]"

        Returns:
            A set of keyword names (e.g., {'jealousy', 'toy'}).
            Returns an empty set if input is NaN, empty, or malformed.
        """
        # Handle if keyword_str is a Series (e.g. due to duplicate movie IDs in keywords.csv)
        if isinstance(keyword_str, pd.Series):
            if keyword_str.empty:
                return set()
            # Attempt to use the first item if it's a series, or handle as error/warning
            # This is a common case if an ID maps to multiple keyword entries.
            # For now, let's print a warning and use the first one.
            # A more robust solution might involve concatenating or choosing one based on criteria.
            print(f"Warning: Multiple keyword entries found for a movie ID. Using the first one. Value: {keyword_str.iloc[0]}")
            keyword_str = keyword_str.iloc[0] # Take the first item

        if pd.isna(keyword_str) or not isinstance(keyword_str, str) or not keyword_str.strip():
            return set()
        try:
            # Safely evaluate the string representation of the list of dicts
            keyword_list = ast.literal_eval(keyword_str)
            # Extract the 'name' from each dictionary in the list
            return {kw['name'].lower() for kw in keyword_list if isinstance(kw, dict) and 'name' in kw}
        except (ValueError, SyntaxError, TypeError):
            # Handle cases where ast.literal_eval fails (e.g., malformed string)
            # print(f"Warning: Could not parse keyword string: {keyword_str}") # Optional: for debugging
            return set()

    def _calculate_idf(self, all_movie_keyword_sets: pd.Series):
        """
        Calculates Inverse Document Frequency (IDF) for all unique keywords.
        IDF = log((Total Number of Documents + 1) / (Document Frequency of Keyword + 1)) + 1 (smoothing)

        Args:
            all_movie_keyword_sets: A Pandas Series where each element is a set of
                                    keyword names for a movie.
        """
        if all_movie_keyword_sets.empty:
            self.idf_scores = {}
            return

        total_movies = len(all_movie_keyword_sets)
        keyword_doc_frequency = Counter()

        for keyword_set in all_movie_keyword_sets:
            if isinstance(keyword_set, set):
                keyword_doc_frequency.update(list(keyword_set)) # Count unique keywords per movie

        self.idf_scores = {}
        for keyword, doc_freq in keyword_doc_frequency.items():
            # Apply smoothing: +1 to numerator and denominator to avoid division by zero
            # and log(1) = 0 for keywords present in all docs.
            # Adding +1 to the final IDF ensures scores are > 0.
            self.idf_scores[keyword] = math.log((total_movies + 1) / (doc_freq + 1)) + 1

    def fit(self, metadata_df: pd.DataFrame, keywords_data_path: str = 'dataset/keywords.csv') -> bool:
        """
        Preprocesses data and fits the recommender.
        - Loads and parses keywords.
        - Calculates IDF scores for keywords.
        - Calculates normalized IMDB weighted scores for movies.

        Args:
            metadata_df: DataFrame containing movie metadata (must include 'id', 'vote_average', 'vote_count').
            keywords_data_path: Path to the keywords.csv file.

        Returns:
            True if fitting was successful, False otherwise.
        """
        if not all(col in metadata_df.columns for col in ['id', 'title', 'vote_average', 'vote_count']):
            print("Error: metadata_df is missing required columns ('id', 'title', 'vote_average', 'vote_count').")
            return False

        self.metadata_df = metadata_df.copy()
        # Ensure 'id' is the index for easy joining and lookup
        if 'id' in self.metadata_df.columns:
             self.metadata_df.set_index('id', inplace=True, drop=False) # Keep 'id' as a column too

        # 1. Load and Parse Keywords
        try:
            print(f"Loading keywords from: {keywords_data_path}")
            keywords_df = pd.read_csv(keywords_data_path)
            # Ensure 'id' in keywords_df is of the same type as metadata_df's index
            keywords_df['id'] = keywords_df['id'].astype(self.metadata_df.index.dtype)
            keywords_df.set_index('id', inplace=True)
        except FileNotFoundError:
            print(f"Error: Keywords file not found at {keywords_data_path}")
            return False
        except Exception as e:
            print(f"Error loading keywords.csv: {e}")
            return False

        # Parse keyword strings and map to movie IDs in metadata_df
        # Create a Series of keyword sets, indexed by movie ID from metadata_df
        self.movie_keyword_sets = self.metadata_df.index.to_series().apply(
            lambda movie_id: self._parse_keyword_string(
                keywords_df.loc[movie_id, 'keywords']
            ) if movie_id in keywords_df.index else set()
        )
        self.metadata_df['keyword_set'] = self.movie_keyword_sets

        # 2. Calculate IDF Scores
        print("Calculating IDF scores...")
        self._calculate_idf(self.movie_keyword_sets)
        if not self.idf_scores:
            print("Warning: IDF scores are empty. This might happen if no keywords were found or parsed.")
            # Depending on desired behavior, this could be a critical failure.
            # For now, we'll allow it but recommendations might be poor.

        # 3. Calculate Normalized IMDB Scores
        print("Calculating normalized IMDB weighted scores...")
        try:
            # Ensure metadata_df has the necessary columns for weighted_score calculation
            # The function calculate_normalized_weighted_scores expects 'vote_average' and 'vote_count'
            imdb_scores_result = calculate_normalized_weighted_scores(
                self.metadata_df,
                vote_count_percentile=self.vote_count_percentile
            )
            # The 'normalized_scores' Series is indexed by the original DataFrame's index
            self.normalized_imdb_scores = imdb_scores_result['normalized_scores']
            self.metadata_df['normalized_imdb_score'] = self.normalized_imdb_scores
        except Exception as e:
            print(f"Error calculating normalized IMDB scores: {e}")
            # import traceback
            # traceback.print_exc()
            return False

        self._fitted = True
        print("KeywordRecommender fitting complete.")
        return True

    def recommend(self, user_keywords_str: str, top_n: int = 10) -> pd.DataFrame:
        """
        Recommends movies based on user keywords.

        Args:
            user_keywords_str: A comma-separated string of keywords from the user.
            top_n: The number of top movies to recommend.

        Returns:
            A pandas DataFrame containing the top_n recommended movies,
            including 'title', 'final_score', 'keyword_relevance_score',
            'normalized_imdb_score', and 'keyword_set'.
            Returns an empty DataFrame if not fitted or on error.
        """
        if not self._fitted:
            print("Error: Recommender has not been fitted. Call fit() first.")
            return pd.DataFrame()

        if not user_keywords_str or not user_keywords_str.strip():
            print("No keywords provided. Returning empty recommendations.")
            return pd.DataFrame()

        user_keywords = {kw.strip().lower() for kw in user_keywords_str.split(',')}
        if not user_keywords:
            print("Parsed keywords are empty. Returning empty recommendations.")
            return pd.DataFrame()
        
        print(f"User keywords: {user_keywords}")

        # Calculate Keyword Relevance Score (KRS) for each movie
        krs_values = []
        for movie_id, movie_data in self.metadata_df.iterrows():
            movie_kws = movie_data.get('keyword_set', set())
            matched_keywords = user_keywords.intersection(movie_kws)
            if matched_keywords:
                krs = sum(self.idf_scores.get(kw, 0) for kw in matched_keywords)
                krs_values.append(krs)
            else:
                krs_values.append(0)
        
        self.metadata_df['krs'] = krs_values

        # Normalize KRS for movies that had at least one match (KRS > 0)
        positive_krs = self.metadata_df[self.metadata_df['krs'] > 0]['krs']
        if not positive_krs.empty:
            min_krs = positive_krs.min()
            max_krs = positive_krs.max()
            krs_range = max_krs - min_krs
            if krs_range == 0: # All positive KRS are the same
                 self.metadata_df['normalized_krs'] = positive_krs.apply(lambda x: 1.0 if x > 0 else 0.0)
            else:
                self.metadata_df['normalized_krs'] = self.metadata_df['krs'].apply(
                    lambda x: (x - min_krs) / krs_range if x > 0 else 0.0
                )
        else: # No movies matched any keywords
            self.metadata_df['normalized_krs'] = 0.0
            print("No movies matched the provided keywords.")
            # return pd.DataFrame() # Or return empty if no matches is considered an error for ranking

        # Calculate Final Score
        # Ensure 'normalized_imdb_score' exists and handle potential NaNs if any movie didn't get a score
        self.metadata_df['final_score'] = (self.alpha * self.metadata_df['normalized_krs'].fillna(0)) + \
                                          (self.beta * self.metadata_df['normalized_imdb_score'].fillna(0))

        # Sort and get top N
        recommendations = self.metadata_df.sort_values(by='final_score', ascending=False)
        
        # Select relevant columns for output
        output_columns = ['title', 'final_score', 'normalized_krs', 'normalized_imdb_score', 'keyword_set', 'overview']
        # Filter for columns that actually exist in recommendations to avoid KeyErrors
        existing_output_columns = [col for col in output_columns if col in recommendations.columns]
        
        # Add movie ID to output
        if 'id' in recommendations.columns:
             final_recs = recommendations[['id'] + existing_output_columns].head(top_n)
        else: # if 'id' was the index and not dropped
             final_recs = recommendations[existing_output_columns].head(top_n)
             final_recs.reset_index(inplace=True) # Bring 'id' from index to column

        return final_recs

if __name__ == '__main__':
    # This is a placeholder for basic testing.
    # In a real scenario, you'd load data using data_loader.py
    print("KeywordRecommender Basic Test (Illustrative)")

    # Create dummy metadata
    data = {
        'id': [1, 2, 3, 4, 5],
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
        'overview': ['Plot A...', 'Plot B...', 'Plot C...', 'Plot D...', 'Plot E...'],
        'vote_average': [7.0, 8.0, 6.5, 7.5, 9.0],
        'vote_count': [1000, 2000, 500, 1500, 3000]
    }
    sample_metadata_df = pd.DataFrame(data)

    # Create dummy keywords.csv content
    # In a real scenario, this would be a file path
    keywords_content = """id,keywords
1,"[{'id': 1, 'name': 'action'}, {'id': 2, 'name': 'future'}]"
2,"[{'id': 1, 'name': 'action'}, {'id': 3, 'name': 'comedy'}, {'id': 4, 'name': 'space'}]"
3,"[{'id': 5, 'name': 'drama'}, {'id': 6, 'name': 'romance'}]"
4,"[{'id': 2, 'name': 'future'}, {'id': 7, 'name': 'thriller'}]"
5,"[{'id': 3, 'name': 'comedy'}, {'id': 8, 'name': 'family'}, {'id': 4, 'name': 'space'}]"
"""
    from io import StringIO
    keywords_file_path_or_buffer = StringIO(keywords_content) # Use StringIO to simulate a file

    # Initialize and fit recommender
    recommender = KeywordRecommender(alpha=0.6, beta=0.4, vote_count_percentile=0.5) # Lower percentile for small dummy data
    
    # We need to ensure the data_loader can be found for the weighted_score import
    # This setup is a bit tricky for a standalone __main__ block if src isn't in PYTHONPATH
    # For this test, we assume the try-except block for imports at the top handles it.

    print("\nFitting recommender...")
    fit_successful = recommender.fit(sample_metadata_df.copy(), keywords_data_path=keywords_file_path_or_buffer)

    if fit_successful:
        print("\nRecommender fitted successfully.")
        print(f"IDF Scores: {recommender.idf_scores}")
        
        print("\n--- Test Case 1: 'action, space' ---")
        recs1 = recommender.recommend("action, space", top_n=3)
        if not recs1.empty:
            print(recs1[['id', 'title', 'final_score', 'normalized_krs', 'normalized_imdb_score', 'keyword_set']])
        else:
            print("No recommendations found.")

        print("\n--- Test Case 2: 'future' ---")
        recs2 = recommender.recommend("future", top_n=3)
        if not recs2.empty:
            print(recs2[['id', 'title', 'final_score', 'normalized_krs', 'normalized_imdb_score', 'keyword_set']])
        else:
            print("No recommendations found.")

        print("\n--- Test Case 3: 'unknown_keyword' ---")
        recs3 = recommender.recommend("unknown_keyword", top_n=3)
        if not recs3.empty:
            print(recs3[['id', 'title', 'final_score', 'normalized_krs', 'normalized_imdb_score', 'keyword_set']])
        else:
            print("No recommendations found.")
            
        print("\n--- Test Case 4: Empty keyword string ---")
        recs4 = recommender.recommend("", top_n=3)
        if not recs4.empty:
            print(recs4[['id', 'title', 'final_score', 'normalized_krs', 'normalized_imdb_score', 'keyword_set']])
        else:
            print("No recommendations found (as expected for empty input).")
    else:
        print("\nRecommender fitting failed.")

    print("\nKeywordRecommender test finished.")