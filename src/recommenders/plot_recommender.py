import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os  # For potential future path handling, though not strictly needed now
# Import the fuzzy search utility
from src.utils.fuzzy_search import find_top_fuzzy_matches  # Import the new function
# Import the weighted score utility
from src.utils.weighted_score import calculate_normalized_weighted_scores
# Import necessary types for type hinting at the top level
from typing import List, Optional, Sequence, Tuple, Union, Dict, Any


class PlotRecommender:
    """
    Recommends movies based on plot similarity using TF-IDF and cosine similarity,
    combined with a weighted rating formula (IMDB formula) for better quality recommendations.
    """

    def __init__(self, similarity_weight: float = 0.7, vote_count_percentile: float = 0.90):
        """
        Initialize the recommender.

        Args:
            similarity_weight: Weight given to plot similarity (0-1). The formula used is:
                              similarity_weight * similarity + (1-similarity_weight) * weighted_score
                              With default 0.7, this means 70% similarity + 30% weighted score.
            vote_count_percentile: The percentile cutoff for minimum votes in weighted score calculation
        """
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.indices = None
        self.titles = None  # Store titles for easy lookup
        self.metadata = None  # Store the full dataframe to access links later

        # Weighted rating parameters (from SimpleRecommender)
        self.similarity_weight = similarity_weight
        self.vote_count_percentile = vote_count_percentile
        self.C = None  # Mean vote across all movies
        self.m = None  # Minimum votes required
        self.weighted_scores = None  # Store pre-calculated scores
        self.normalized_scores = None  # Store pre-calculated normalized scores

    def fit(self, metadata_df: pd.DataFrame):
        """
        Fits the recommender to the provided metadata.
        Calculates the TF-IDF matrix for movie overviews and prepares weighted rating parameters.

        Args:
            metadata_df: DataFrame containing movie metadata, requires 'overview', 'title',
                        'vote_average', and 'vote_count' columns.
        """
        print("Fitting PlotRecommender...")

        # Check required columns
        required_columns = ['overview', 'title', 'vote_average', 'vote_count']
        missing_columns = [
            col for col in required_columns if col not in metadata_df.columns]
        if missing_columns:
            raise ValueError(
                f"Metadata DataFrame missing required columns: {missing_columns}")

        # Ensure 'overview' has no NaNs
        metadata_df['overview'] = metadata_df['overview'].fillna('')

        # Calculate TF-IDF matrix
        print("Calculating TF-IDF matrix...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            metadata_df['overview'])
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

        # Calculate weighted rating parameters using the utility module
        print("Calculating weighted rating parameters...")
        weighted_scores_result = calculate_normalized_weighted_scores(
            metadata_df,
            vote_count_percentile=self.vote_count_percentile
        )

        self.C = weighted_scores_result['C']
        self.m = weighted_scores_result['m']
        self.weighted_scores = weighted_scores_result['scores']
        self.normalized_scores = weighted_scores_result['normalized_scores']
        
        print(f"Mean rating (C): {self.C:.2f}")
        print(f"Minimum votes (m): {self.m:.0f}")
        print(f"Raw weighted scores range: {weighted_scores_result['min_score']:.2f} to {weighted_scores_result['max_score']:.2f}")
        print(f"Normalized scores range: 0.00 to 1.00")

        # Create reverse map of indices and movie titles
        metadata_df = metadata_df.reset_index()  # Ensure index aligns with matrix rows
        self.metadata = metadata_df  # Store for later lookup
        self.titles = self.metadata['title']
        # Ensure indices map titles to the *new* index after reset_index
        self.indices = pd.Series(
            self.metadata.index, index=self.metadata['title']).drop_duplicates()
        print("Title to index mapping created.")
        print("Fitting complete.")

    def recommend(self, movie_title: str, top_n: int = 10) -> Optional[Union[List[Tuple[str, Optional[str], float]], List[str]]]:
        """
        Recommends movies similar to the given movie title based on a combination of 
        plot similarity and weighted rating.

        Handles exact matches, fuzzy matches (returning options), and no matches.

        Args:
            movie_title: The title of the movie to get recommendations for.
            top_n: The number of recommendations to return.

        Returns:
            - List[Tuple[str, Optional[str], float]]: If exact match found, returns list of (title, imdb_id_full, combined_score).
            - List[str]: If only fuzzy matches found, returns list of potential titles for user selection.
            - None: If no match found or model not fitted.
        """
        # Check if the model is fitted (tfidf_matrix and indices are available)
        if self.tfidf_matrix is None or self.indices is None or self.metadata is None:
            print("Error: Recommender has not been fitted yet. Call fit() first.")
            return None  # Return None for not fitted

        # --- Match Finding Logic ---
        exact_match_found = False
        potential_matches = []

        if movie_title in self.indices:
            idx = self.indices[movie_title]
            print(f"Found exact match for title: '{movie_title}'")
            exact_match_found = True
        else:
            print(
                f"Exact title '{movie_title}' not found. Attempting fuzzy search...")
            available_titles = self.indices.index.unique().tolist()  # Use unique titles
            fuzzy_matches = find_top_fuzzy_matches(
                movie_title, available_titles, limit=5, score_cutoff=75)

            if fuzzy_matches:
                potential_matches = [match[0]
                                     for match in fuzzy_matches]  # Get only the titles
                print(f"Found potential fuzzy matches: {potential_matches}")
                return potential_matches  # Signal UI to ask user
            else:
                print(
                    f"Error: Movie title '{movie_title}' not found, and no close match found via fuzzy search.")
                return None  # Return None for no match

        # --- Recommendation Generation (only if exact match found) ---
        if not exact_match_found:
            return None

        # Calculate cosine similarity on-the-fly for the given movie
        print(f"Calculating similarity for '{movie_title}'...")
        tfidf_vector = self.tfidf_matrix[idx]
        cosine_similarities = linear_kernel(
            tfidf_vector, self.tfidf_matrix).flatten()

        # Get the pairwise similarity scores
        sim_scores = list(enumerate(cosine_similarities))
        
        # STEP 1: First sort by pure similarity to get the most similar movies
        # Sort by similarity score
        sim_scores_sorted = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Skip the first one (the movie itself) and get N+10 most similar movies
        # We get extra movies to ensure we have enough after filtering
        candidate_count = min(top_n + 10, len(sim_scores_sorted) - 1)
        top_similar_movies = sim_scores_sorted[1:candidate_count+1]
        
        print(f"Selected {len(top_similar_movies)} most similar movies as candidates")
        
        # STEP 2: For these top similar movies, calculate the combined score and re-rank
        combined_scores = []
        for movie_idx, similarity in top_similar_movies:
            if movie_idx < len(self.normalized_scores):
                # Use pre-normalized weighted score (0-1 range)
                norm_weighted_score = self.normalized_scores.iloc[movie_idx]
                raw_weighted_score = self.weighted_scores.iloc[movie_idx]
                
                # Apply the weights (0.3 for weighted score, 0.7 for similarity)
                combined_score = ((1 - self.similarity_weight) * norm_weighted_score) + \
                    (self.similarity_weight * similarity)
                
                combined_scores.append(
                    (movie_idx, combined_score, similarity, raw_weighted_score, norm_weighted_score))
        
        # Sort the candidate movies by combined score
        combined_scores = sorted(
            combined_scores, key=lambda x: x[1], reverse=True)
        
        # Limit to the requested number of recommendations
        combined_scores = combined_scores[:top_n]
        
        # Retrieve movie information
        recommendations = []
        for movie_idx, combined_score, similarity, raw_weighted_score, norm_weighted_score in combined_scores:
            title = self.metadata.iloc[movie_idx]['title']
            imdb_id = self.metadata.iloc[movie_idx]['imdb_id_full']
            # Add the original vote_average from metadata
            original_rating = self.metadata.iloc[movie_idx]['vote_average']
            
            # Return all components for formula display
            # Returns: (title, imdb_id, combined_score, similarity, norm_weighted_score, original_rating)
            recommendations.append((
                title,
                imdb_id,
                round(combined_score, 3),
                round(similarity, 3),
                round(norm_weighted_score, 3),
                round(original_rating, 1)
            ))

        return recommendations

    def get_details(self) -> Dict[str, Any]:
        """
        Returns details about the recommender setup.
        
        Returns:
            dict: Information about the recommender parameters
        """
        return {
            "name": "Hybrid Plot Recommender",
            "type": "hybrid",
            "similarity_weight": self.similarity_weight,
            "weighted_rating_weight": 1 - self.similarity_weight,
            "mean_vote_threshold": self.C,
            "minimum_vote_threshold": self.m,
            "vote_percentile": self.vote_count_percentile,
            "score_formula": f"{self.similarity_weight:.1f} * similarity + {1-self.similarity_weight:.1f} * weighted_score",
            "weighted_scores_normalized": True,
            "ranking_method": "Two-stage: First select most similar movies by TF-IDF, then rank by combined score"
        }

# Type hints are now imported at the top


if __name__ == '__main__':
    # Example Usage (requires data_loader.py and dataset folder)
    # Run this from the project root directory
    print(f"Current working directory: {os.getcwd()}")
    # Adjust imports for example usage if needed, assuming running from root
    try:
        # Add project root to path if running script directly
        import sys
        sys.path.insert(0, os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        from data_loader import load_metadata
        # Import the correct fuzzy search function for example usage
        from utils.fuzzy_search import find_top_fuzzy_matches
    except ImportError as e:
        print(f"Error: Could not import necessary modules for example: {e}")
        print("Make sure data_loader.py and utils/fuzzy_search.py are accessible.")
        exit()

    print("\n--- PlotRecommender Example ---")
    metadata = load_metadata()  # Load data using the function from data_loader.py

    if not metadata.empty:
        recommender = PlotRecommender(similarity_weight=0.7)
        recommender.fit(metadata)

        # Show the recommender details
        details = recommender.get_details()
        print("\nRecommender Configuration:")
        for key, value in details.items():
            print(f"{key}: {value}")

        # Test with a known movie title
        test_title = "The Dark Knight Rises"
        print(f"\nRecommendations for '{test_title}':")
        recommendations = recommender.recommend(test_title)
        if isinstance(recommendations, list) and recommendations and isinstance(recommendations[0], tuple):
            print(
                "Rank | Title                     | IMDb ID                  | Combined Score | Similarity | Normalized Weighted Score | Original Rating")
            print(
                "---- | ------------------------- | ------------------------ | -------------- | ---------- | ------------------------ | ---------------")
            for i, (title, imdb_id, score, similarity, norm_weighted_score, original_rating) in enumerate(recommendations):
                link = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else "N/A"
                print(
                    f"{i+1:<4} | {title:<25} | {link:<24} | {score:.3f} | {similarity:.3f} | {norm_weighted_score:.3f} | {original_rating:.1f}")
        elif isinstance(recommendations, list):  # Fuzzy matches returned
            print(f"Potential matches found: {recommendations}")
        else:  # None returned
            print("No recommendations found (or movie title not in dataset).")

        # Test with a non-existent movie
        print(f"\nRecommendations for 'NonExistent Movie':")
        recommendations_nonexistent = recommender.recommend(
            "NonExistent Movye")  # Intentional typo
        if recommendations_nonexistent is None:
            print("Correctly handled non-existent movie (returned None).")
        elif isinstance(recommendations_nonexistent, list):
            print(
                f"Potential matches for non-existent movie: {recommendations_nonexistent}")

        # Test fuzzy matching directly (will return potential matches)
        print(f"\nRecommendations for 'Dark Night Rises':")  # Fuzzy query
        recommendations_fuzzy = recommender.recommend("Dark Night Rises")
        if isinstance(recommendations_fuzzy, list) and recommendations_fuzzy and isinstance(recommendations_fuzzy[0], str):
            print(
                f"Potential matches returned for fuzzy query: {recommendations_fuzzy}")
            # Example of selecting the first match and re-querying
            selected_title = recommendations_fuzzy[0]
            print(f"\nRe-querying with selected title: '{selected_title}'")
            recommendations_selected = recommender.recommend(selected_title)
            if isinstance(recommendations_selected, list) and recommendations_selected and isinstance(recommendations_selected[0], tuple):
                print("Rank | Title                     | IMDb ID                  | Combined Score | Similarity | Normalized Weighted Score | Original Rating")
                print("---- | ------------------------- | ------------------------ | -------------- | ---------- | ------------------------ | ---------------")
                for i, (title, imdb_id, score, similarity, norm_weighted_score, original_rating) in enumerate(recommendations_selected):
                    link = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else "N/A"
                    print(
                        f"{i+1:<4} | {title:<25} | {link:<24} | {score:.3f} | {similarity:.3f} | {norm_weighted_score:.3f} | {original_rating:.1f}")
            else:
                print("Failed to get recommendations for selected title.")

    else:
        print("\nSkipping PlotRecommender example because metadata failed to load.")
