import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os # For potential future path handling, though not strictly needed now
# Import the fuzzy search utility
from src.utils.fuzzy_search import find_top_fuzzy_matches # Import the new function
# Import necessary types for type hinting at the top level
from typing import List, Optional, Sequence, Tuple, Union

class PlotRecommender:
    """
    Recommends movies based on plot similarity using TF-IDF and cosine similarity.
    """
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        # self.cosine_sim = None # No longer storing the full matrix
        self.indices = None
        self.titles = None # Store titles for easy lookup
        self.metadata = None # Store the full dataframe to access links later

    def fit(self, metadata_df: pd.DataFrame):
        """
        Fits the recommender to the provided metadata.
        Calculates the TF-IDF matrix for movie overviews.

        Args:
            metadata_df: DataFrame containing movie metadata, requires 'overview' and 'title' columns.
        """
        print("Fitting PlotRecommender...")
        # Ensure 'overview' is present and has no NaNs (should be handled by data_loader, but good practice)
        if 'overview' not in metadata_df.columns:
            raise ValueError("Metadata DataFrame must contain an 'overview' column.")
        metadata_df['overview'] = metadata_df['overview'].fillna('')

        # Calculate TF-IDF matrix
        print("Calculating TF-IDF matrix...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(metadata_df['overview'])
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

        # Cosine similarity will be calculated on-demand in the recommend method

        # Create reverse map of indices and movie titles
        # Reset index if needed to ensure it aligns with matrix rows
        # Store the original df before reset_index if needed, or just use the reset one
        metadata_df = metadata_df.reset_index() # Ensure index aligns with matrix rows
        self.metadata = metadata_df # Store for later link lookup
        self.titles = self.metadata['title']
        # Ensure indices map titles to the *new* index after reset_index
        self.indices = pd.Series(self.metadata.index, index=self.metadata['title']).drop_duplicates()
        print("Title to index mapping created.")
        print("Fitting complete.")

    def recommend(self, movie_title: str, top_n: int = 10) -> Optional[List[Union[Tuple[str, Optional[str]], str]]]:
        """
        Recommends movies similar to the given movie title based on plot similarity.

        Handles exact matches, fuzzy matches (returning options), and no matches.

        Args:
            movie_title: The title of the movie to get recommendations for.
            top_n: The number of recommendations to return.

        Returns:
            - List[Tuple[str, Optional[str]]]: If exact match found, returns list of (title, imdb_id_full).
            - List[str]: If only fuzzy matches found, returns list of potential titles for user selection.
            - None: If no match found or model not fitted.
        """
        # Check if the model is fitted (tfidf_matrix and indices are available)
        if self.tfidf_matrix is None or self.indices is None or self.metadata is None:
            print("Error: Recommender has not been fitted yet. Call fit() first.")
            return None # Return None for not fitted

        # --- Match Finding Logic ---
        exact_match_found = False
        potential_matches = []

        if movie_title in self.indices:
            idx = self.indices[movie_title]
            print(f"Found exact match for title: '{movie_title}'")
            exact_match_found = True
        else:
            print(f"Exact title '{movie_title}' not found. Attempting fuzzy search...")
            available_titles = self.indices.index.unique().tolist() # Use unique titles
            fuzzy_matches = find_top_fuzzy_matches(movie_title, available_titles, limit=5, score_cutoff=75)

            if fuzzy_matches:
                potential_matches = [match[0] for match in fuzzy_matches] # Get only the titles
                # If the top fuzzy match is very strong (e.g., score > 95), consider it an exact match? Optional.
                # top_score = fuzzy_matches[0][1]
                # if top_score > 95:
                #     best_match_title = potential_matches[0]
                #     print(f"High confidence fuzzy match '{best_match_title}' (score {top_score}). Treating as exact.")
                #     idx = self.indices[best_match_title]
                #     movie_title = best_match_title
                #     exact_match_found = True
                # else:
                #     # Return potential matches for user selection
                print(f"Found potential fuzzy matches: {potential_matches}")
                return potential_matches # Signal UI to ask user
            else:
                print(f"Error: Movie title '{movie_title}' not found, and no close match found via fuzzy search.")
                return None # Return None for no match

        # --- Recommendation Generation (only if exact match found) ---
        if not exact_match_found:
             # This part should not be reached if fuzzy matches were returned above,
             # but included for safety.
             return None

        # Calculate cosine similarity on-the-fly for the given movie
        print(f"Calculating similarity for '{movie_title}'...")
        tfidf_vector = self.tfidf_matrix[idx]
        cosine_similarities = linear_kernel(tfidf_vector, self.tfidf_matrix).flatten()

        # Get the pairwise similarity scores
        sim_scores = list(enumerate(cosine_similarities))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the top_n most similar movies (skip the first one - itself)
        sim_scores = sim_scores[1:top_n+1]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Retrieve titles and IMDb IDs for the recommended movies
        recommendations = []
        for i in movie_indices:
            title = self.metadata.iloc[i]['title']
            # Use the pre-formatted 'imdb_id_full' column
            imdb_id = self.metadata.iloc[i]['imdb_id_full']
            recommendations.append((title, imdb_id))

        return recommendations

# Type hints are now imported at the top

if __name__ == '__main__':
    # Example Usage (requires data_loader.py and dataset folder)
    # Run this from the project root directory
    print(f"Current working directory: {os.getcwd()}")
    # Adjust imports for example usage if needed, assuming running from root
    try:
        # Add project root to path if running script directly
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data_loader import load_metadata
        # Import the correct fuzzy search function for example usage
        from utils.fuzzy_search import find_top_fuzzy_matches
    except ImportError as e:
         print(f"Error: Could not import necessary modules for example: {e}")
         print("Make sure data_loader.py and utils/fuzzy_search.py are accessible.")
         exit()

    print("\n--- PlotRecommender Example ---")
    metadata = load_metadata() # Load data using the function from data_loader.py

    if not metadata.empty:
        recommender = PlotRecommender()
        recommender.fit(metadata)

        # Test with a known movie title
        test_title = "The Dark Knight Rises"
        print(f"\nRecommendations for '{test_title}':")
        recommendations = recommender.recommend(test_title)
        if isinstance(recommendations, list) and recommendations and isinstance(recommendations[0], tuple):
            print("Rank | Title                     | IMDb ID")
            print("---- | ------------------------- | ---------")
            for i, (title, imdb_id) in enumerate(recommendations):
                link = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else "N/A"
                print(f"{i+1:<4} | {title:<25} | {link}")
        elif isinstance(recommendations, list): # Fuzzy matches returned
             print(f"Potential matches found: {recommendations}")
        else: # None returned
            print("No recommendations found (or movie title not in dataset).")

        # Test with a non-existent movie
        print(f"\nRecommendations for 'NonExistent Movie':")
        recommendations_nonexistent = recommender.recommend("NonExistent Movye") # Intentional typo
        if recommendations_nonexistent is None:
            print("Correctly handled non-existent movie (returned None).")
        elif isinstance(recommendations_nonexistent, list):
             print(f"Potential matches for non-existent movie: {recommendations_nonexistent}")


        # Test fuzzy matching directly (will return potential matches)
        print(f"\nRecommendations for 'Dark Night Rises':") # Fuzzy query
        recommendations_fuzzy = recommender.recommend("Dark Night Rises")
        if isinstance(recommendations_fuzzy, list) and recommendations_fuzzy and isinstance(recommendations_fuzzy[0], str):
             print(f"Potential matches returned for fuzzy query: {recommendations_fuzzy}")
             # Example of selecting the first match and re-querying
             selected_title = recommendations_fuzzy[0]
             print(f"\nRe-querying with selected title: '{selected_title}'")
             recommendations_selected = recommender.recommend(selected_title)
             if isinstance(recommendations_selected, list) and recommendations_selected and isinstance(recommendations_selected[0], tuple):
                 print("Rank | Title                     | IMDb ID")
                 print("---- | ------------------------- | ---------")
                 for i, (title, imdb_id) in enumerate(recommendations_selected):
                     link = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else "N/A"
                     print(f"{i+1:<4} | {title:<25} | {link}")
             else:
                 print("Failed to get recommendations for selected title.")

        elif recommendations_fuzzy is None:
             print("No recommendations found for fuzzy query.")
    else:
        print("\nSkipping PlotRecommender example because metadata failed to load.")