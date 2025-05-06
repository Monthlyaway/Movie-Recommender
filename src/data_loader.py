import pandas as pd
import os
import numpy as np # For handling potential NaN during merge

def load_metadata(data_dir: str = 'dataset', use_name_removed: bool = False) -> pd.DataFrame:
    """
    Loads and merges movies_metadata.csv (or movies_metadata_name_removed.csv) and links_small.csv.

    Args:
        data_dir: The directory containing the dataset files.
        use_name_removed: Whether to use the version with person names removed from overviews.

    Returns:
        A pandas DataFrame containing merged movie metadata including titles,
        overviews, and IMDb IDs. Returns an empty DataFrame on error.
    """
    metadata_filename = 'movies_metadata_name_removed.csv' if use_name_removed else 'movies_metadata.csv'
    metadata_path = os.path.join(data_dir, metadata_filename)
    links_path = os.path.join(data_dir, 'links.csv') # Or links.csv if using full dataset

    try:
        print(f"Attempting to load metadata from: {os.path.abspath(metadata_path)}")
        metadata = pd.read_csv(metadata_path, low_memory=False)
        print("Metadata loaded successfully.")

        # --- Data Cleaning for Metadata ---
        # Fill NaN overviews
        metadata['overview'] = metadata['overview'].fillna('')
        # Convert 'id' to numeric, coercing errors to NaN, then drop rows with invalid IDs
        metadata['id'] = pd.to_numeric(metadata['id'], errors='coerce')
        metadata.dropna(subset=['id'], inplace=True)
        # Convert 'id' to integer (required for merging with links tmdbId)
        metadata['id'] = metadata['id'].astype(int)

        print(f"Attempting to load links from: {os.path.abspath(links_path)}")
        links = pd.read_csv(links_path)
        print("Links loaded successfully.")

        # --- Data Cleaning for Links ---
        # Convert tmdbId to numeric, coercing errors to NaN, then drop rows with invalid IDs
        links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')
        links.dropna(subset=['tmdbId'], inplace=True)
        # Convert tmdbId to integer
        links['tmdbId'] = links['tmdbId'].astype(int)

        # --- Merge DataFrames ---
        print("Merging metadata and links...")
        # Merge metadata with links based on the TMDB ID
        # Use left merge to keep all movies from metadata, even if no link exists
        merged_df = pd.merge(metadata, links, left_on='id', right_on='tmdbId', how='left')

        # --- Create Full IMDb ID ---
        # imdbId from links might be NaN after merge, or just numbers. Handle both.
        # Pad with leading zeros to 7 digits and prepend 'tt'
        def format_imdb_id(imdb_id):
            if pd.isna(imdb_id):
                return None
            try:
                # Convert to int first to remove potential '.0', then format
                return f"tt{int(imdb_id):07d}"
            except (ValueError, TypeError):
                return None # Handle cases where conversion fails

        merged_df['imdb_id_full'] = merged_df['imdbId'].apply(format_imdb_id)

        # Drop redundant columns if desired (optional)
        # merged_df.drop(columns=['movieId', 'imdbId', 'tmdbId'], inplace=True)

        print("Data loading and merging complete.")
        return merged_df

    except FileNotFoundError as e:
        print(f"Error: File not found. {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred during data loading/merging: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed error traceback
        return pd.DataFrame()

def load_ratings(data_dir: str = 'dataset', use_small: bool = True) -> pd.DataFrame:
    """
    Loads ratings data from either ratings_small.csv or ratings.csv.
    
    Args:
        data_dir: The directory containing the dataset files.
        use_small: Whether to use the smaller ratings dataset (default: True)
        
    Returns:
        A pandas DataFrame containing user ratings.
    """
    file_name = 'ratings_small.csv' if use_small else 'ratings.csv'
    ratings_path = os.path.join(data_dir, file_name)
    
    try:
        print(f"Attempting to load ratings from: {os.path.abspath(ratings_path)}")
        ratings = pd.read_csv(ratings_path)
        print("Ratings loaded successfully.")
        return ratings
    except FileNotFoundError:
        print(f"Error: Ratings file not found at {ratings_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred loading ratings: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    print(f"Current working directory: {os.getcwd()}")
    # Test metadata loading
    df = load_metadata()
    if not df.empty:
        print("\nMerged DataFrame Info:")
        df.info()
        print("\nFirst 5 rows of merged data:")
        # Show relevant columns
        print(df[['id', 'title', 'overview', 'imdbId', 'imdb_id_full']].head())
        print(f"\nShape: {df.shape}")
        print(f"\nOverview column NaN count: {df['overview'].isnull().sum()}")
        print(f"\nimdb_id_full column NaN count: {df['imdb_id_full'].isnull().sum()}")
        # Check a known movie
        print("\nExample lookup (Toy Story):")
        toy_story = df[df['title'] == 'Toy Story']
        if not toy_story.empty:
            print(toy_story[['id', 'title', 'imdb_id_full']].iloc[0])
        else:
            print("Toy Story not found in merged data.")
    else:
        print("\nFailed to load and merge data.")
        
    # Test ratings loading
    ratings_df = load_ratings()
    if not ratings_df.empty:
        print("\nRatings DataFrame Info:")
        ratings_df.info()
        print("\nFirst 5 rows of ratings data:")
        print(ratings_df.head())
        print(f"\nShape: {ratings_df.shape}")
        print(f"\nUnique users: {ratings_df['userId'].nunique()}")
        print(f"\nUnique movies: {ratings_df['movieId'].nunique()}")
        print(f"\nRating distribution:\n{ratings_df['rating'].value_counts().sort_index()}")
    else:
        print("\nFailed to load ratings data.")