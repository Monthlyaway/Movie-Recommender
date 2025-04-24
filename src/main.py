import os
import sys
import argparse

# Add the src directory to the Python path to allow absolute imports
# This makes the script runnable from the project root (Movie-Recommender/)
# using 'python src/main.py'
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(project_root)) # Add the parent directory (Movie-Recommender)

try:
    from src.data_loader import load_metadata
    from src.recommenders.plot_recommender import PlotRecommender
    from src.ui.cli import run_ui
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you are running this script from the project root directory")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root added to sys.path: {os.path.dirname(project_root)}")
    sys.exit(1)

# --- Configuration ---
DATA_DIRECTORY = 'dataset' # Relative path from project root

# --- Main Execution ---
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Movie Recommender System')
    parser.add_argument('--use-name-removed', action='store_true',
                       help='Use the version of metadata with person names removed from overviews')
    args = parser.parse_args()

    print("--- Movie Recommender System ---")
    if args.use_name_removed:
        print("Using metadata with person names removed from plot overviews")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Loading data from: {os.path.abspath(DATA_DIRECTORY)}")

    # 1. Load Data
    metadata_df = load_metadata(DATA_DIRECTORY, use_name_removed=args.use_name_removed)

    if metadata_df.empty:
        print("\nFatal Error: Could not load metadata. Exiting.")
        sys.exit(1)

    # 2. Initialize and Fit Recommender (Plot-based for now)
    # TODO: Add logic here later to select different recommenders
    print("\nInitializing Plot Recommender...")
    plot_recommender = PlotRecommender()
    try:
        plot_recommender.fit(metadata_df)
        print("Recommender fitting successful.")
    except Exception as e:
        print(f"\nFatal Error: Failed to fit the recommender: {e}")
        # Optionally log the full traceback
        sys.exit(1)


    # 3. Run Command-Line Interface
    print("\nStarting Command-Line Interface...")
    try:
        run_ui(plot_recommender)
    except Exception as e:
        print(f"\nFatal Error: An error occurred during UI execution: {e}")
        # Optionally log the full traceback
        sys.exit(1)

    print("\n--- Exiting Movie Recommender System ---")