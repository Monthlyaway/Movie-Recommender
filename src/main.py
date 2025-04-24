import os
import sys
import argparse

# Add the src directory to the Python path to allow absolute imports
# This makes the script runnable from the project root (Movie-Recommender/)
# using 'python src/main.py'
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(project_root)) # Add the parent directory (Movie-Recommender)

try:
    from src.data_loader import load_metadata, load_ratings
    from src.recommenders.plot_recommender import PlotRecommender
    from src.recommenders.simple_recommender import SimpleRecommender
    from src.recommenders.association_recommender import AssociationRecommender
    from src.ui.cli import run_ui
    from src.utils.dataset_stats import analyze_and_display
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you are running this script from the project root directory")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root added to sys.path: {os.path.dirname(project_root)}")
    sys.exit(1)

# --- Configuration ---
DATA_DIRECTORY = 'dataset' # Relative path from project root

def setup_parser():
    """
    Set up command-line argument parser with subparsers for each recommender type.
    
    Returns:
        argparse.ArgumentParser: The configured parser
    """
    parser = argparse.ArgumentParser(description='Movie Recommender System')
    
    # Global options that apply to all recommenders
    parser.add_argument('--use-name-removed', action='store_true',
                      help='Use the version of metadata with person names removed from overviews')
    
    # Create subparsers for each recommender type
    subparsers = parser.add_subparsers(dest='recommender', help='Recommender type to use')
    
    # Default recommender if none specified
    parser.set_defaults(recommender='plot')
    
    # Plot Recommender (content-based)
    plot_parser = subparsers.add_parser('plot', help='Content-based recommender using plot similarity')
    
    # Simple Recommender (popularity-based)
    simple_parser = subparsers.add_parser('simple', help='Simple recommender based on weighted ratings')
    simple_parser.add_argument('--percentile', type=float, default=0.90,
                             help='Vote count percentile threshold (default: 0.90)')
    
    # Association Recommender (collaborative filtering)
    assoc_parser = subparsers.add_parser('association', help='Association rule mining recommender')
    assoc_parser.add_argument('--min-support', type=float, default=0.06,
                            help='Minimum support threshold for association rules (default: 0.06)')
    assoc_parser.add_argument('--min-confidence', type=float, default=0.3,
                            help='Minimum confidence for association rules (default: 0.3)')
    assoc_parser.add_argument('--min-lift', type=float, default=1.2,
                            help='Minimum lift for association rules (default: 1.2)')
    assoc_parser.add_argument('--rating-threshold', type=float, default=6,
                            help='Minimum rating to consider a movie as liked (default: 6)')
    
    # Dataset Statistics (utility)
    stats_parser = subparsers.add_parser('stats', help='Show dataset statistics')
    
    return parser

# --- Main Execution ---
if __name__ == "__main__":
    # Parse command line arguments
    parser = setup_parser()
    args = parser.parse_args()

    print("--- Movie Recommender System ---")
    print(f"Mode: {args.recommender.capitalize()}")
    if args.use_name_removed:
        print("Using metadata with person names removed from plot overviews")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Loading data from: {os.path.abspath(DATA_DIRECTORY)}")

    # 1. Load Data
    metadata_df = load_metadata(DATA_DIRECTORY, use_name_removed=args.use_name_removed)

    if metadata_df.empty:
        print("\nFatal Error: Could not load metadata. Exiting.")
        sys.exit(1)
        
    # Special case: if we're just showing stats, do that and exit
    if args.recommender == 'stats':
        # Load ratings data too for more complete stats
        ratings_df = load_ratings(DATA_DIRECTORY)
        analyze_and_display(metadata_df, ratings_df)
        sys.exit(0)

    # 2. Initialize and Fit Selected Recommender
    print(f"\nInitializing {args.recommender.capitalize()} Recommender...")
    
    if args.recommender == 'simple':
        recommender = SimpleRecommender(vote_count_percentile=args.percentile)
        recommender.fit(metadata_df)
    elif args.recommender == 'association':
        # Load ratings data (additional data needed for this recommender)
        ratings_df = load_ratings(DATA_DIRECTORY)
        if ratings_df.empty:
            print("\nFatal Error: Could not load ratings data. Exiting.")
            sys.exit(1)
        
        recommender = AssociationRecommender(
            min_support=args.min_support,
            min_confidence=args.min_confidence,
            min_lift=args.min_lift,
            rating_threshold=args.rating_threshold
        )
        # Association recommender needs both metadata and ratings
        result = recommender.fit(metadata_df, ratings_df)
        
        # Check if fitting was successful
        if result is None:
            print("\nFatal Error: AssociationRecommender fitting failed. Exiting.")
            print("Try adjusting the parameters as suggested above.")
            sys.exit(1)
    else:  # Default to plot recommender
        recommender = PlotRecommender()
        recommender.fit(metadata_df)
        
    print("Recommender fitting successful.")

    # 3. Run Command-Line Interface
    print("\nStarting Command-Line Interface...")
    try:
        run_ui(recommender)
    except Exception as e:
        print(f"\nFatal Error: An error occurred during UI execution: {e}")
        # Optionally log the full traceback
        sys.exit(1)

    print("\n--- Exiting Movie Recommender System ---")