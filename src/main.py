import os
import sys
import argparse

# Add the src directory to the Python path to allow absolute imports
# This makes the script runnable from the project root (Movie-Recommender/)
# using 'python src/main.py'
project_root = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory (Movie-Recommender)
sys.path.insert(0, os.path.dirname(project_root))

try:
    from src.data_loader import load_metadata, load_ratings
    from src.recommenders.plot_recommender import PlotRecommender
    from src.recommenders.simple_recommender import SimpleRecommender
    from src.recommenders.association_recommender import AssociationRecommender
    from src.recommenders.keyword_recommender import KeywordRecommender # Added
    from src.ui.cli import run_ui
    from src.utils.dataset_stats import analyze_and_display
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you are running this script from the project root directory")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root added to sys.path: {os.path.dirname(project_root)}")
    sys.exit(1)

# --- Configuration ---
DATA_DIRECTORY = 'dataset'  # Relative path from project root


def setup_parser():
    """
    Set up command-line argument parser with subparsers for each recommender type.

    Returns:
        argparse.ArgumentParser: The configured parser
    """
    parser = argparse.ArgumentParser(description='Movie Recommender System')

    # Global options that apply to all recommenders
    parser.add_argument('--use-name-removed', action='store_true', default=True,
                        help='Use the version of metadata with person names removed from overviews (default)')
    parser.add_argument('--use-original', action='store_true',
                        help='Use the original metadata without removing person names from overviews')

    # Create subparsers for each recommender type
    subparsers = parser.add_subparsers(
        dest='recommender', help='Recommender type to use')

    # Default recommender if none specified
    parser.set_defaults(recommender='plot')

    # Plot Recommender (content-based)
    plot_parser = subparsers.add_parser(
        'plot', help='Content-based recommender using plot similarity')
    plot_parser.add_argument('--hybrid', action='store_true',
                             help='Use hybrid plot recommender with weighted rating integration')
    plot_parser.add_argument('--similarity-weight', type=float, default=0.7,
                             help='Weight for plot similarity in hybrid mode (0-1), default: 0.7')
    plot_parser.add_argument('--percentile', type=float, default=0.90,
                             help='Vote count percentile threshold for weighted ratings (default: 0.90)')

    # Simple Recommender (popularity-based)
    simple_parser = subparsers.add_parser(
        'simple', help='Simple recommender based on weighted ratings')
    simple_parser.add_argument('--percentile', type=float, default=0.90,
                               help='Vote count percentile threshold (default: 0.90)')

    # Association Recommender (collaborative filtering)
    assoc_parser = subparsers.add_parser(
        'association', help='Association rule mining recommender')
    assoc_parser.add_argument('--min-support', type=float, default=0.04,
                              help='Minimum support threshold for association rules (default: 0.04)')
    assoc_parser.add_argument('--min-confidence', type=float, default=0.3,
                              help='Minimum confidence for association rules (default: 0.3)')
    assoc_parser.add_argument('--min-lift', type=float, default=1.2,
                              help='Minimum lift for association rules (default: 1.2)')
    assoc_parser.add_argument('--rating-threshold', type=float, default=4,
                              help='Minimum rating to consider a movie as liked (default: 4)')

    # Dataset Statistics (utility)
    stats_parser = subparsers.add_parser(
        'stats', help='Show dataset statistics')

    # Keyword Recommender
    keyword_parser = subparsers.add_parser(
        'keyword', help='Keyword-based recommender using keyword relevance and IMDB scores')
    keyword_parser.add_argument('--alpha', type=float, default=0.7,
                                help='Weight for keyword relevance score (0-1), default: 0.7')
    keyword_parser.add_argument('--beta', type=float, default=0.3,
                                help='Weight for IMDB weighted score (0-1), default: 0.3')
    keyword_parser.add_argument('--kw-vote-percentile', type=float, default=0.85,
                                help='Vote count percentile for IMDB score calculation within keyword recommender (default: 0.85)')

    return parser


# --- Main Execution ---
if __name__ == "__main__":
    # Parse command line arguments
    parser = setup_parser()
    args = parser.parse_args()

    print("--- Movie Recommender System ---")
    print(f"Mode: {args.recommender.capitalize()}")

    # Handle metadata version selection
    use_name_removed = True  # Default
    if hasattr(args, 'use_original') and args.use_original:
        use_name_removed = False
        print("Using original metadata with person names in plot overviews")
    else:
        print("Using metadata with person names removed from plot overviews (default)")

    print(f"Current working directory: {os.getcwd()}")
    print(f"Loading data from: {os.path.abspath(DATA_DIRECTORY)}")

    # 1. Load Data
    metadata_df = load_metadata(
        DATA_DIRECTORY, use_name_removed=use_name_removed)

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
    elif args.recommender == 'keyword':
        recommender = KeywordRecommender(
            alpha=args.alpha,
            beta=args.beta,
            vote_count_percentile=args.kw_vote_percentile
        )
        # KeywordRecommender needs metadata_df and path to keywords.csv
        fit_successful = recommender.fit(metadata_df, keywords_data_path=os.path.join(DATA_DIRECTORY, 'keywords.csv'))
        if not fit_successful:
            print("\nFatal Error: KeywordRecommender fitting failed. Exiting.")
            sys.exit(1)
    else:  # Default to plot recommender (or if 'plot' is explicitly chosen)
        if hasattr(args, 'hybrid') and args.hybrid: # Check if args has 'hybrid'
            print(
                f"Initializing Hybrid Plot Recommender (similarity weight: {args.similarity_weight}, vote percentile: {args.percentile})")
            recommender = PlotRecommender(
                similarity_weight=args.similarity_weight,
                vote_count_percentile=args.percentile
            )
        else:
            print("Initializing standard Plot Recommender")
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
