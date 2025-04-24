"""
Dataset Statistics Utility Module

This module provides functions to analyze movie dataset statistics
and display them in a formatted way.
"""

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import ast
from collections import Counter

console = Console()

def extract_genres(metadata_df):
    """
    Extract and count genres from the metadata DataFrame.
    
    Args:
        metadata_df: DataFrame containing movie metadata with genres column
        
    Returns:
        dict: A counter of genre occurrences
    """
    all_genres = []
    
    for genres_str in metadata_df['genres'].dropna():
        try:
            # Parse the genres string which is in JSON format
            genres_list = ast.literal_eval(genres_str)
            # Extract genre names
            for genre in genres_list:
                if isinstance(genre, dict) and 'name' in genre:
                    all_genres.append(genre['name'])
        except (ValueError, SyntaxError):
            # Skip malformed genre entries
            continue
            
    return Counter(all_genres)

def analyze_dataset(metadata_df, ratings_df=None):
    """
    Analyze the movie dataset and return statistics.
    
    Args:
        metadata_df: DataFrame containing movie metadata
        ratings_df: Optional DataFrame containing user ratings
        
    Returns:
        dict: Various statistics about the movie dataset
    """
    stats = {}
    
    # Basic metadata stats
    stats['total_movies'] = len(metadata_df)
    stats['avg_vote'] = metadata_df['vote_average'].mean()
    stats['median_vote'] = metadata_df['vote_average'].median()
    stats['min_vote'] = metadata_df['vote_average'].min()
    stats['max_vote'] = metadata_df['vote_average'].max()
    
    # Vote count stats
    stats['avg_vote_count'] = metadata_df['vote_count'].mean()
    stats['median_vote_count'] = metadata_df['vote_count'].median()
    stats['max_vote_count'] = metadata_df['vote_count'].max()
    
    # Release years
    if 'release_date' in metadata_df.columns:
        try:
            metadata_df['release_year'] = pd.to_datetime(metadata_df['release_date'], errors='coerce').dt.year
            valid_years = metadata_df['release_year'].dropna()
            stats['earliest_year'] = int(valid_years.min()) if not valid_years.empty else None
            stats['latest_year'] = int(valid_years.max()) if not valid_years.empty else None
            stats['most_common_year'] = valid_years.value_counts().idxmax() if not valid_years.empty else None
        except:
            stats['year_stats_error'] = "Could not parse release dates"
    
    # Runtime stats
    if 'runtime' in metadata_df.columns:
        valid_runtimes = metadata_df['runtime'].dropna()
        stats['avg_runtime'] = valid_runtimes.mean()
        stats['min_runtime'] = valid_runtimes.min()
        stats['max_runtime'] = valid_runtimes.max()
    
    # Genre stats
    try:
        genre_counts = extract_genres(metadata_df)
        stats['top_genres'] = genre_counts.most_common(10)
        stats['total_genres'] = len(genre_counts)
    except:
        stats['genre_stats_error'] = "Could not parse genres"
    
    # Ratings stats (if available)
    if ratings_df is not None and not ratings_df.empty:
        stats['total_ratings'] = len(ratings_df)
        stats['unique_users'] = ratings_df['userId'].nunique()
        stats['avg_rating'] = ratings_df['rating'].mean()
        stats['rating_distribution'] = ratings_df['rating'].value_counts().sort_index().to_dict()
    
    return stats

def display_stats(stats, include_ratings=False):
    """
    Display dataset statistics in a formatted way using Rich.
    
    Args:
        stats: Dictionary of statistics
        include_ratings: Whether to include ratings statistics
    """
    console.print(Panel.fit("[bold magenta]Movie Dataset Statistics[/bold magenta]", border_style="cyan"))
    
    # Basic stats table
    basic_table = Table(title="Basic Statistics", show_header=True, header_style="bold green")
    basic_table.add_column("Statistic", style="dim")
    basic_table.add_column("Value")
    
    basic_table.add_row("Total Movies", f"{stats['total_movies']:,}")
    
    if 'earliest_year' in stats and stats['earliest_year'] is not None:
        basic_table.add_row("Year Range", f"{stats['earliest_year']} - {stats['latest_year']}")
        basic_table.add_row("Most Common Year", f"{stats['most_common_year']}")
    
    basic_table.add_row("Average Vote", f"{stats['avg_vote']:.2f} / 10")
    basic_table.add_row("Median Vote", f"{stats['median_vote']:.2f} / 10")
    basic_table.add_row("Vote Range", f"{stats['min_vote']:.1f} - {stats['max_vote']:.1f}")
    
    basic_table.add_row("Average Vote Count", f"{stats['avg_vote_count']:.1f}")
    basic_table.add_row("Median Vote Count", f"{stats['median_vote_count']:.1f}")
    basic_table.add_row("Maximum Vote Count", f"{stats['max_vote_count']:,}")
    
    if 'avg_runtime' in stats:
        basic_table.add_row("Average Runtime", f"{stats['avg_runtime']:.1f} minutes")
        basic_table.add_row("Runtime Range", f"{stats['min_runtime']:.0f} - {stats['max_runtime']:.0f} minutes")
    
    console.print(basic_table)
    
    # Top genres table
    if 'top_genres' in stats:
        genre_table = Table(title="Top 10 Genres", show_header=True, header_style="bold blue")
        genre_table.add_column("Rank", style="dim")
        genre_table.add_column("Genre")
        genre_table.add_column("Movie Count")
        genre_table.add_column("Percentage", justify="right")
        
        for i, (genre, count) in enumerate(stats['top_genres'], 1):
            percentage = count / stats['total_movies'] * 100
            genre_table.add_row(
                str(i), 
                genre, 
                f"{count:,}",
                f"{percentage:.1f}%"
            )
        
        console.print(genre_table)
    
    # Ratings stats (if available)
    if include_ratings and 'total_ratings' in stats:
        ratings_table = Table(title="Ratings Statistics", show_header=True, header_style="bold yellow")
        ratings_table.add_column("Statistic", style="dim")
        ratings_table.add_column("Value")
        
        ratings_table.add_row("Total Ratings", f"{stats['total_ratings']:,}")
        ratings_table.add_row("Unique Users", f"{stats['unique_users']:,}")
        ratings_table.add_row("Average Rating", f"{stats['avg_rating']:.2f} / 5")
        
        console.print(ratings_table)
        
        # Rating distribution
        if 'rating_distribution' in stats:
            dist_table = Table(title="Rating Distribution", show_header=True, header_style="bold red")
            dist_table.add_column("Rating")
            dist_table.add_column("Count")
            dist_table.add_column("Percentage", justify="right")
            
            for rating, count in sorted(stats['rating_distribution'].items()):
                percentage = count / stats['total_ratings'] * 100
                dist_table.add_row(
                    f"{rating:.1f}", 
                    f"{count:,}",
                    f"{percentage:.1f}%"
                )
            
            console.print(dist_table)
    
    console.print("\n[bold cyan]Analysis Complete![/bold cyan]")

def analyze_and_display(metadata_df, ratings_df=None):
    """
    Analyze the dataset and display statistics.
    
    Args:
        metadata_df: DataFrame containing movie metadata
        ratings_df: Optional DataFrame containing user ratings
    """
    console.print("[bold]Analyzing movie dataset...[/bold]")
    stats = analyze_dataset(metadata_df, ratings_df)
    display_stats(stats, include_ratings=ratings_df is not None)

if __name__ == "__main__":
    # Example usage
    import os
    import sys
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    try:
        from src.data_loader import load_metadata, load_ratings
        
        # Load data
        metadata_df = load_metadata()
        ratings_df = load_ratings()
        
        if not metadata_df.empty:
            analyze_and_display(metadata_df, ratings_df)
        else:
            print("Failed to load metadata.")
    except ImportError:
        print("Error importing modules. Run this script from the project root directory.")
        sys.exit(1) 