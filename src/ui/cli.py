from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
import os
from typing import List, Optional, Union, Tuple # Import necessary types

# Assuming PlotRecommender is the only one for now
# from ..recommenders.plot_recommender import PlotRecommender # Relative import might fail if run directly

console = Console()

def select_movie_from_matches(matches: List[str]) -> Optional[str]:
    """
    Prompts the user to select a movie from a list of fuzzy matches.

    Args:
        matches: A list of potential movie titles.

    Returns:
        The selected movie title, or None if the user cancels or input is invalid.
    """
    console.print("\n[yellow]Multiple possible matches found. Please select one:[/yellow]")
    choices = [str(i + 1) for i in range(len(matches))] + ["c"] # Add 'c' for cancel

    table = Table(title="Potential Matches", show_header=True, header_style="bold cyan")
    table.add_column("Option", style="dim", width=6)
    table.add_column("Movie Title")

    for i, title in enumerate(matches):
        table.add_row(str(i + 1), title)
    table.add_row("c", "[italic]Cancel search[/italic]")

    console.print(table)

    selection = Prompt.ask("Enter the number of the movie you meant, or 'c' to cancel", choices=choices)

    if selection.lower() == 'c':
        return None
    try:
        selected_index = int(selection) - 1
        if 0 <= selected_index < len(matches):
            return matches[selected_index]
        else:
            console.print("[red]Invalid selection number.[/red]")
            return None
    except ValueError:
        console.print("[red]Invalid input.[/red]")
        return None

def display_association_rules(recommender):
    """
    Displays random frequent itemsets from the recommender.
    
    Args:
        recommender: An initialized and fitted AssociationRecommender
    """
    if not hasattr(recommender, 'get_random_rules'):
        console.print("[bold red]This recommender doesn't support itemset display.[/bold red]")
        return
    
    try:
        # Ask user how many itemsets to display
        try:
            num_itemsets = int(Prompt.ask("\nHow many random itemsets to display?", default="5"))
            if num_itemsets <= 0:
                console.print("[bold red]Please enter a positive number.[/bold red]")
                return
        except ValueError:
            console.print("[bold red]Invalid number.[/bold red]")
            return
            
        # Get random itemsets
        itemsets = recommender.get_random_rules(num_itemsets)
        if not itemsets:
            console.print("[bold yellow]No itemsets available to display.[/bold yellow]")
            return
            
        # Display itemsets in a table
        console.print(f"\n[bold green]Displaying {len(itemsets)} Random Frequent Itemsets:[/bold green]")
        
        for i, itemset in enumerate(itemsets):
            # Create a nicely formatted panel for each itemset
            movies_str = ", ".join(itemset['movies'])
            
            panel_content = (
                f"[bold cyan]Itemset #{i+1}:[/bold cyan]\n\n"
                f"[bold]Movies that frequently appear together:[/bold]\n"
                f"[yellow]{movies_str}[/yellow]\n\n"
                f"[dim]Support: {itemset['support']:.4f}[/dim]"
            )
            
            panel = Panel(
                panel_content,
                title=f"[bold]Frequent Itemset {i+1}/{len(itemsets)}[/bold]",
                border_style="blue"
            )
            
            console.print(panel)
            
            # Ask to see more itemsets if not the last one
            if i < len(itemsets) - 1:
                see_more = Prompt.ask("See next itemset?", choices=["y", "n"], default="y")
                if see_more.lower() == "n":
                    break
                    
        console.print("\n[bold green]End of itemsets display.[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error displaying itemsets: {e}[/bold red]")


def run_ui(recommender):
    """
    Runs the command-line interface for the movie recommender.

    Args:
        recommender: An initialized and fitted recommender object.
                     Must have a 'recommend(title, top_n)' method that returns:
                     - List[Tuple[str, Optional[str]]] for exact match recommendations.
                     - List[str] for potential fuzzy matches.
                     - None if no match found.
    """
    console.print("[bold cyan]Welcome to the Movie Recommender![/bold cyan]")
    
    # Try to import the dataset stats module
    try:
        from src.utils.dataset_stats import analyze_and_display
        from src.data_loader import load_ratings
        has_stats_module = True
    except ImportError:
        has_stats_module = False
    
    # Detect which type of recommender we're using
    recommender_name = recommender.__class__.__name__
    is_simple_recommender = recommender_name == 'SimpleRecommender'
    is_association_recommender = recommender_name == 'AssociationRecommender'
    is_content_based = not (is_simple_recommender or is_association_recommender)
    
    if is_simple_recommender:
        console.print(f"[bold]Using Simple Movie Recommender[/bold] (IMDB Weighted Rating Formula)")
        if hasattr(recommender, 'get_details'):
            details = recommender.get_details()
            console.print(f"Minimum votes threshold: {details.get('minimum_vote_threshold', 'N/A'):.0f}")
            console.print(f"Mean rating across all movies: {details.get('mean_vote_threshold', 'N/A'):.2f}")
            console.print(f"Total qualified movies: {details.get('qualified_movies_count', 'N/A')}")
    elif is_association_recommender:
        console.print(f"[bold]Using Association Rule Mining Recommender[/bold]")
        if hasattr(recommender, 'get_details'):
            details = recommender.get_details()
            console.print(f"Min support: {details.get('min_support', 'N/A')}")
            console.print(f"Min confidence: {details.get('min_confidence', 'N/A')}")
            console.print(f"Min lift: {details.get('min_lift', 'N/A')}")
            console.print(f"Rating threshold: {details.get('rating_threshold', 'N/A')}")
            console.print(f"Number of rules: {details.get('rules_count', 'N/A')}")
    else:
        # Check if it's a hybrid plot recommender
        is_hybrid = hasattr(recommender, 'get_details') and 'hybrid' in recommender.get_details().get('type', '')
        
        if is_hybrid:
            console.print(f"[bold]Using Hybrid Plot-Based Movie Recommender[/bold]")
            details = recommender.get_details()
            console.print(f"Plot similarity weight: {details.get('similarity_weight', 'N/A'):.2f}")
            console.print(f"Weighted rating weight: {details.get('weighted_rating_weight', 'N/A'):.2f}")
            console.print(f"Mean rating (C): {details.get('mean_vote_threshold', 'N/A'):.2f}")
            console.print(f"Minimum votes (m): {details.get('minimum_vote_threshold', 'N/A'):.0f}")
        else:
            console.print(f"[bold]Using Plot-Based Movie Recommender[/bold]")
    
    # User preferences
    show_plots = True

    while True:
        # Show options menu
        console.print("\n[bold]Options:[/bold]")
        
        if is_simple_recommender:
            console.print("1. View top rated movies")
            choices = ["1", "3", "4"] if has_stats_module else ["1", "3"]
        elif is_content_based or is_association_recommender:
            if is_content_based:  # Only Plot recommender has plot details
                console.print(f"1. Show plots: {'[green]ON[/green]' if show_plots else '[red]OFF[/red]'}")
            else:
                console.print("1. Display random association rules")
            console.print("2. Search for a movie")
            choices = ["1", "2", "3", "4"] if has_stats_module else ["1", "2", "3"]
        
        # Add dataset statistics option if available
        if has_stats_module:
            console.print("3. View dataset statistics")
            console.print("4. Quit")
        else:
            console.print("3. Quit")
        
        default_choice = "1" if is_simple_recommender else "2"
        choice = Prompt.ask("\n[bold]Select an option[/bold]", choices=choices, default=default_choice)
        
        if choice == "1":
            if is_simple_recommender:
                # Simple recommender: get top movies
                try:
                    top_n = Prompt.ask("\nHow many top movies to display?", default="10")
                    top_n = int(top_n)
                    if top_n <= 0:
                        console.print("[bold red]Please enter a positive number.[/bold red]")
                        continue
                        
                    console.print(f"\nFetching top {top_n} movies by weighted rating...")
                    result = recommender.recommend(top_n=top_n)
                    
                    if result:
                        # Unpack results and parameters if returned in new format
                        if isinstance(result, tuple) and len(result) == 2:
                            recommendations, params = result
                        else:
                            recommendations = result
                            params = None
                        
                        # Display results table
                        table = Table(title=f"Top {top_n} Movies by Weighted Rating", show_header=True, header_style="bold magenta")
                        table.add_column("Rank", style="dim", width=6)
                        table.add_column("Movie Title")
                        table.add_column("IMDb Link")
                        
                        # Check if we have additional score data (weighted_score, avg_user_rating)
                        has_detailed_scores = len(recommendations[0]) > 2
                        if has_detailed_scores:
                            table.add_column("Weighted Score (1-10)", width=18)
                            table.add_column("User Rating (1-10)", width=18)
                        
                        for i, rec in enumerate(recommendations):
                            # Process based on the format of recommendation results
                            if has_detailed_scores:
                                # Unpack all data including scores
                                title, imdb_id, weighted_score, avg_user_rating = rec
                                
                                # Format scores for display
                                score_display = f"[bold cyan]{weighted_score:.2f}[/bold cyan]"
                                
                                # Color-code user ratings
                                if avg_user_rating > 8:
                                    rating_display = f"[green]{avg_user_rating:.1f}[/green]"
                                elif avg_user_rating < 4 and avg_user_rating > 0:
                                    rating_display = f"[red]{avg_user_rating:.1f}[/red]"
                                elif avg_user_rating == 0:
                                    rating_display = "[dim]No ratings[/dim]"
                                else:
                                    rating_display = f"{avg_user_rating:.1f}"
                                
                                # Create the table row with all columns
                                link = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else "[dim]N/A[/dim]"
                                table.add_row(str(i + 1), title, link, score_display, rating_display)
                            else:
                                # Original format with just title and IMDb ID
                                title, imdb_id = rec
                                link = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else "[dim]N/A[/dim]"
                                table.add_row(str(i + 1), title, link)
                            
                        console.print(table)
                        
                        # Display Weighted Rating Parameters if available
                        if params and 'C' in params and 'm' in params:
                            params_table = Table(title="Weighted Rating Parameters", show_header=True, header_style="bold cyan")
                            params_table.add_column("Parameter", width=30)
                            params_table.add_column("Value", width=15)
                            params_table.add_row("Global Mean Rating (C)", f"{params['C']}")
                            params_table.add_row("Minimum Votes Threshold (m)", f"{params['m']}")
                            if 'rating_scale' in params:
                                params_table.add_row("Original User Rating Scale", f"1-{params['rating_scale']}")
                            params_table.add_row("Display Scale", "1-10 (normalized)")
                            console.print(params_table)
                    else:
                        console.print("[bold red]No results returned from recommender.[/bold red]")
                    
                except ValueError:
                    console.print("[bold red]Please enter a valid number.[/bold red]")
                except Exception as e:
                    console.print(f"[bold red]An error occurred: {e}[/bold red]")
            elif is_content_based:
                # Plot recommender: toggle plot display
                show_plots = not show_plots
                console.print(f"Plot display is now {'[green]ON[/green]' if show_plots else '[red]OFF[/red]'}")
            elif is_association_recommender:
                # Association recommender: display random rules
                display_association_rules(recommender)
            continue
        elif choice == "3" and has_stats_module:
            # Show dataset statistics
            console.print("\n[bold]Loading dataset statistics...[/bold]")
            try:
                # Get the metadata from the recommender if possible
                if hasattr(recommender, 'metadata'):
                    metadata_df = recommender.metadata
                    # Try to load ratings data
                    ratings_df = load_ratings()
                    analyze_and_display(metadata_df, ratings_df)
                else:
                    console.print("[bold red]Cannot access dataset. Try running 'python src/main.py stats' instead.[/bold red]")
            except Exception as e:
                console.print(f"[bold red]Error displaying statistics: {e}[/bold red]")
            continue
        elif (choice == "3" and not has_stats_module) or (choice == "4" and has_stats_module):
            console.print("[bold yellow]Goodbye![/bold yellow]")
            break
        
        # Option 2: Only relevant for plot and association recommenders
        if is_simple_recommender:
            console.print("[bold yellow]Invalid option for the Simple Recommender.[/bold yellow]")
            continue
            
        # Search for a movie with the plot or association recommender
        movie_title_input = Prompt.ask("\n[bold green]Enter a movie title to get recommendations[/bold green]")

        if not movie_title_input:
            console.print("[bold red]Please enter a movie title.[/bold red]")
            continue

        console.print(f"\nSearching for '[italic]{movie_title_input}[/italic]'...")
        try:
            # Initial recommendation call
            result = recommender.recommend(movie_title_input, top_n=10)
            
            # Track the actual title we'll use (handles fuzzy match cases)
            actual_title = movie_title_input

            # --- Handle Different Result Types ---

            # Case 1: Fuzzy matches returned, need user selection
            if isinstance(result, list) and result and isinstance(result[0], str):
                selected_title = select_movie_from_matches(result)
                if selected_title:
                    console.print(f"\nFetching recommendations for selected title '[italic]{selected_title}[/italic]'...")
                    # Call recommend again with the selected title (should be an exact match now)
                    result = recommender.recommend(selected_title, top_n=10)
                    actual_title = selected_title
                else:
                    console.print("[yellow]Search cancelled.[/yellow]")
                    result = None # Treat cancellation as no result found for display logic

            # Case 2: Exact match recommendations returned (or after selection)
            if isinstance(result, list) and result and isinstance(result[0], tuple):
                try:
                    # For content-based recommender with plot access
                    if is_content_based and hasattr(recommender, 'indices') and hasattr(recommender, 'metadata'):
                        # Get original movie details
                        original_movie_idx = recommender.indices[actual_title]
                        original_movie_data = recommender.metadata.iloc[original_movie_idx]
                        original_movie_title = original_movie_data['title']
                        original_movie_plot = original_movie_data['overview']
                        original_movie_imdb = original_movie_data['imdb_id_full']
                        
                        # Display original movie details
                        if show_plots:
                            original_panel = Panel(
                                f"[bold]{original_movie_title}[/bold]\n\n[italic]{original_movie_plot}[/italic]\n\n"
                                f"[dim]IMDb: https://www.imdb.com/title/{original_movie_imdb}/[/dim]",
                                title="[bold blue]Searched Movie[/bold blue]",
                                border_style="blue",
                                width=100
                            )
                            console.print(original_panel)
                        else:
                            console.print(f"\n[bold blue]Searched Movie:[/bold blue] [bold]{original_movie_title}[/bold]")
                            console.print(f"[dim]IMDb: https://www.imdb.com/title/{original_movie_imdb}/[/dim]")
                    else:
                        # For association recommender or any other type
                        console.print(f"\n[bold blue]Recommendations for:[/bold blue] [bold]{actual_title}[/bold]")
                    
                    # Display recommendations table
                    # Use different title based on recommender type
                    table_title = f"Top 10 {'Movies Similar to' if is_content_based else 'Movies Liked by People Who Also Liked'} '[italic]{actual_title}[/italic]'"
                    table = Table(title=table_title, show_header=True, header_style="bold magenta")
                    table.add_column("Rank", style="dim", width=6)
                    table.add_column("Recommended Movie Title")
                    table.add_column("IMDb Link")
                    
                    # Add score column for Hybrid Plot Recommender
                    if is_content_based and len(result[0]) > 2:
                        # Check if we have the full formula components
                        has_formula_components = len(result[0]) >= 6
                        if has_formula_components:
                            table.add_column("Score Formula", width=40)
                        else:
                            table.add_column("Score", width=10)
                            # Add original rating column if available
                            if len(result[0]) > 3:
                                table.add_column("Original Rating", width=14)

                    # Check recommendation format
                    has_score = is_content_based and len(result[0]) > 2
                    has_original_rating = is_content_based and len(result[0]) > 3
                    has_formula_components = is_content_based and len(result[0]) >= 6
                    
                    for i, rec in enumerate(result):
                        # Unpack recommendation data based on format
                        if has_formula_components:
                            title, imdb_id, combined_score, similarity, norm_weighted_score, original_rating = rec
                            similarity_weight = recommender.similarity_weight if hasattr(recommender, 'similarity_weight') else 0.7
                            weighted_score_weight = 1 - similarity_weight
                            
                            # Create formula display: 0.307 = 0.7 * 0.4 + 0.3 * 0.1
                            formula = (
                                f"[bold cyan]{combined_score:.3f}[/bold cyan] = "
                                f"[green]{similarity_weight:.1f}[/green] * [blue]{similarity:.3f}[/blue] + "
                                f"[green]{weighted_score_weight:.1f}[/green] * [yellow]{norm_weighted_score:.3f}[/yellow]"
                            )
                            
                        elif has_original_rating:
                            title, imdb_id, score, original_rating = rec
                            score_display = f"[bold cyan]{score:.3f}[/bold cyan]"
                            rating_display = f"[yellow]{original_rating:.1f}[/yellow]"
                        elif has_score:
                            title, imdb_id, score = rec
                            score_display = f"[bold cyan]{score:.3f}[/bold cyan]"
                        else:
                            title, imdb_id = rec
                            
                        link = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else "[dim]N/A[/dim]"
                        
                        if has_formula_components:
                            table.add_row(str(i + 1), title, link, formula)
                        elif has_original_rating:
                            table.add_row(str(i + 1), title, link, score_display, rating_display)
                        elif has_score:
                            table.add_row(str(i + 1), title, link, score_display)
                        else:
                            table.add_row(str(i + 1), title, link)

                    console.print(table)
                    
                    # If it's a content-based recommender with a get_details method, display configuration
                    if is_content_based and hasattr(recommender, 'get_details'):
                        try:
                            details = recommender.get_details()
                            if 'similarity_weight' in details:
                                weights_table = Table(title="Hybrid Recommender Configuration", show_header=True, header_style="bold cyan")
                                weights_table.add_column("Parameter", width=30)
                                weights_table.add_column("Value", width=15)
                                weights_table.add_row("Plot Similarity Weight", f"{details['similarity_weight']:.2f}")
                                weights_table.add_row("Weighted Rating Weight", f"{details['weighted_rating_weight']:.2f}")
                                weights_table.add_row("Mean Rating (C)", f"{details['mean_vote_threshold']:.2f}")
                                weights_table.add_row("Min. Vote Count (m)", f"{int(details['minimum_vote_threshold'])}")
                                console.print(weights_table)
                        except Exception as e:
                            console.print(f"[yellow]Note: Could not display recommender details: {e}[/yellow]")
                    
                    # Display detailed recommendations with plots if enabled
                    if show_plots and is_content_based:
                        console.print("\n[bold magenta]Recommended Movies Details:[/bold magenta]")
                        for i, rec in enumerate(result):
                            # Unpack recommendation data based on format
                            if has_formula_components:
                                title, imdb_id, combined_score, similarity, norm_weighted_score, original_rating = rec
                                similarity_weight = recommender.similarity_weight if hasattr(recommender, 'similarity_weight') else 0.7
                                weighted_score_weight = 1 - similarity_weight
                                
                                # Create formula display: 0.307 = 0.7 * 0.4 + 0.3 * 0.1
                                formula = (
                                    f"[bold cyan]{combined_score:.3f}[/bold cyan] = "
                                    f"[green]{similarity_weight:.1f}[/green] * [blue]{similarity:.3f}[/blue] + "
                                    f"[green]{weighted_score_weight:.1f}[/green] * [yellow]{norm_weighted_score:.3f}[/yellow]"
                                )
                                
                            elif has_original_rating:
                                title, imdb_id, score, original_rating = rec
                                score_display = f"[bold cyan]{score:.3f}[/bold cyan]"
                                rating_display = f"[yellow]{original_rating:.1f}[/yellow]"
                            elif has_score:
                                title, imdb_id, score = rec
                                score_display = f"[bold cyan]{score:.3f}[/bold cyan]"
                                original_rating = None
                            else:
                                title, imdb_id = rec
                                score = None
                                original_rating = None
                                
                            try:
                                # Find the movie data in the metadata
                                movie_idx = recommender.indices[title]
                                movie_data = recommender.metadata.iloc[movie_idx]
                                movie_plot = movie_data['overview']
                                
                                # Create panel content with score if available
                                panel_content = f"[bold]{title}[/bold]"
                                if has_formula_components:
                                    formula_display = (
                                        f"\n[bold cyan]Score: {combined_score:.3f}[/bold cyan]\n"
                                        f"[dim]Formula: {combined_score:.3f} = "
                                        f"{similarity_weight:.1f} * {similarity:.3f} (similarity) + "
                                        f"{weighted_score_weight:.1f} * {norm_weighted_score:.3f} (weighted score)[/dim]"
                                    )
                                    panel_content += formula_display
                                elif has_original_rating:
                                    panel_content += f" [bold cyan](Score: {score:.3f})[/bold cyan] [yellow](Original Rating: {original_rating:.1f})[/yellow]"
                                elif score is not None:
                                    panel_content += f" [bold cyan](Score: {score:.3f})[/bold cyan]"
                                    
                                panel_content += f"\n\n[italic]{movie_plot}[/italic]\n\n" \
                                                f"[dim]IMDb: https://www.imdb.com/title/{imdb_id}/[/dim]"
                                
                                # Create a panel for each recommended movie
                                rec_panel = Panel(
                                    panel_content,
                                    title=f"[bold green]#{i+1} Recommendation[/bold green]",
                                    border_style="green",
                                    width=100
                                )
                                console.print(rec_panel)
                            except (KeyError, IndexError) as e:
                                console.print(f"[yellow]Could not retrieve full details for '{title}': {e}[/yellow]")
                                continue
                            
                            # Ask if user wants to see more details after each recommendation
                            if i < len(result) - 1:
                                see_more = Prompt.ask("\nSee next recommendation?", choices=["y", "n"], default="y")
                                if see_more.lower() == "n":
                                    break
                except (KeyError, IndexError) as e:
                    console.print(f"[bold red]Error retrieving movie details: {e}[/bold red]")
                    # Still show the basic recommendation table
                    table = Table(title=f"Top 10 Recommendations", show_header=True, header_style="bold magenta")
                    table.add_column("Rank", style="dim", width=6)
                    table.add_column("Recommended Movie Title")
                    table.add_column("IMDb Link")
                    
                    # Add score column for Hybrid Plot Recommender
                    if is_content_based and len(result[0]) > 2:
                        table.add_column("Score", width=10)
                        # Add original rating column if available
                        if len(result[0]) > 3:
                            table.add_column("Original Rating", width=14)

                    # Check if the recommendation tuple has 3 elements (title, imdb_id, score)
                    has_score = is_content_based and len(result[0]) > 2
                    has_original_rating = is_content_based and len(result[0]) > 3
                    
                    for i, rec in enumerate(result):
                        # Unpack recommendation data based on format
                        if has_original_rating:
                            title, imdb_id, score, original_rating = rec
                            score_display = f"[bold cyan]{score:.3f}[/bold cyan]"
                            rating_display = f"[yellow]{original_rating:.1f}[/yellow]"
                        elif has_score:
                            title, imdb_id, score = rec
                            score_display = f"[bold cyan]{score:.3f}[/bold cyan]"
                        else:
                            title, imdb_id = rec
                            
                        link = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else "[dim]N/A[/dim]"
                        
                        if has_original_rating:
                            table.add_row(str(i + 1), title, link, score_display, rating_display)
                        elif has_score:
                            table.add_row(str(i + 1), title, link, score_display)
                        else:
                            table.add_row(str(i + 1), title, link)

                    console.print(table)

            # Case 3: No match found (or search cancelled)
            elif result is None:
                 # Error message was already printed by the recommender or selection function
                console.print("[bold red]No recommendations found. Try a different movie.[/bold red]")
                
        except Exception as e:
            console.print(f"[bold red]An error occurred: {e}[/bold red]")
            import traceback
            traceback.print_exc() # Print detailed error for debugging

if __name__ == '__main__':
    # Example Usage (requires PlotRecommender, data_loader, utils, and dataset)
    # Run this from the project root directory
    print(f"Current working directory: {os.getcwd()}")

    try:
        # Attempt imports assuming execution from project root
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add project root
        from src.data_loader import load_metadata
        from src.recommenders.plot_recommender import PlotRecommender
        # Try to import SimpleRecommender if available
        try:
            from src.recommenders.simple_recommender import SimpleRecommender
            has_simple_recommender = True
        except ImportError:
            has_simple_recommender = False
    except ImportError as e:
        print("\nError: Could not import necessary modules.")
        print(f"Details: {e}")
        print("Please run this UI through 'python src/main.py' from the project root directory.")
        exit()


    print("\n--- CLI Example ---")
    metadata = load_metadata()

    if not metadata.empty:
        print("Initializing and fitting recommender for UI example...")
        # Let user choose recommender type if both are available
        recommender_type = "plot"
        if has_simple_recommender:
            recommender_type = Prompt.ask(
                "Choose recommender type", 
                choices=["plot", "simple"], 
                default="plot"
            )
            
        if recommender_type == "simple" and has_simple_recommender:
            recommender = SimpleRecommender()
        else:
            recommender = PlotRecommender()
            
        recommender.fit(metadata)
        print("Starting UI...")
        run_ui(recommender)
    else:
        print("\nCould not load metadata. Cannot run UI example.")