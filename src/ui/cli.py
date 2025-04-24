from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
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

    while True:
        movie_title_input = Prompt.ask("\n[bold green]Enter a movie title to get recommendations (or type 'quit' to exit)[/bold green]")

        if movie_title_input.lower() == 'quit':
            console.print("[bold yellow]Goodbye![/bold yellow]")
            break

        if not movie_title_input:
            console.print("[bold red]Please enter a movie title.[/bold red]")
            continue

        console.print(f"\nSearching for '[italic]{movie_title_input}[/italic]'...")
        try:
            # Initial recommendation call
            result = recommender.recommend(movie_title_input, top_n=10)

            # --- Handle Different Result Types ---

            # Case 1: Fuzzy matches returned, need user selection
            if isinstance(result, list) and result and isinstance(result[0], str):
                selected_title = select_movie_from_matches(result)
                if selected_title:
                    console.print(f"\nFetching recommendations for selected title '[italic]{selected_title}[/italic]'...")
                    # Call recommend again with the selected title (should be an exact match now)
                    result = recommender.recommend(selected_title, top_n=10)
                else:
                    console.print("[yellow]Search cancelled.[/yellow]")
                    result = None # Treat cancellation as no result found for display logic

            # Case 2: Exact match recommendations returned (or after selection)
            if isinstance(result, list) and result and isinstance(result[0], tuple):
                table = Table(title=f"Top 10 Recommendations for '[italic]{movie_title_input}[/italic]'", show_header=True, header_style="bold magenta")
                table.add_column("Rank", style="dim", width=6)
                table.add_column("Recommended Movie Title")
                table.add_column("IMDb Link") # New column for links

                for i, (title, imdb_id) in enumerate(result):
                    link = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else "[dim]N/A[/dim]"
                    table.add_row(str(i + 1), title, link)

                console.print(table)

            # Case 3: No match found (or search cancelled)
            elif result is None:
                 # Error message was already printed by the recommender or selection function
                 console.print("[yellow]No recommendations generated.[/yellow]")

            # Case 4: Unexpected result type (should not happen with current recommender)
            elif result is not None:
                 console.print(f"[bold red]Unexpected result type from recommender: {type(result)}[/bold red]")


        except Exception as e:
            console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
            # import traceback
            # traceback.print_exc() # Uncomment for detailed error traceback

        console.print("-" * 60) # Separator

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
    except ImportError as e:
        print("\nError: Could not import necessary modules.")
        print(f"Details: {e}")
        print("Please run this UI through 'python src/main.py' from the project root directory.")
        exit()


    print("\n--- CLI Example ---")
    metadata = load_metadata()

    if not metadata.empty:
        print("Initializing and fitting recommender for UI example...")
        recommender = PlotRecommender()
        recommender.fit(metadata)
        print("Starting UI...")
        run_ui(recommender)
    else:
        print("\nCould not load metadata. Cannot run UI example.")