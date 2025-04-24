from typing import List, Optional, Sequence, Tuple
from thefuzz import process, fuzz

def find_best_match(query: str, choices: Sequence[str], score_cutoff: int = 80) -> Optional[str]:
    """
    Finds the single best fuzzy match for a query string within a list of choices.

    Args:
        query: The string to search for.
        choices: A sequence of strings to search within.
        score_cutoff: The minimum similarity score (0-100) required.

    Returns:
        The best matching string if score is above cutoff, otherwise None.
    """
    if not choices or not query:
        return None
    result = process.extractOne(query, choices, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
    if result:
        best_match, score = result
        print(f"Fuzzy search (best): Found '{best_match}' for query '{query}' with score {score}")
        return best_match
    else:
        # print(f"Fuzzy search (best): No match found for query '{query}' with score cutoff {score_cutoff}")
        return None

def find_top_fuzzy_matches(query: str, choices: Sequence[str], limit: int = 5, score_cutoff: int = 75) -> List[Tuple[str, int]]:
    """
    Finds the top N fuzzy matches for a query string within a list of choices.

    Args:
        query: The string to search for.
        choices: A sequence of strings to search within.
        limit: The maximum number of matches to return.
        score_cutoff: The minimum similarity score (0-100) required for a match to be included.

    Returns:
        A list of tuples, where each tuple is (match_string, score), sorted by score descending.
        Returns an empty list if no matches meet the cutoff.
    """
    if not choices or not query:
        return []

    # extract returns a list of (match, score) tuples
    results = process.extract(query, choices, scorer=fuzz.WRatio, limit=limit)

    # Filter results by score_cutoff
    filtered_results = [(match, score) for match, score in results if score >= score_cutoff]

    if filtered_results:
        print(f"Fuzzy search (top): Found {len(filtered_results)} matches for query '{query}' with score cutoff {score_cutoff}")
    # else:
        # print(f"Fuzzy search (top): No matches found for query '{query}' with score cutoff {score_cutoff}")

    return filtered_results


if __name__ == '__main__':
    # Example Usage
    movie_list = [
        "The Dark Knight",
        "The Dark Knight Rises",
        "Batman Begins",
        "Inception",
        "Interstellar",
        "Pulp Fiction",
        "Dark Shadows" # Add another similar title
    ]

    print("--- Fuzzy Search Examples ---")

    query1 = "Dark Knight"
    matches1 = find_top_fuzzy_matches(query1, movie_list, limit=3)
    print(f"Query: '{query1}' -> Top Matches: {matches1}\n")
    # Expected: [('The Dark Knight', score), ('The Dark Knight Rises', score), ('Dark Shadows', score)]

    query2 = "Batmn Begins"
    matches2 = find_top_fuzzy_matches(query2, movie_list, limit=3, score_cutoff=85)
    print(f"Query: '{query2}' (cutoff 85) -> Top Matches: {matches2}\n")
    # Expected: [('Batman Begins', score)]

    query3 = "Totally Wrong Movie"
    matches3 = find_top_fuzzy_matches(query3, movie_list)
    print(f"Query: '{query3}' -> Top Matches: {matches3}\n")
    # Expected: []

    # Test find_best_match remains functional
    best_match_test = find_best_match("dark night", movie_list)
    print(f"Query (best): 'dark night' -> Match: '{best_match_test}'\n") # Expected: The Dark Knight