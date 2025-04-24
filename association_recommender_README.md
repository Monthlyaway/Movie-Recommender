# Association Rule Mining Movie Recommender

## 1. Introduction

This document describes the implementation and theory behind the Association Rule Mining Movie Recommender, a collaborative filtering approach that discovers hidden patterns in user behavior to generate recommendations. Unlike content-based recommenders that rely on movie features (like plot similarity) or simple recommenders that rank by overall popularity, association rule mining identifies relationships between movies based on user behavior, such as "users who liked Movie A also liked Movie B."

### 1.1 Problem Statement

Traditional recommendation systems often suffer from the following limitations:
- Content-based systems require rich metadata and may miss unexpected connections
- Rating-based systems can be skewed by popularity and fail to capture niche preferences
- Both approaches may not capture the subtle patterns of human viewing preferences

Association rule mining addresses these limitations by uncovering statistical relationships in user behavior without requiring content knowledge or assuming overall popularity as a proxy for relevance.

## 2. Theoretical Background

### 2.1 Association Rules

Association rule mining is a rule-based machine learning method for discovering relationships between variables in large datasets. Originally developed for market basket analysis (e.g., which products are frequently purchased together), it can be effectively applied to movie recommendation by treating each user's liked movies as a "transaction."

An association rule takes the form:
```
{Antecedent} → {Consequent}
```

For example: `{The Dark Knight} → {Inception}` indicates that users who liked "The Dark Knight" also tended to like "Inception."

### 2.2 Key Metrics

Association rules are evaluated using three primary metrics:

1. **Support**: The frequency with which a rule appears in the dataset.
   - Mathematically: `support(X→Y) = frequency(X∪Y) / total_transactions`
   - Represents how popular or common a rule is
   - Example: A support of 0.05 means 5% of users liked both movies

2. **Confidence**: The reliability of the inference made by a rule.
   - Mathematically: `confidence(X→Y) = support(X∪Y) / support(X)`
   - Conditional probability of finding Y given X
   - Example: A confidence of 0.8 means 80% of users who liked Movie X also liked Movie Y

3. **Lift**: How much more likely Y is, given X, compared to if there was no relationship.
   - Mathematically: `lift(X→Y) = confidence(X→Y) / support(Y)`
   - Measures how much the rule improves predictions over random chance
   - Example: A lift of 3.0 means users who liked Movie X are 3 times more likely to like Movie Y than the average user

### 2.3 FP-Growth Algorithm

The FP-Growth (Frequent Pattern Growth) algorithm is an efficient method for mining frequent itemsets without candidate generation. Compared to alternatives like Apriori, FP-Growth is significantly faster, especially for large datasets.

#### How FP-Growth Works:

1. **Scan 1**: Count the frequency of each item and filter those below minimum support
2. **Build FP-Tree**: Construct a compact tree structure representing frequent itemsets
   - Each path represents a set of movies that co-occur
   - Shared paths indicate common prefixes in the transactions
   - Header table links all nodes containing the same movie
3. **Mine Frequent Patterns**: Recursively extract conditional pattern bases and build conditional FP-trees
4. **Generate Association Rules**: From frequent itemsets, derive rules that meet minimum confidence/lift thresholds

This approach is particularly well-suited for movie recommendation because it efficiently handles sparse data (most users rate only a small fraction of all available movies).

## 3. Implementation

### 3.1 Data Preprocessing

The implementation follows these preprocessing steps:

1. **Rating Threshold**: Identify "liked" movies by filtering ratings above a threshold (default: 3.5)
   - This converts the numerical rating problem into a binary "liked/not liked" framework
   - Only movies that users genuinely enjoyed contribute to pattern discovery

2. **Transaction Creation**: Group ratings by user to create "transactions"
   - Each transaction is a list of movie IDs that a particular user liked
   - These represent the co-occurrence patterns we seek to mine

```python
# Filter ratings to only include "liked" movies
liked_ratings = ratings_df[ratings_df['rating'] >= self.rating_threshold]

# Group ratings by user to create transactions
transactions = liked_ratings.groupby('userId')['movieId'].apply(list).tolist()
```

### 3.2 Transaction Encoding

For the FP-Growth algorithm to process the transactions, they must be encoded into a binary format:

```python
# Encode transactions for FP-Growth
from mlxtend.preprocessing import TransactionEncoder
encoder = TransactionEncoder()
encoded_data = encoder.fit_transform(transactions)
df_encoded = pd.DataFrame(encoded_data, columns=encoder.columns_)
```

Each row in the resulting dataframe represents a user, and each column represents a movie. A value of `True` indicates the user liked that movie.

### 3.3 Mining Frequent Patterns

The mlxtend library's implementation of FP-Growth is used to discover frequent patterns:

```python
# Run FP-Growth to find frequent itemsets
from mlxtend.frequent_patterns import fpgrowth
frequent_itemsets = fpgrowth(
    df_encoded, 
    min_support=self.min_support, 
    use_colnames=True
)
```

The `min_support` parameter determines the minimum frequency threshold for itemsets to be considered "frequent." Setting this value requires careful consideration:
- Too high: Only obvious, highly popular patterns will be discovered
- Too low: Computational complexity increases exponentially and may include coincidental patterns

### 3.4 Generating Association Rules

From the frequent itemsets, association rules are derived:

```python
# Generate association rules
from mlxtend.frequent_patterns import association_rules
rules = association_rules(
    frequent_itemsets, 
    metric="confidence", 
    min_threshold=self.min_confidence
)

# Filter rules by lift
rules = rules[rules['lift'] >= self.min_lift]
```

Rules are filtered using both:
- `min_confidence`: Ensures rules have sufficient predictive power
- `min_lift`: Ensures rules represent genuine associations rather than coincidental co-occurrences

### 3.5 Recommendation Generation

To generate recommendations for a specific movie:

1. Find rules where the movie appears in the antecedent
2. Sort rules by confidence and lift
3. Extract consequent movies as recommendations
4. Eliminate duplicates and limit to top-N

```python
# Find all rules where the movie is in the antecedent
matching_rules = []
for _, rule in self.rules.iterrows():
    antecedent = rule['antecedents']
    if movie_id in antecedent and len(antecedent) == 1:
        matching_rules.append((rule['consequents'], rule['confidence'], rule['lift']))

# Sort by confidence and lift
matching_rules.sort(key=lambda x: (x[1], x[2]), reverse=True)

# Extract unique top-N recommendations
recommendations = []
seen_movies = set()
for consequent, _, _ in matching_rules:
    for movie_id in consequent:
        if movie_id not in seen_movies:
            # Add to recommendations
            seen_movies.add(movie_id)
            if len(recommendations) >= top_n:
                break
```

## 4. System Architecture

The association recommender is integrated into the existing movie recommendation system with these components:

1. **AssociationRecommender Class**: Core implementation in `src/recommenders/association_recommender.py`
2. **Data Loading**: Extended `data_loader.py` with `load_ratings()` function
3. **Command-Line Interface**: Updated `main.py` with association recommender options
4. **User Interface**: Enhanced CLI in `ui/cli.py` to handle association-based recommendations
5. **Demo Script**: Added `association_demo.py` for standalone testing

### 4.1 Class Structure

The `AssociationRecommender` class follows this structure:

- **Constructor**: Initialize parameters (min_support, min_confidence, min_lift, rating_threshold)
- **fit()**: Process data and generate rules
- **recommend()**: Find recommendations for a given movie
- **_get_imdb_id()**: Helper method to retrieve IMDb IDs for recommendations
- **get_details()**: Return system configuration details

## 5. Experimental Results

### 5.1 Parameter Sensitivity

The system's effectiveness depends on careful parameter tuning:

| Parameter | Too Low | Too High | Recommended |
|-----------|---------|----------|-------------|
| min_support | Many rules, slow performance, potential noise | Few rules, missing valuable patterns | 0.005-0.02 |
| min_confidence | Unreliable recommendations | Too restrictive, few recommendations | 0.3-0.5 |
| min_lift | Includes coincidental patterns | Might miss some valid rules | 1.2-3.0 |
| rating_threshold | Includes movies users felt neutral about | Limited data, sparse rules | 3.5-3.5 |

### 5.2 Example Rules

Sample association rules generated (actual values will vary based on dataset):

```
Rule: ['The Dark Knight'] → ['Batman Begins']
Support: 0.0312, Confidence: 0.6233, Lift: 8.2455

Rule: ['Toy Story'] → ['Finding Nemo']
Support: 0.0254, Confidence: 0.4123, Lift: 4.7841

Rule: ['The Matrix'] → ['The Matrix Reloaded']
Support: 0.0289, Confidence: 0.5521, Lift: 12.3241
```

### 5.3 Comparison with Other Approaches

Comparing recommendations for "The Dark Knight":

**Plot Recommender (Content-based)**:
- Batman Begins
- The Dark Knight Rises
- Batman: The Killing Joke
- Watchmen
- V for Vendetta

**Association Recommender (Collaborative)**:
- Batman Begins
- The Dark Knight Rises
- Inception
- The Prestige
- Interstellar

Note how the association recommender captures director Christopher Nolan's films that aren't superhero/comic related, demonstrating its ability to find patterns beyond obvious content similarity.

## 6. Usage Guide

### 6.1 Command-Line Arguments

```bash
python src/main.py --recommender association [options]
```

Options:
- `--min-support`: Minimum support threshold (default: 0.06)
- `--min-confidence`: Minimum confidence threshold (default: 0.3)
- `--min-lift`: Minimum lift threshold (default: 1.2)
- `--rating-threshold`: Minimum rating to consider a movie "liked" (default: 4.0)

### 6.2 Interactive Rule Exploration

The system provides an interactive way to explore the generated association rules:

1. Start the recommender with `python src/main.py --recommender association`
2. From the main menu, select option 1: "Display random association rules"
3. Enter the number of random rules you wish to view
4. For each rule, the system will display:
   - The antecedent movies (IF user likes these)
   - The consequent movies (THEN they'll likely also like these)
   - Key metrics (support, confidence, lift)
   - Option to continue viewing more rules or return to main menu

This feature provides insight into the patterns discovered by the algorithm and helps understand the reasoning behind recommendations.

### 6.3 Programmatic Usage

```python
from src.data_loader import load_metadata, load_ratings
from src.recommenders.association_recommender import AssociationRecommender

# Load data
metadata_df = load_metadata()
ratings_df = load_ratings(use_small=True)

# Create and fit recommender
recommender = AssociationRecommender(
    min_support=0.06,
    min_confidence=0.3,
    min_lift=1.2,
    rating_threshold=4.0
)
recommender.fit(metadata_df, ratings_df)

# Get recommendations
recommendations = recommender.recommend("Inception", top_n=10)

# Explore random rules
random_rules = recommender.get_random_rules(num_rules=5)
for rule in random_rules:
    print(f"Rule: {rule['antecedents']} → {rule['consequents']}")
    print(f"Support: {rule['support']:.4f}, Confidence: {rule['confidence']:.4f}, Lift: {rule['lift']:.4f}")
```

### 6.4 Demo Script

For a quick demonstration of the system's capabilities:

```bash
python src/association_demo.py
```

This will:
1. Load the datasets
2. Create and fit the recommender
3. Display sample association rules
4. Show recommendations for several popular movies

## 7. Discussion

### 7.1 Strengths

1. **Serendipitous Discovery**: Finds unexpected connections between movies
2. **No Content Knowledge Required**: Works without plot, genre, or other metadata
3. **Captures Collective Wisdom**: Leverages the "wisdom of crowds" to find patterns
4. **Scalable**: FP-Growth algorithm efficiently handles large datasets

### 7.2 Limitations

1. **Cold Start Problem**: Cannot recommend new movies without sufficient user interactions
2. **Sparsity Issues**: Requires sufficient overlapping user preferences to generate rules
3. **Popularity Bias**: Very popular movies may appear in many rules
4. **Computationally Intensive**: Rule generation can be resource-intensive for large datasets

### 7.3 Future Improvements

1. **Weighted Ratings**: Consider rating magnitude rather than binary like/dislike
2. **Temporal Dynamics**: Incorporate recency of ratings to capture evolving preferences
3. **Hybrid Approach**: Combine association rules with content-based filtering
4. **Rule Pruning**: Implement redundancy elimination to focus on the most useful rules
5. **Multi-Item Queries**: Support recommendations based on multiple liked movies

## 8. Conclusion

The Association Rule Mining Recommender demonstrates the power of collaborative filtering to discover patterns in user behavior and generate meaningful movie recommendations. By leveraging the FP-Growth algorithm, the system efficiently mines frequent itemsets and derives association rules that capture the subtle relationships between movies in users' preferences.

Unlike content-based approaches, this recommender can discover connections that aren't obvious from movie metadata alone, providing a complementary perspective that enhances the overall recommendation system. The implementation shows how traditional data mining techniques can be effectively applied to recommendation problems, providing users with suggestions based on the collective wisdom of other users with similar tastes.

## 9. References

1. Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. ACM SIGMOD Record, 29(2), 1-12.
2. Agrawal, R., Imieliński, T., & Swami, A. (1993). Mining association rules between sets of items in large databases. ACM SIGMOD Record, 22(2), 207-216.
3. Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to recommender systems handbook. In Recommender systems handbook (pp. 1-35). Springer.
4. Raschka, S. (2018). MLxtend: Providing machine learning and data science utilities and extensions to Python's scientific computing stack. Journal of Open Source Software, 3(24), 638. 