# Weighted Rating System for Movie Recommendations

## 1. Introduction

The Weighted Rating approach is a popularity-based recommendation method that addresses a fundamental challenge in rating systems: how to balance the average rating of an item with the number of ratings it has received. This approach is widely used in various domains, including the movie industry, where the Internet Movie Database (IMDB) employs a variant of this formula to rank films.

The core insight behind weighted ratings is that items with very few ratings, even if those ratings are high, might not be as reliably good as items with many ratings and a slightly lower average. Conversely, highly popular items should not dominate recommendations solely based on the quantity of ratings they receive.

## 2. Mathematical Formulation

### 2.1 The Bayesian Average

The weighted rating formula can be interpreted as a Bayesian average, which incorporates prior belief about the average rating across all items. 

Let:
- $R$ be the mean rating for a given movie
- $v$ be the number of ratings for the movie
- $C$ be the mean rating across all movies in the dataset
- $m$ be the minimum number of ratings required to be considered (often set at a specific percentile of the vote count distribution)

The weighted rating $W$ is then calculated as:

$$W = \frac{v}{v+m} \cdot R + \frac{m}{v+m} \cdot C$$

### 2.2 Formula Interpretation

This formula can be understood as:
1. When $v$ is small (few ratings), $W$ is closer to $C$ (the global mean)
2. When $v$ is large (many ratings), $W$ is closer to $R$ (the movie's own mean)

The parameter $m$ acts as a regularization term that determines how quickly we transition from trusting the global average to trusting the item's own average as the number of ratings increases.

### 2.3 Statistical Foundations

From a statistical perspective, this formula represents a shrinkage estimator that pulls individual estimates toward a common value. It can be derived from a Bayesian framework where:
- $C$ is the prior mean
- $m$ is related to the prior strength or precision
- $R$ and $v$ are the observed data

The posterior mean in this Bayesian framework is precisely the weighted rating $W$.

## 3. Parameter Selection

### 3.1 Minimum Votes Threshold ($m$)

The choice of $m$ significantly impacts the recommendations:

- **Higher $m$**: More conservative, favoring movies with many ratings
- **Lower $m$**: More liberal, giving newer or niche movies a better chance

A common approach is to set $m$ as a percentile of the vote count distribution, typically the 90th or 95th percentile. This ensures that movies must have a substantial number of ratings to be prominently featured in recommendations based solely on their own ratings.

### 3.2 Global Mean ($C$)

The global mean $C$ represents the central tendency of all ratings in the system. It serves as the baseline toward which movies with few ratings are pulled.

In practical applications, $C$ is calculated as:

$$C = \frac{\sum_{i=1}^{N} r_i}{N}$$

where $r_i$ is the average rating of movie $i$, and $N$ is the total number of movies in the dataset.

## 4. Application to Movie Recommendations

### 4.1 Filtering Process

The implementation typically follows these steps:

1. Calculate the mean rating across all movies ($C$)
2. Determine the minimum votes threshold ($m$), often using a percentile
3. Filter to include only movies with at least $m$ votes
4. Calculate the weighted rating for each qualified movie
5. Sort movies by their weighted rating

### 4.2 Mathematical Example

Consider a movie with an average rating $R = 8.2$ and vote count $v = 200$, in a system where $C = 7.0$ and $m = 500$:

$$W = \frac{200}{200+500} \cdot 8.2 + \frac{500}{200+500} \cdot 7.0 = 0.286 \cdot 8.2 + 0.714 \cdot 7.0 = 7.34$$

Despite its high average rating of 8.2, the movie's weighted rating is pulled down to 7.34 because it has relatively few votes compared to the threshold.

## 5. Theoretical Properties

### 5.1 Advantages

1. **Bias Reduction**: Mitigates the bias toward movies with very few ratings
2. **Reliability**: Places more weight on statistically significant sample sizes
3. **Simplicity**: Computationally efficient and easy to implement
4. **Interpretability**: The formula and resulting rankings are transparent and explainable

### 5.2 Limitations

1. **Non-personalized**: Provides the same recommendations to all users
2. **Popularity Bias**: Still inherently favors popular movies over niche content
3. **Cold Start Problem**: New movies with few ratings are penalized regardless of quality
4. **No Content Consideration**: Ignores movie attributes and user preferences

## 6. Relationship to Other Methods

### 6.1 Comparison with Raw Averages

Unlike simple arithmetic means, weighted ratings prevent obscure items with few but perfect ratings from dominating recommendations.

### 6.2 Comparison with Advanced Collaborative Filtering

While collaborative filtering methods (matrix factorization, neighborhood methods) provide personalized recommendations by finding patterns in user-item interactions, the weighted rating approach provides a simpler, non-personalized alternative that serves as a solid baseline and is especially useful for new users with no rating history.

### 6.3 Complementary Role

In practice, weighted ratings often play a complementary role in recommendation systems:
- As a "popular now" or "critically acclaimed" section
- As a fallback when personalized recommendations cannot be generated
- As an initial ranking that is further refined by personalization algorithms

## 7. Conclusion

The weighted rating approach offers a mathematically sound method for ranking items based on both their average rating and the confidence in that average derived from the number of ratings. While not personalized, it provides a robust baseline that addresses many of the statistical biases inherent in naive rating systems.

Its simplicity, interpretability, and effectiveness have made it a staple in recommendation systems across many domains, particularly in scenarios where providing generally popular and highly-rated content is the primary goal.

## References

1. Amatriain, X., & Basilico, J. (2015). Recommender systems in industry: A Netflix case study. In *Recommender systems handbook* (pp. 385-419). Springer.

2. Ekstrand, M. D., Riedl, J. T., & Konstan, J. A. (2011). Collaborative filtering recommender systems. *Foundations and Trends in Human-Computer Interaction, 4(2)*, 81-173.

3. Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian data analysis*. CRC press.

4. IMDb. (n.d.). IMDb Ratings FAQ. Retrieved from https://help.imdb.com/article/imdb/track-movies-tv/ratings-faq/G67Y87TFYYP6TWAV 