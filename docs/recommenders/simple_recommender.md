# 加权评分推荐器 (Simple Recommender) 

## 一、代码原理分析

本部分实现了一个简单的电影推荐系统，其核心思想是基于 IMDb 提出的**加权评分公式（Weighted Rating Formula）**，旨在避免“高分但冷门”的电影在推荐系统中被错误地高排名，从而提高推荐的可靠性和公平性。

### 1.1 加权评分公式（IMDB Weighted Rating）

推荐系统使用的评分计算公式如下：

$$
Score = \frac{v}{v + m} \cdot R + \frac{m}{v + m} \cdot C
$$


其中：

- `v`：该电影的投票总数（vote_count）  
- `m`：参与评分排名所需的最小投票数（minimum votes）  
- `R`：该电影的平均评分（vote_average）  
- `C`：所有电影的平均评分（dataset mean vote）

### 1.2 数学推导含义

这是一个加权平均数公式，用于平衡电影自身评分 `R` 与总体评分 `C` 的影响：

- 当 `v` 较大时，`v / (v + m)` 趋近于 1，说明该电影得到了大量投票，其评分更具可信度；
- 当 `v` 较小时，`m / (v + m)` 的权重更大，系统更偏向使用平均评分 `C` 替代。

这样做能有效防止只有几个评分的电影“冲榜”。

---

## 二、核心代码及解释

### 2.1 `fit()` 方法核心逻辑分析

```python
weighted_scores_result = calculate_all_weighted_scores(
    metadata_df, 
    vote_count_percentile=self.vote_count_percentile
)

self.C = weighted_scores_result['C']
self.m = weighted_scores_result['m']

qualified = metadata_df.copy().loc[metadata_df['vote_count'] >= self.m]
qualified['score'] = weighted_scores_result['scores'].loc[qualified.index]
```

首先使用calculate_all_weighted_scores调用工具函数，计算整体平均分'c'、最小票数阈值'm'，以及每部电影的加权分数。

后续的qualified是筛选出的满足最低票数要求的电影，作为“候选电影”。

最后，对这些候选电影，使用加权评分score作为排序依据。 

### 2.2 用户评分补充分析

```python
avg_ratings = self.ratings_df.groupby('movieId')['rating'].mean().reset_index()
avg_ratings.rename(columns={'rating': 'avg_user_rating'}, inplace=True)
...
qualified = pd.merge(qualified, avg_ratings, left_on='movieId', right_on='movieId', how='left')
qualified['avg_user_rating'] = qualified['avg_user_rating'].fillna(0)
```

对每部电影，根据数据集中的用户评分数据文件ratings.csv计算平均用户评分。

如果用户的评分最大值不为10，则将其按比例映射到10分制来统一标准。

在最终输出结果中将计算得到的信息合并进候选电影中作为补充展示数据。

### 2.3 recommend()方法核心逻辑

```python
top_movies = self.qualified_movies.head(top_n)

for _, movie in top_movies.iterrows():
    imdb_id = movie.get('imdb_id_full', None)
    weighted_score = float(movie['score'])
    avg_user_rating = float(movie.get('avg_user_rating', 0))
    results.append((movie['title'], imdb_id, weighted_score, avg_user_rating))
```

按照加权评分对所有符合条件的电影排序后，提取前top_n个电影。

返回的信息包括：电影名称、IMDb_ID、加权评分和用户评分。

## 三、实验设置于结果展示

本章节将展示加权评分推荐器的启动方式和输出结果示例。

### 3.1 启动命令

```bash
python src/main.py simple
```
### 3.2 输出结果示例

![](..\images\chen_1.png)
![](..\images\chen_2.png)


## 四、总结

本推荐系统以 IMDb 的加权评分公式为核心，有效平衡评分数量与评分质量，兼顾了用户评分数据，使得结果更具可解释性与实用性。该方法简单但有效，适合用于中小型电影推荐系统的原型验证。
