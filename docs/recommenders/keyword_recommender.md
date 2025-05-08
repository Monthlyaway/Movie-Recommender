# 关键词电影推荐系统实验报告

## 引言 (Introduction)

### 项目背景与目的
电影推荐系统对于提升用户在海量影片中的发现体验至关重要。本项目旨在构建一个基于关键词的电影推荐器，通过结合用户输入的关键词与电影的固有属性（如流行度、内容相关性）来进行推荐。本实验报告将详细阐述该推荐器的原理、核心代码实现、运行方式，并讨论其结果、优缺点及未来改进方向。

### 数据集说明
本推荐系统使用的数据主要包括：
*   **电影元数据**: 包含电影ID、标题、概述、平均评分 (`vote_average`)、投票数 (`vote_count`)等。
*   **电影关键词数据**: 记录每部电影关联的关键词列表。


## 关键词推荐器原理 (Principle of the Keyword Recommender)

关键词推荐系统的核心思想是结合用户明确表达的兴趣（通过输入的关键词）与电影本身固有的属性（如流行度、内容相关性）来进行综合推荐。

其主要实现步骤如下：

### 关键词处理与逆文档频率 (IDF) 计算
1.  **关键词提取**: 从关键词数据中解析出每部电影的关键词集合。此步骤由 `_parse_keyword_string` 方法完成。
2.  **IDF值计算**: 计算每个独立关键词的逆文档频率 (IDF) 以评估其重要性。公式如下：

    $$IDF(\text{keyword}) = \log\left(\frac{N + 1}{df_{\text{keyword}} + 1}\right) + 1$$

    其中：
    *   $N$ 是电影总数。
    *   $df_{\text{keyword}}$ 是包含该关键词的电影数量。
    此逻辑在 `_calculate_idf` 方法中实现。

### 电影加权评分计算 (IMDB-style Weighted Score)
为了评估电影的整体质量和受欢迎程度，系统采用IMDB式的加权评分，综合考虑平均评分 (`vote_average`) 和投票数 (`vote_count`)，并进行归一化。此功能由 [`../../src/utils/weighted_score.py`](../../src/utils/weighted_score.py) 函数提供，并在 `KeywordRecommender` 的 `fit` 方法中调用。

### 用户查询与关键词相关性评分 (KRS)
当用户输入查询关键词时：
1.  **用户关键词解析**: 将输入字符串解析为关键词集合。
2.  **匹配与KRS计算**: 计算电影的关键词相关性评分 (KRS)：

    $$KRS(\text{movie}) = \sum IDF(\text{matched\_keyword})$$

    即电影所有匹配关键词的IDF值之和。
3.  **KRS归一化**: 将KRS归一化到0-1范围。

### 最终推荐分数计算与排序
结合关键词相关性与电影质量计算最终分数：
*   **最终分数 (Final Score)**:

    $$FinalScore(\text{movie}) = \alpha \times KRS(\text{movie}) + \beta \times IMDB\_Score(\text{movie})$$

    其中 $\alpha$ 和 $\beta$ 是权重参数，控制关键词匹配精确度与电影流行度的相对重要性。
*   **生成推荐**: 电影根据最终分数降序排序，返回Top-N结果。
此流程在 `recommend` 方法中实现。

## 核心代码解析 (Core Code Analysis)

本章节将选取 [`../../src/recommenders/keyword_recommender.py`](../../src/recommenders/keyword_recommender.py) 文件中的关键 Python 代码片段进行展示和解析，以阐明关键词推荐器的内部实现细节。

### 初始化

推荐器的初始化方法负责设置关键参数和数据结构的初始状态。

```python
class KeywordRecommender:
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, vote_count_percentile: float = 0.85):
        """
        Initializes the KeywordRecommender.

        Args:
            alpha: Weight for the keyword relevance score (KRS).
            beta: Weight for the normalized IMDB weighted score.
            vote_count_percentile: Percentile for minimum votes in IMDB score calculation.
        """
        self.alpha = alpha
        self.beta = beta
        self.vote_count_percentile = vote_count_percentile

        self.metadata_df = pd.DataFrame()
        self.idf_scores = {}  # Stores IDF score for each keyword
        self.movie_keyword_sets = pd.Series(dtype='object') # Movie ID to keyword set
        self.normalized_imdb_scores = pd.Series(dtype='float64') # Movie ID to normalized IMDB score
        self._fitted = False # Flag to check if fit has been called
```
**解析**:
*   `alpha` 和 `beta`: 这两个浮点型参数分别定义了在计算最终推荐分数时，关键词相关性评分 (KRS) 和归一化IMDB加权评分的权重。它们的和通常建议为1。默认值分别为 `0.7` 和 `0.3`，表示在默认情况下，关键词匹配的相关性占比较大。
*   `vote_count_percentile`: 此参数用于计算IMDB加权评分时，确定作为有效投票数基线的百分位数。例如，`0.85` 表示只考虑投票数超过数据集中85%电影的那些电影的评分，以增加评分的可靠性。
*   `metadata_df`: 用于存储加载后的电影元数据。
*   `idf_scores`: 一个字典，用于存储每个关键词计算得到的IDF值。
*   `movie_keyword_sets`: 一个Pandas Series，其索引是电影ID，值是对应电影的关键词集合。
*   `normalized_imdb_scores`: 一个Pandas Series，存储每部电影归一化后的IMDB加权评分。
*   `_fitted`: 一个布尔标志，用于指示推荐器是否已经过 `fit` 方法的训练/数据准备。


### IDF计算 (`_calculate_idf`)
该方法计算数据集中所有独立关键词的逆文档频率 (IDF)。

```python
    def _calculate_idf(self, all_movie_keyword_sets: pd.Series):
        """
        Calculates Inverse Document Frequency (IDF) for all unique keywords.
        IDF = log((Total Number of Documents + 1) / (Document Frequency of Keyword + 1)) + 1 (smoothing)
        """
        if all_movie_keyword_sets.empty:
            self.idf_scores = {}
            return

        total_movies = len(all_movie_keyword_sets)
        keyword_doc_frequency = Counter() # Counts document frequency for each keyword

        for keyword_set in all_movie_keyword_sets:
            if isinstance(keyword_set, set):
                keyword_doc_frequency.update(list(keyword_set))

        self.idf_scores = {}
        for keyword, doc_freq in keyword_doc_frequency.items():
            self.idf_scores[keyword] = math.log((total_movies + 1) / (doc_freq + 1)) + 1
```
**解析**:
*   `all_movie_keyword_sets`: 输入是一个Pandas Series，包含了数据集中每部电影的关键词集合。
*   `total_movies`: 数据集中的电影总数。
*   `keyword_doc_frequency`: 使用 `collections.Counter` 来统计每个关键词在多少部电影中出现过（即文档频率 `df`）。
*   对于每个关键词，IDF值根据公式 `log((total_movies + 1) / (doc_freq + 1)) + 1` 计算。平滑处理（`+1`）用于避免当 `doc_freq` 为0或极大时可能出现的问题。
*   计算得到的IDF值存储在 `self.idf_scores` 字典中。

### 数据拟合/准备 (`fit`)
`fit` 方法是推荐器进行数据预处理和计算必要评分（如IDF、IMDB加权分）的入口。

```python
    def fit(self, metadata_df: pd.DataFrame, keywords_data_path: str = 'dataset/keywords.csv') -> bool:
        # ... (Error checking for metadata_df columns) ...
        self.metadata_df = metadata_df.copy()
        if 'id' in self.metadata_df.columns:
             self.metadata_df.set_index('id', inplace=True, drop=False)

        # 1. Load and Parse Keywords
        try:
            keywords_df = pd.read_csv(keywords_data_path)
            keywords_df['id'] = keywords_df['id'].astype(self.metadata_df.index.dtype)
            keywords_df.set_index('id', inplace=True)
        except Exception as e:
            # ... (Error handling) ...
            return False

        self.movie_keyword_sets = self.metadata_df.index.to_series().apply(
            lambda movie_id: self._parse_keyword_string(
                keywords_df.loc[movie_id, 'keywords']
            ) if movie_id in keywords_df.index else set()
        )
        self.metadata_df['keyword_set'] = self.movie_keyword_sets

        # 2. Calculate IDF Scores
        self._calculate_idf(self.movie_keyword_sets)
        # ... (Warning if IDF scores are empty) ...

        # 3. Calculate Normalized IMDB Scores
        try:
            imdb_scores_result = calculate_normalized_weighted_scores(
                self.metadata_df, # Requires 'vote_average', 'vote_count'
                vote_count_percentile=self.vote_count_percentile
            )
            self.normalized_imdb_scores = imdb_scores_result['normalized_scores']
            self.metadata_df['normalized_imdb_score'] = self.normalized_imdb_scores
        except Exception as e:
            # ... (Error handling) ...
            return False

        self._fitted = True
        return True
```
**解析**:
1.  **加载元数据**: 复制传入的 `metadata_df` 并以电影 `id` 设置索引。
2.  **加载和解析关键词**:
    *   从指定的 `keywords_data_path` (默认为 `dataset/keywords.csv`) 加载关键词数据。
    *   将关键词数据中的 `id` 列设置为索引，并确保其类型与元数据索引一致。
    *   对于元数据中的每一部电影，调用 `_parse_keyword_string` 方法来获取其关键词集合，并将结果存储在 `self.movie_keyword_sets` 和 `self.metadata_df['keyword_set']` 中。
3.  **计算IDF分数**: 调用 `_calculate_idf` 方法计算所有关键词的IDF值。
4.  **计算归一化IMDB分数**:
    *   调用 [`calculate_normalized_weighted_scores`](../../src/utils/weighted_score.py:1) 函数（来自 [`../../src/utils/weighted_score.py:1`](../../src/utils/weighted_score.py:1)）。此函数需要 `metadata_df` 中包含 `vote_average` 和 `vote_count` 列。
    *   将返回的归一化IMDB分数存储在 `self.normalized_imdb_scores` 和 `self.metadata_df['normalized_imdb_score']` 中。
5.  设置 `self._fitted = True` 标记表示数据准备完成。

### 生成推荐 (`recommend`)
此方法是推荐器的核心，根据用户输入的关键词生成电影推荐列表。

```python
    def recommend(self, user_keywords_str: str, top_n: int = 10) -> pd.DataFrame:
        if not self._fitted:
            # ... (Error if not fitted) ...
            return pd.DataFrame()

        user_keywords = {kw.strip().lower() for kw in user_keywords_str.split(',')}
        # ... (Handle empty user_keywords) ...

        # Calculate Keyword Relevance Score (KRS) for each movie
        krs_values = []
        for movie_id, movie_data in self.metadata_df.iterrows():
            movie_kws = movie_data.get('keyword_set', set())
            matched_keywords = user_keywords.intersection(movie_kws)
            if matched_keywords:
                krs = sum(self.idf_scores.get(kw, 0) for kw in matched_keywords)
                krs_values.append(krs)
            else:
                krs_values.append(0)
        self.metadata_df['krs'] = krs_values

        # Normalize KRS for movies that had at least one match (KRS > 0)
        positive_krs = self.metadata_df[self.metadata_df['krs'] > 0]['krs']
        if not positive_krs.empty:
            min_krs, max_krs = positive_krs.min(), positive_krs.max()
            krs_range = max_krs - min_krs
            if krs_range == 0:
                 self.metadata_df['normalized_krs'] = positive_krs.apply(lambda x: 1.0 if x > 0 else 0.0)
            else:
                self.metadata_df['normalized_krs'] = self.metadata_df['krs'].apply(
                    lambda x: (x - min_krs) / krs_range if x > 0 else 0.0
                )
        else:
            self.metadata_df['normalized_krs'] = 0.0
            # ... (Handle no keyword matches) ...

        # Calculate Final Score
        self.metadata_df['final_score'] = (self.alpha * self.metadata_df['normalized_krs'].fillna(0)) + \
                                          (self.beta * self.metadata_df['normalized_imdb_score'].fillna(0))

        # Sort and get top N recommendations
        recommendations = self.metadata_df.sort_values(by='final_score', ascending=False)
        # ... (Select and format output columns) ...
        return final_recs
```
**解析**:
1.  **检查拟合状态**: 确保 `fit` 方法已被调用。
2.  **用户关键词处理**: 将用户输入的逗号分隔的关键词字符串（`user_keywords_str`）转换为小写的关键词集合 `user_keywords`。
3.  **计算KRS**:
    *   遍历 `self.metadata_df` 中的每一部电影。
    *   获取该电影的关键词集合 (`movie_kws`)。
    *   找出用户关键词与电影关键词的交集 (`matched_keywords`)。
    *   如果存在匹配的关键词，则该电影的KRS计算为这些匹配关键词的IDF值之和（从 `self.idf_scores` 中获取IDF值，未知关键词的IDF视为0）。
    *   将计算得到的KRS存储在 `self.metadata_df['krs']` 列中。
4.  **归一化KRS**:
    *   对 `self.metadata_df['krs']` 列中大于0的值进行最小-最大规范化 (Min-Max Normalization)，将其缩放到 `[0, 1]` 区间。结果存储在 `self.metadata_df['normalized_krs']`。
    *   如果所有电影的KRS都为0（即没有匹配），则所有 `normalized_krs` 也为0。
5.  **计算最终分数**:
    *   根据公式 `final_score = alpha * normalized_krs + beta * normalized_imdb_score` 计算每部电影的最终推荐分数。
    *   使用 `.fillna(0)` 处理可能因某些电影没有IMDB评分或KRS为0而产生的NaN值。
6.  **排序与返回**:
    *   根据 `final_score` 对电影进行降序排序。
    *   选取指定的 `top_n` 部电影，并选择如标题、最终分数、各项原始分数、关键词集合等列作为最终推荐结果返回。

## 实验设置与结果展示 (Experiment Setup & Results)

本章节将描述运行关键词推荐系统的实验环境、启动参数，并通过具体的命令行示例来展示其交互方式和预期的输出结果。

### 启动推荐系统与参数说明
关键词推荐器通过项目根目录下的 [`../../src/main.py`](../../src/main.py) 脚本启动。运行该脚本时，需要指定 `keyword` 作为推荐器类型。

**基本启动命令**:
```bash
python src/main.py keyword
```

### 测试用例与命令行示例
由于无法在此报告中直接动态运行代码，以下将通过命令行交互的模拟形式，展示推荐系统的使用方法和计划测试的关键词类型。

**启动与交互流程**:
1.  打开终端或命令行界面。
2.  导航到项目根目录 (`Movie-Recommender/`)。
3.  执行如 `python src/main.py keyword [可选参数]` 命令。
4.  系统会首先加载数据并初始化推荐器。
5.  初始化完成后，系统会提示用户输入关键词，例如：
    ```
    Enter keywords (comma-separated), or type 'exit' to quit:
    ```
6.  用户输入关键词后按回车，系统将输出推荐结果。

**计划测试的关键词场景与示例**:

**场景 1: 测试常见、高频关键词**
*   **目的**: 观察推荐系统如何处理广泛存在于多部电影中的常见关键词，以及 `alpha` 和 `beta` 权重如何影响这类查询的结果。
*   **示例关键词输入**:
    *   `action`
    *   `comedy`
    *   `drama`
    *   `love`
    然后当提示时，依次输入上述单个关键词。
*   **结果**:
    ![Action](../images/yang_action.png)
    ![Action](../images/yang_action_2.png)
    ![Action](../images/yang_action_3.png)


**场景 2: 测试常见关键词的组合**
*   **目的**: 检验系统处理多个常见关键词组合的能力，观察推荐结果是否能较好地反映这些关键词的交集特性。
*   **示例关键词输入**:
    *   `action, comedy`
    *   `drama, romance`
    *   `sci-fi, adventure`
*   **命令行示例 (使用默认参数)**:
    ```bash
    python src/main.py keyword
    ```
    然后当提示时，输入上述关键词组合。
*   **预期结果位置**:
    ![drama, romance](../images/yang_comb_1.png)
    ![drama, romance](../images/yang_comb_2.png)
    ![drama, romance](../images/yang_comb_3.png)
    

**场景 3: 测试不常见关键词的组合**
*   **目的**: 观察系统如何处理多个不常见关键词的组合，这可能导致非常小众或精确的推荐结果。
*   **示例关键词输入**:
    *   `dystopia, mockumentary`
*   **命令行示例 (可能调高 alpha 权重)**:
    ```bash
    python src/main.py keyword --alpha 0.8 --beta 0.2
    ```
    然后当提示时，输入上述关键词组合。
![dystopia, mockumentary](../images/yang_rare_1.png)
![dystopia, mockumentary](../images/yang_rare_2.png)
![dystopia, mockumentary](../images/yang_rare_3.png)


## 讨论 (Discussion)

本章节将对关键词推荐器的特性、参数影响、优点和局限性进行讨论。

### 参数影响分析

关键词推荐器的核心行为受到几个关键参数的显著影响：

*   **`alpha` 和 `beta` 权重**:
    *   这两个参数共同决定了关键词相关性评分 (Normalized KRS) 和电影的IMDB加权评分 (Normalized IMDB Score) 在最终推荐分数中的相对重要性。
    *   **高 `alpha` 值 (例如 `alpha=0.9, beta=0.1`)**: 会使推荐结果更侧重于与用户输入关键词高度匹配的电影。即使某些电影的IMDB评分不是顶级，但只要它们与查询关键词非常相关（拥有多个高IDF值的匹配关键词），它们就可能排名靠前。这种设置适合那些对自己想看的内容有明确关键词指向的用户。
    *   **高 `beta` 值 (例如 `alpha=0.1, beta=0.9`)**: 会使推荐结果更偏向那些本身具有较高IMDB加权评分的电影，即那些被大众认为质量较高或非常受欢迎的影片。在这种设置下，关键词的匹配度虽然仍然是考虑因素，但其影响力会减弱。如果用户输入的关键词匹配到的电影普遍评分不高，那么最终推荐列表可能会被高分但关键词匹配度不高的电影占据。
    *   **均衡的 `alpha` 和 `beta` (例如 `alpha=0.5, beta=0.5` 或默认的 `alpha=0.7, beta=0.3`)**: 试图在关键词的精确匹配和电影的普适流行度之间取得平衡。默认值略微偏向关键词匹配，这符合关键词推荐器的初衷。
    *   在实验中，通过调整 `alpha` 和 `beta` 并观察同一组关键词查询下的推荐列表变化，可以清晰地看到这种权重调整的效果。

*   **`kw-vote-percentile` (用于IMDB评分计算)**:
    *   此参数定义了在计算IMDB加权评分时，用于筛选电影的最小投票数阈值所对应的百分位。例如，`0.85` 表示只考虑那些投票数超过数据集中85%的电影的评分。
    *   **较高的 `kw-vote-percentile` (例如 `0.90` 或 `0.95`)**: 会使得IMDB加权评分的计算更为严格，只有那些拥有非常多投票数（通常意味着更广为人知和经过更多人评价）的电影才会被纳入考量。这可以提高IMDB评分的"可靠性"，但也可能排除掉一些小众但高质量的电影。
    *   **较低的 `kw-vote-percentile` (例如 `0.70` 或 `0.50`)**: 会放宽对投票数的要求，使得更多电影（包括一些投票数较少的）能参与IMDB加权评分的计算。这可能引入一些评分人数较少带来的不确定性，但也可能让一些冷门佳片有机会进入推荐。
    *   这个参数主要影响 `normalized_imdb_score` 的分布和值，间接影响最终的 `final_score`。

### 推荐器优点

*   **直观且易于理解**: 用户通过输入自己感兴趣的关键词来获取推荐，这种交互方式非常直接。推荐结果也可以通过展示匹配的关键词和各项得分（KRS、IMDB分）来提供一定的可解释性。
*   **对冷启动友好**: 对于新用户，系统不需要其历史行为数据即可进行推荐，只需要用户提供当前的兴趣关键词。这解决了许多推荐系统中常见的用户冷启动问题。对于新加入系统的电影，只要其关键词被正确索引，也能很快被推荐出来。


### 推荐器局限性

*   **关键词的字面匹配**:
    *   当前的实现主要基于关键词的字面完全匹配。它无法理解同义词（如 "sci-fi" vs "science fiction" 如果未统一处理）、近义词或更深层次的语义关联。用户可能需要输入非常精确的关键词才能获得最佳匹配。
    *   对于用户输入的多关键词查询，当前是简单地将匹配到的关键词IDF值相加，没有考虑关键词之间的顺序或组合关系。
*   **结果多样性可能不足**: 如果某些高IDF的关键词或高IMDB评分的电影主导了推荐分数，可能会导致推荐结果的集中化，缺乏多样性。
