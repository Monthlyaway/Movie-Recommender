# 电影推荐系统

这是一个基于Python开发的电影推荐系统，由我们小组共同完成。该系统主要面向电影爱好者，通过多种推荐算法为用户提供个性化的电影推荐。

## 项目概述

我们小组尝试使用了四种不同的方法来构建电影推荐系统，每种方法由小组的一位成员负责开发和实现。这些方法各有特色，适用于不同的推荐场景：

1. [**加权评分推荐器**](docs/recommenders/simple_recommender.md) - 基于IMDB加权评分公式，根据用户评分和投票数量推荐高质量电影
2. [**情节相似度推荐器**](docs/recommenders/plot_recommender.md) - 通过分析电影情节内容，推荐具有相似故事情节的电影
3. [**关键词推荐器**](docs/recommenders/keyword_recommender.md) - 利用电影的关键词标签，为用户推荐具有相似主题和元素的电影
4. [**频繁项集推荐器**](docs/recommenders/association_recommender.md) - 基于用户的观影历史，利用频繁模式挖掘推荐经常一起被喜欢的电影

## 数据集

本项目使用了以下[数据集](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download)：
- MovieLens 数据集（包含电影元数据和用户评分）
- IMDB 数据（用于补充电影信息）

## 系统要求

- Python 3.11+
- 相关依赖库（见requirements.txt文件）

## 使用方法

```bash
# 安装依赖
pip install -r requirements.txt

# 运行推荐系统
python src/main.py --help
```

可用的推荐器模式：
```bash
# 加权评分推荐器
python src/main.py simple

# 情节相似度推荐器
python src/main.py plot

# 关键词推荐器
python src/main.py keyword

# 关联规则推荐器
python src/main.py association
```

有哪些选项可以选？

```bash
python src/main.py --help
python src/main.py simple --help
python src/main.py plot --help
python src/main.py keyword --help
python src/main.py association --help
```



## 项目结构

```
├── dataset/           # 数据集文件
├── docs/              # 文档
│   └── recommenders/  # 各推荐器详细文档
├── src/               # 源代码
│   ├── recommenders/  # 推荐器实现
│   ├── utils/         # 工具函数
│   ├── ui/            # 用户界面
│   └── main.py        # 主程序入口
└── requirements.txt   # 项目依赖
```

## 团队成员

- 陈可豪：负责加权评分推荐器(Simple Recommender)
- 王海天：负责情节相似度推荐器(Plot Recommender)
- 杨淳瑜：负责关键词推荐器(Keyword Recommender)
- 王熙同：负责关联规则推荐器(Association Recommender)
