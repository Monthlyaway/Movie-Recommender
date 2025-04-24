# Movie Recommender Usage Guide

This document explains how to use the Movie Recommender System from the command line.

## Basic Usage

```bash
python src/main.py [RECOMMENDER] [OPTIONS]
```

Where `[RECOMMENDER]` is one of: `plot`, `simple`, or `association`.

## Global Options

The following options apply to all recommenders:

```bash
--use-name-removed     Use the version of metadata with person names removed from plot overviews
--help, -h             Show help message
```

## Recommender Types

### Plot Recommender (Default)

The Plot Recommender suggests movies with similar plot content.

```bash
python src/main.py plot
```

### Simple Recommender

The Simple Recommender ranks movies by popularity using the IMDB weighted rating formula.

```bash
python src/main.py simple [OPTIONS]
```

Options:
```
--percentile FLOAT     Vote count percentile threshold (default: 0.90)
```

### Association Recommender

The Association Recommender finds movies that users tend to like together based on rating patterns.

```bash
python src/main.py association [OPTIONS]
```

Options:
```
--min-support FLOAT       Minimum support threshold (default: 0.06)
--min-confidence FLOAT    Minimum confidence threshold (default: 0.3)
--min-lift FLOAT          Minimum lift threshold (default: 1.2)
--rating-threshold FLOAT  Minimum rating to consider a movie as liked (default: 3.5)
```

## Examples

1. **Run the default Plot Recommender**:
   ```bash
   python src/main.py
   ```

2. **Run the Simple Recommender with a different percentile**:
   ```bash
   python src/main.py simple --percentile 0.95
   ```

3. **Run the Association Recommender with custom parameters**:
   ```bash
   python src/main.py association --min-support 0.05 --min-confidence 0.4 --rating-threshold 4.0
   ```

4. **Run any recommender with person names removed from plot overviews**:
   ```bash
   python src/main.py plot --use-name-removed
   ```

## Interactive Interface

After starting the recommender, you'll be presented with an interactive command-line interface. The available options depend on which recommender you're using:

- **Plot Recommender**: Toggle plot display, search for movies by title
- **Simple Recommender**: View top-rated movies
- **Association Recommender**: View random association rules, search for movies by title

Follow the on-screen prompts to interact with the system. 