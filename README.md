# IMDB Movie Rating Prediction

Regression of a movie's IMDB score from metadata such as budget, gross, vote and review counts, Facebook likes,
director, cast, genre and country. Eight scikit-learn/XGBoost regressors are compared on a held-out test set.

Practice project (2021).

## Data

`movie_metadata.csv` (about 1.5 MB, in the repo): 5,043 movies and 28 columns. The notebook links the Kaggle
[IMDB Extensive Dataset](https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset) as its source; the committed
file has the column layout of the Kaggle "IMDB 5000 Movie Dataset" (`movie_metadata.csv`).

## Approach

1. EDA: pair plot, score distribution, counts by language and director, scores by country, content rating and
   aspect ratio; a short look at Hindi-language films with scores of 7 or more.
2. Rows with any missing value are dropped (5,043 to 3,756) and 8 columns are removed (title, link, color, language,
   aspect ratio, plot keywords, face count in poster, third actor's name), leaving 19 features.
3. The 6 text columns (director, actor 1, actor 2, genres, country, content rating) are label-encoded.
4. `log(x + 1)` is applied to every column, including the label-encoded ones and the target `imdb_score`.
5. 80/20 random split (3,004 train, 752 test), default hyperparameters except a depth-9 decision tree and
   10-fold internal CV for Lasso, Ridge and ElasticNet.

## Results

R² on the 752 test movies (target is `log(imdb_score + 1)`):

| Model | Train R² | Test R² |
|---|---|---|
| XGBoost | 0.989 | **0.612** |
| Random Forest | 0.930 | 0.571 |
| Ridge (CV) | 0.400 | 0.410 |
| Lasso (CV) | 0.393 | 0.407 |
| ElasticNet (CV) | 0.393 | 0.407 |
| Linear Regression | 0.408 | 0.402 |
| Decision Tree (depth 9) | 0.719 | 0.315 |
| KNN (k=5) | 0.519 | 0.262 |

The best model, XGBoost, explains about 61% of the variance in the (log) score on unseen movies, with a test RMSE of
0.095 on the log scale. Both tree ensembles overfit heavily (train R² 0.93–0.99).

## How to run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
jupyter notebook "IMDB Movie Ratings Prediction.ipynb"
```

## Notes

- **The "Accuracy: 95–96%" lines in the notebook are not a classification accuracy.** They are `100 − MAPE`, computed
  on the log-transformed score. The log squeezes the 1–10 rating scale into roughly 0.7–2.4, so percentage errors are
  small for almost any model. R² (above) is the meaningful measure here.
- Label-encoded names (directors, actors, genres) are arbitrary integers, so taking their log does not make sense;
  one-hot or target encoding would be better choices for those columns.
- Several features (vote counts, user review counts, Facebook likes, gross) are only known after a film is released,
  so the model estimates a rating from popularity signals rather than predicting it before release.
- No cross-validation or hyperparameter tuning for the main models; results come from a single split.
