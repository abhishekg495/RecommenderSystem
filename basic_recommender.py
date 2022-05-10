import numpy as np
import pandas as pd


def best_average_rating(movies_cleaned_df):
    v = movies_cleaned_df["vote_count"]
    R = movies_cleaned_df["vote_average"]
    C = R.mean()
    m = v.quantile(0.75)

    movies_cleaned_df["weighted_average"] = ((R * v) + (C * m)) / (v + m)
    movies_sorted_ranking = movies_cleaned_df.sort_values(
        "weighted_average", ascending=False
    )
    return movies_sorted_ranking[["original_title", "weighted_average","link"]].head(20)
