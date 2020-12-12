"""
Contain all functions to process csv data.
"""

from typing import List
import re

import pandas as pd


def fusion_file(list_path: List[str], new_filepath: str):
    list_df = list()
    for path in list_path:
        df = pd.read_csv(path)
        # Extract year from file name
        year = int(re.search(r'\d+', path).group())
        df["year"] = 2000 + year
        list_df.append(df)

    final_df = pd.concat(list_df, ignore_index=True)
    final_df.to_csv(new_filepath)


def compute_remaining_contract_years(df: pd.DataFrame):

    df["contract"] = (df["contract_valid_until"] - df["year"]).clip(0).fillna(0)
    return df


def remove_free_players(df: pd.DataFrame):
    return df[df["contract"] > 0]

