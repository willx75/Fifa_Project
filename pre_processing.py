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

    df["contract_remaining_year"] = (df["contract_valid_until"] - df["year"]).clip(0).fillna(0)
    return df


def remove_free_players(df: pd.DataFrame):
    return df[df["contract"] > 0]


def mean_per_category(df: pd.DataFrame, prefix: str):
    """
    Average all variables beginning with prefix into prefix variable.
    Exemple : movement_acceleration, movement_print_speed, movement_agility... 
    are averaged inside movement.
    """

    list_keys = [key for key in df.keys() if key.startswith(prefix)]
    df[prefix] = df[list_keys].fillna(0).mean(axis=1)
    return df


def group_team_position(df: pd.DataFrame):
    F = "F"
    M = "M"
    B = "B"
    GK = "GK"
    mapping = dict(W=F, T=F, F=F, S=F, M=M, B=B, K=GK)

    def _map(x):
        if x != x:
            # NaN, indicates no position
            return "N"
        return mapping[x[-1]]
    df["team_position"] = df["team_position"].apply(_map)
    return df
