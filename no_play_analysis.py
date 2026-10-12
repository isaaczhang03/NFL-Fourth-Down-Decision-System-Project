import pandas as pd
from load_data import load_pbp

pbp = load_pbp()

# Sort events in order within each game
pbp = pbp.sort_values(
    ["game_id", "order_sequence", "play_id"]
).reset_index(drop=True)

# Identify fourth-down no-play records
mask = (
    (pbp["down"] == 4) &
    (pbp["play_type"] == "no_play")
)

# Identify the next recorded event within each game
next_play = pbp.groupby("game_id").shift(-1)

no_plays = pbp.loc[mask].copy()

no_plays["next_down"] = next_play.loc[mask, "down"]
no_plays["next_play_type"] = next_play.loc[mask, "play_type"]
no_plays["next_posteam"] = next_play.loc[mask, "posteam"]

# Does the same offense remain on fourth down?
no_plays["repeated_fourth_down"] = (
    (no_plays["posteam"] == no_plays["next_posteam"]) &
    (no_plays["next_down"] == 4)
)

print("Total no-play records:", len(no_plays))

print("\nImmediately followed by another fourth down:")
print(no_plays["repeated_fourth_down"].value_counts())

print("\nNext down:")
print(no_plays["next_down"].value_counts(dropna=False))

print("\nNext play type:")
print(no_plays["next_play_type"].value_counts(dropna=False))

print("\nExamples:")
print(
    no_plays[
        [
            "game_id",
            "penalty_type",
            "ydstogo",
            "yardline_100",
            "next_down",
            "next_play_type",
            "repeated_fourth_down"
        ]
    ].head(15).to_string(index=False)
)