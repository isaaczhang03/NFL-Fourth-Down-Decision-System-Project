import pandas as pd
from load_data import load_fourth_downs

df = load_fourth_downs()

print("Total fourth-down observations:", len(df))

# Variables relevant to the project
columns = [
    "game_id",
    "play_id",
    "season",
    "posteam",
    "defteam",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "play_type",
    "yards_gained",
    "fourth_down_converted",
    "fourth_down_failed",
    "field_goal_result",
    "wp",
    "wpa"
]

# Check missing values
print("\nMissing values:")
for col in columns:
    missing = df[col].isna().sum()
    percent = missing / len(df) * 100
    print(col, ":", missing, "(", round(percent, 2), "%)")

# Check duplicate play identifiers
duplicates = df.duplicated(["game_id", "play_id"]).sum()
print("\nDuplicate plays:", duplicates)

# Check impossible game-state values
print("\nUnusual game-state values:")
print("Negative yards to gain:", (df["ydstogo"] < 0).sum())
print("Field position outside 0-100:",
      ((df["yardline_100"] < 0) | (df["yardline_100"] > 100)).sum())
print("Negative game time:", (df["game_seconds_remaining"] < 0).sum())
print("Invalid offensive timeout counts:",
      ((df["posteam_timeouts_remaining"] < 0) |
       (df["posteam_timeouts_remaining"] > 3)).sum())

# Check penalty and unusual-play indicators
print("\nSpecial cases:")
print("Penalties:", (df["penalty"] == 1).sum())
print("Blocked punts:", (df["punt_blocked"] == 1).sum())
print("Aborted plays:", (df["aborted_play"] == 1).sum())
print("Deleted plays:", (df["play_deleted"] == 1).sum())

# Check outcome consistency
print("\nOutcome checks:")
go = df[df["play_type"].isin(["run", "pass"])]

print("Go-for-it observations:", len(go))
print("Marked converted:", (go["fourth_down_converted"] == 1).sum())
print("Marked failed:", (go["fourth_down_failed"] == 1).sum())

print("Neither converted nor failed:",
      ((go["fourth_down_converted"] != 1) &
       (go["fourth_down_failed"] != 1)).sum())

fg = df[df["play_type"] == "field_goal"]

print("\nField-goal results:")
print(fg["field_goal_result"].value_counts(dropna=False))