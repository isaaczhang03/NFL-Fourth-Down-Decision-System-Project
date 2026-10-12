import pandas as pd
from sklearn.metrics import brier_score_loss
from load_data import load_fourth_downs

df = load_fourth_downs()

# Separate the three actions
go = df[df["play_type"].isin(["run", "pass"])].copy()
fg = df[df["play_type"] == "field_goal"].copy()
punt = df[df["play_type"] == "punt"].copy()

# Use older seasons to construct baseline estimates
go_train = go[go["season"] < 2025].copy()
fg_train = fg[fg["season"] < 2025].copy()
punt_train = punt[punt["season"] < 2025].copy()

# Go-for-it conversion rate by distance
go_train = go_train[
    (go_train["fourth_down_converted"] == 1) |
    (go_train["fourth_down_failed"] == 1)
]

go_train["distance_group"] = pd.cut(
    go_train["ydstogo"],
    bins=[-1, 1, 3, 5, 10, 100],
    labels=["1", "2-3", "4-5", "6-10", "11+"]
)

print("\nGo-for-it conversion baseline:")
print(
    go_train.groupby("distance_group", observed=True)[
        "fourth_down_converted"
    ].agg(["count", "mean"]).round(3)
)

# Field-goal make rate by kick distance
fg_train["kick_group"] = pd.cut(
    fg_train["kick_distance"],
    bins=[0, 30, 40, 50, 60, 100],
    include_lowest=True
)

fg_train["made"] = (
    fg_train["field_goal_result"] == "made"
).astype(int)

print("\nField-goal success baseline:")
print(
    fg_train.groupby("kick_group", observed=True)["made"]
    .agg(["count", "mean"]).round(3)
)

# Punt baseline: net change in field position
# This is only a preliminary approximation.
punt_train["estimated_net_distance"] = (
    punt_train["kick_distance"] -
    punt_train["return_yards"].fillna(0)
)

print("\nPunt distance baseline:")
print(
    punt_train["estimated_net_distance"].describe().round(2)
)

punt_train["estimated_net_distance"] = (
    punt_train["kick_distance"] -
    punt_train["return_yards"].fillna(0)
)

negative = punt_train[punt_train["estimated_net_distance"] < 0]

print("Negative net distances:", len(negative))

print(
    negative[
        [
            "game_id",
            "play_id",
            "desc",
            "kick_distance",
            "return_yards",
            "punt_blocked",
            "touchback",
            "penalty"
        ]
    ].head(15).to_string(index=False)
)

# Evaluate go-for-it baseline on 2025
go_test = go[go["season"] == 2025].copy()

go_test = go_test[
    (go_test["fourth_down_converted"] == 1) |
    (go_test["fourth_down_failed"] == 1)
].copy()

go_test["distance_group"] = pd.cut(
    go_test["ydstogo"],
    bins=[-1, 1, 3, 5, 10, 100],
    labels=["1", "2-3", "4-5", "6-10", "11+"]
)

go_rates = go_train.groupby(
    "distance_group", observed=True
)["fourth_down_converted"].mean()

go_test["prediction"] = (
    go_test["distance_group"].map(go_rates).astype(float)
)

go_score = brier_score_loss(
    go_test["fourth_down_converted"],
    go_test["prediction"]
)

print("Go-for-it Brier score:", round(go_score, 4))


# Evaluate field-goal baseline on 2025
fg_test = fg[fg["season"] == 2025].copy()

fg_test["kick_group"] = pd.cut(
    fg_test["kick_distance"],
    bins=[0, 30, 40, 50, 60, 100],
    include_lowest=True
)

fg_test["made"] = (
    fg_test["field_goal_result"] == "made"
).astype(int)

fg_rates = fg_train.groupby(
    "kick_group", observed=True
)["made"].mean()

fg_test["prediction"] = (
    fg_test["kick_group"].map(fg_rates).astype(float)
)

valid = fg_test["prediction"].notna()

fg_score = brier_score_loss(
    fg_test.loc[valid, "made"],
    fg_test.loc[valid, "prediction"]
)

print("Field-goal Brier score:", round(fg_score, 4))
print("Field goals evaluated:", valid.sum())
print("Field goals excluded:", (~valid).sum())