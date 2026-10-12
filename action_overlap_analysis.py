import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_fourth_downs

df = load_fourth_downs()

# Group runs and passes together as go for it
df["action"] = df["play_type"].replace({
    "punt": "Punt",
    "field_goal": "Field Goal",
    "pass": "Go For It",
    "run": "Go For It"
})

# Variables that describe the game situation
features = [
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining"
]

# Compare the game situations for each action
print("Game state summary by action:")
for feature in features:
    print("\n", feature)
    print(df.groupby("action")[feature].describe().round(2))

# Check for missing values
print("\nMissing values:")
for feature in features:
    missing = df[feature].isna().sum()
    percent = missing / len(df) * 100
    print(feature, ":", missing, "missing (", round(percent, 2), "% )")

# Divide the field into 20-yard sections
df["field_zone"] = pd.cut(
    df["yardline_100"],
    bins=[0, 20, 40, 60, 80, 100],
    include_lowest=True
)

# Count the number of each action in each field zone
print("\nAction counts by field position:")
counts = pd.crosstab(df["field_zone"], df["action"])
print(counts)

# Find the percentage of each action in each field zone
print("\nAction percentages by field position:")
percentages = pd.crosstab(df["field_zone"], df["action"], normalize="index") * 100
print(percentages.round(2))

# Create distance groups
df["distance_group"] = pd.cut(
    df["ydstogo"],
    bins=[-1, 1, 3, 5, 10, 100],
    labels=["1", "2-3", "4-5", "6-10", "11+"]
)

# Count actions for each combination
counts = pd.crosstab(
    [df["field_zone"], df["distance_group"]],
    df["action"]
)

print("\nAction counts by field position and distance:")
print(counts.to_string())

# Number of distinct actions observed in each situation
overlap = (counts > 0).sum(axis=1)
overlap_table = overlap.unstack()

print("\nNumber of observed actions per situation:")
print(overlap_table)

# Plot overlap heatmap
plt.figure(figsize=(10, 6))
plt.imshow(
    overlap_table,
    aspect="auto",
    vmin=0,
    vmax=3
)

plt.colorbar(label="Number of observed actions")
plt.xticks(
    range(len(overlap_table.columns)),
    overlap_table.columns
)
plt.yticks(
    range(len(overlap_table.index)),
    overlap_table.index.astype(str)
)

plt.xlabel("Yards to gain")
plt.ylabel("Yards from opponent end zone")
plt.title("Fourth-Down Action Overlap by Game State")
plt.tight_layout()
plt.show()

# Calculate historical action probabilities in each region
counts = pd.crosstab(
    [df["field_zone"], df["distance_group"]],
    df["action"]
)

# Convert counts to probabilities
propensities = counts.div(counts.sum(axis=1), axis=0)

print("\nHistorical action probabilities:")
print(propensities.round(3).to_string())

# Check how many observations support each action
min_counts = counts.min(axis=1)

print("\nRegions with at least 20 observations per action:")
supported = counts[min_counts >= 20]
print(supported.to_string())

# Percentage of observations in these supported regions
region_totals = counts.sum(axis=1)

supported_plays = region_totals[min_counts >= 20].sum()
total_plays = len(df)

print("\nObservations in supported regions:", supported_plays)
print("Percent of sample:", round(supported_plays / total_plays * 100, 2))