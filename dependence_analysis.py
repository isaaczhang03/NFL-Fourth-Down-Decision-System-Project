import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_fourth_downs

df = load_fourth_downs()

df["action"] = df["play_type"].replace({
    "punt": "Punt",
    "field_goal": "Field Goal",
    "pass": "Go For It",
    "run": "Go For It"
})

# Number of fourth-down plays in each game
games = df.groupby("game_id").size()

print("\nFourth-down observations per game:")
print(games.describe().round(2))

print("\nTotal games:", df["game_id"].nunique())
print("Games with multiple fourth downs:", (games > 1).sum())

# Number of fourth-down plays per offensive team
teams = df.groupby("posteam").size()

print("\nFourth-down observations per team:")
print(teams.sort_values(ascending=False).to_string())

# Action frequencies by season
print("\nAction counts by season:")
season_counts = pd.crosstab(df["season"], df["action"])
print(season_counts.to_string())

print("\nAction percentages by season:")
season_pct = pd.crosstab(
    df["season"],
    df["action"],
    normalize="index"
) * 100

print(season_pct.round(2).to_string())

# Number of fourth-down plays per team per season
team_season = df.groupby(["season", "posteam"]).size()

print("\nTeam-season observation counts:")
print(team_season.describe().round(2))

# Check for duplicated identifiers
duplicates = df.duplicated(subset=["game_id", "play_id"]).sum()

print("\nDuplicate game/play identifiers:", duplicates)

season_pct.plot(
    kind="line",
    marker="o",
    figsize=(8, 5)
)

plt.xlabel("NFL Season")
plt.ylabel("Percentage of Fourth-Down Plays")
plt.title("Fourth-Down Action Selection by Season")
plt.xticks([2022, 2023, 2024, 2025])
plt.grid(alpha=0.3)
plt.legend(title="Action")
plt.tight_layout()
plt.show()