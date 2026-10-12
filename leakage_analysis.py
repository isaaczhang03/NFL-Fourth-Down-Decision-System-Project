from load_data import load_fourth_downs

df = load_fourth_downs()

# Variables available before the fourth-down decision
pre_play = [
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "posteam",
    "defteam",
    "season",
    "week"
]

# Variables determined after the play
post_play = [
    "yards_gained",
    "fourth_down_converted",
    "fourth_down_failed",
    "field_goal_result",
    "wpa",
    "epa",
    "score_differential_post",
    "posteam_score_post"
]

print("Pre-play variables:")
for col in pre_play:
    print(col, "Missing:", df[col].isna().sum())

print("\nPost-play variables (outcomes only):")
for col in post_play:
    print(col, "Missing:", df[col].isna().sum())

# Make sure the predictor list contains no outcome variables
leakage = set(pre_play).intersection(post_play)

print("\nVariables appearing in both lists:", leakage)