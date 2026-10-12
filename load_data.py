import nflreadpy as nfl

def load_pbp():
    return nfl.load_pbp([2022, 2023, 2024, 2025]).to_pandas()

def load_fourth_downs():
    pbp = load_pbp()

    fourth_downs = pbp[
        (pbp["down"] == 4) &
        (pbp["play_type"].isin(["run", "pass", "punt", "field_goal"]))
    ].copy()

    return fourth_downs


if __name__ == "__main__":
    pbp = load_pbp()
    fourth_downs = load_fourth_downs()

    print(fourth_downs.shape)
    print(fourth_downs["play_type"].value_counts())