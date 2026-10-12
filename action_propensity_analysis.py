import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from load_data import load_fourth_downs

df = load_fourth_downs()

df["action"] = df["play_type"].replace({
    "punt": "Punt",
    "field_goal": "Field Goal",
    "pass": "Go For It",
    "run": "Go For It"
})

features = [
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining"
]

# Use earlier seasons to estimate selection patterns
train = df[df["season"] < 2025].copy()
test = df[df["season"] == 2025].copy()

X_train = train[features]
y_train = train["action"]

X_test = test[features]
y_test = test["action"]

# Standardize inputs and estimate historical action probabilities
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

# Predict action probabilities on the held-out season
probabilities = model.predict_proba(X_test)
actions = model.classes_

results = pd.DataFrame(probabilities, columns=actions)
print("\nEstimated action probabilities:")
print(results.head(10).round(3).to_string(index=False))

# Summarize the model's predicted probability ranges
print("\nProbability distributions:")
print(results.describe().round(3).to_string())

# Count situations where all actions receive at least 5% probability
supported = (results >= 0.05).all(axis=1)

print("\nObservations with at least 5% estimated probability for every action:")
print(supported.sum())
print("Percentage:", round(supported.mean() * 100, 2))

print("\nOverlap sensitivity analysis:")

for threshold in [0.01, 0.05, 0.10]:
    supported = (results >= threshold).all(axis=1)

    count = supported.sum()
    percent = supported.mean() * 100

    print(
        f"Threshold: {threshold:.0%} | "
        f"Observations: {count} | "
        f"Percentage: {percent:.2f}%"
    )

# Multiclass log loss
loss = log_loss(y_test, probabilities, labels=actions)
print("2025 multiclass log loss:", round(loss, 4))

# Compare model probabilities against overall action frequencies
print("\nActual vs predicted action frequencies:")

for i, action in enumerate(actions):
    actual = (y_test == action).mean()
    predicted = probabilities[:, i].mean()

    print(action)
    print("Actual:", round(actual, 3))
    print("Predicted:", round(predicted, 3))

# Calibration for each action
for i, action in enumerate(actions):
    calibration = pd.DataFrame({
        "predicted": probabilities[:, i],
        "actual": (y_test.to_numpy() == action).astype(int)
    })

    calibration["bin"] = pd.cut(
        calibration["predicted"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True
    )

    summary = calibration.groupby(
        "bin", observed=True
    )[["predicted", "actual"]].mean()

    plt.figure(figsize=(6, 5))
    plt.plot(summary["predicted"], summary["actual"], marker="o")
    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("Average predicted probability")
    plt.ylabel("Observed action frequency")
    plt.title(action + " Propensity Calibration")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()