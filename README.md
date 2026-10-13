# NFL Fourth-Down Decision System

## Project Overview

This project examines NFL fourth-down decisions to determine whether a team should punt, attempt a field goal, or go for it to maximize its probability of winning.

For Milestone 2, the focus is on exploring historical data, understanding coaching decisions, checking data quality, and developing preliminary models.

## Data

The project uses NFL play-by-play data from the **2022–2025 seasons**, accessed through the `nflreadpy` Python package.

The dataset includes **16,287 classified fourth-down plays** across 1,139 games.

## Repository Contents

The repository contains Python scripts covering:

- Data loading and fourth-down identification
- Data quality and missingness
- Dependence and historical action selection
- Action overlap and propensity analysis
- Data leakage checks
- Preliminary baseline models

The Milestone 2 dossier summarizes the methods, results, and limitations.

## How to Run

Install the required Python libraries:

```bash
pip install nflreadpy pandas numpy matplotlib scikit-learn jupyter
```

Download the repository and run the Python scripts from the main project folder. Start with `load_data.py`, which provides the data-loading functions used by the other scripts.

If using the analysis notebook, open it in Jupyter and select **Restart Kernel and Run All Cells**.

## Preliminary Results

The analysis found that fourth-down decisions depend heavily on field position and yards to gain. Historical observations also show limited overlap between the three actions in certain game situations.

The preliminary models achieved the following 2025 Brier scores:

- **Go-for-it conversion:** 0.2303
- **Field-goal success:** 0.1094

## Next Steps

Future work will improve outcome predictions, address limitations in historical action selection, and develop a system that compares the expected win probabilities of the three fourth-down options.

## Data Source

[nflverse](https://github.com/nflverse) — NFL play-by-play data.