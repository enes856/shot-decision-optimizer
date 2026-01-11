# ---
# jupyter:
#   kernelspec:
#     display_name: Python (shot-decision-optimizer)
#     language: python
#     name: shot-decision-optimizer
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.12.6
# ---

# %% [markdown]
# # Explorative Datenanalyse der NBA Wurfdaten
#
# This script performs an exploratory data analysis (EDA) of the file
# `nba_raw_shot_data.parquet`. The focus is on impact value analytics –
# insights that directly inform the assessment and improvement of shot quality.
#
# The script is jupytext-compatible and can be synced as a notebook.

# %% [markdown]
# ## Import der nötigen Bibliotheken

# %%
import pandas as pd
pd.set_option('display.max_columns', None)
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append("../utils")
from utils import *

# %% [markdown]
# ## Setup
# - Seaborn and Matplotlib styles
# - Output directory for figures

# %%
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})

FIGURES_DIR = Path("../figures/eda")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Laden der Daten

# %%
shot_data = load_data(file_path="../data/raw/nba_raw_shot_data.parquet")

# %% [markdown]
# ## Teil 1: Allgemeine Datenübersicht

# %% [markdown]
# ### Datenzusammenfassung

# %%
summarize_data(shot_data)

# %% [markdown]
# ### Detailiertere Untersuchung

# %%
explore_data(shot_data)

# %% [markdown]
# ### Duplikat-Analyse

# %%
summarize_duplicates(shot_data)

# %% [markdown]
# ### Datentypen und Statistiken

# %%
summarize_data_types_and_stats(shot_data)

# %% [markdown]
# ### Ausreißer-Analyse

# %%
detect_outliers_iqr(shot_data)

# %% [markdown]
# ## Teil 2: Erweiterte Analyse und Visualisierungen

# %%
plot_distance_profile(shot_data)

# %%
plot_zone_efficiency(shot_data)

# %%
plot_time_pressure(shot_data)

# %% [markdown]
# ## Teil 3: Interaktive Dashboards

# %%
PLAYERS_OF_INTEREST = [
    "LeBron James",
    "Stephen Curry",
    "James Harden",
    "Kevin Durant",
    "Paul George",
    "Nikola Jokic"
]

# Filter to available players
available_players = [
    p for p in PLAYERS_OF_INTEREST 
    if p in shot_data['PLAYER_NAME'].values
]

TEAM_NAMES = sorted(shot_data['TEAM_NAME'].unique())
SEASONS = sorted(shot_data['SEASON'].unique())

print(f"Available players: {available_players}")
print(f"Available teams: {len(TEAM_NAMES)}")
print(f"Available seasons: {SEASONS}")

# %% [markdown]
# ### HEX Shot Chart

# %%
# Dropdown for players
player_dropdown = Dropdown(
    options=available_players,
    description='Player:',
    style={'description_width': '120px'}
)

# Multi-select for seasons
season_select = SelectMultiple(
    options=SEASONS,
    value=[SEASONS[-1]],  # Default: last season
    description='Seasons:',
    rows=3,
    style={'description_width': '120px'}
)

# Output for plot
output_shot_chart = Output()

def update_shot_chart(player, seasons):
    """Update shot chart based on widget values."""
    with output_shot_chart:
        output_shot_chart.clear_output(wait=True)
        if not seasons:
            print("Please select at least one season.")
        else:
            fig = plot_hex_shot_chart(shot_data, player, list(seasons))
            fig.show()

# Widget observers
player_dropdown.observe(
    lambda change: update_shot_chart(player_dropdown.value, season_select.value),
    names='value'
)
season_select.observe(
    lambda change: update_shot_chart(player_dropdown.value, season_select.value),
    names='value'
)

# Layout for shot chart
shot_chart_controls = VBox([
    HBox([player_dropdown, season_select]),
    output_shot_chart
])

print("SHOT CHART")
shot_chart_controls

# %%
update_shot_chart(player_dropdown.value, season_select.value)

# %%
# Dropdown for teams
team_dropdown = Dropdown(
    options=TEAM_NAMES,
    description='Team:',
    style={'description_width': '120px'}
)

# Dropdown for season
season_dropdown = Dropdown(
    options=SEASONS,
    value=SEASONS[-1],  # Default: last season
    description='Season:',
    style={'description_width': '120px'}
)

# Output for PPP plot
output_ppp_heatmap = Output()

def update_ppp_heatmap(team, season):
    """Update PPP heatmap based on widget values."""
    with output_ppp_heatmap:
        output_ppp_heatmap.clear_output(wait=True)
        ppp_matrix = calculate_ppp(shot_data, team, season, top_n=10)
        fig = plot_ppp_heatmap(ppp_matrix, team, season)
        fig.show()

# Widget observers
team_dropdown.observe(
    lambda change: update_ppp_heatmap(team_dropdown.value, season_dropdown.value),
    names='value'
)
season_dropdown.observe(
    lambda change: update_ppp_heatmap(team_dropdown.value, season_dropdown.value),
    names='value'
)

# Layout for PPP heatmap
ppp_controls = VBox([
    HBox([team_dropdown, season_dropdown]),
    output_ppp_heatmap
])

print("PPP HEATMAP")
ppp_controls

# %%
update_ppp_heatmap(team_dropdown.value, season_dropdown.value)

# %%
