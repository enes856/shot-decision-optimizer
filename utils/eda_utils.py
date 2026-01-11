import pandas as pd
import duckdb as ddb
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from ipywidgets import Dropdown, SelectMultiple, Output, VBox, HBox, interactive_output
from typing import Iterable, Optional

def summarize_data(data: pd.DataFrame) -> None:
    """
    Generates an overview summary for a single pandas DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to summarize.

    Returns:
        None: Prints the overview summary.
    """
    overview = {
        'Rows': len(data),
        'Columns': data.shape[1],
        'Memory (MB)': round(data.memory_usage(deep=True).sum() / 1024**2, 2)
    }

    print("=" * 80)
    print("DATAFRAME OVERVIEW SUMMARY")
    print("=" * 80)

    overview_df = pd.DataFrame([overview])
    display(overview_df)

    print(f"\nTotal rows: {overview['Rows']:,}".replace(',', '.'))
    print(f"Total memory usage: {overview['Memory (MB)']:.2f} MB")

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads a parquet file into a pandas DataFrame.

    Parameters:
        file_path (str): The path to the parquet file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    data = pd.read_parquet(file_path)
    return data

def explore_data(df: pd.DataFrame) -> None:
    """
    Explore a pandas DataFrame and display key information.

    The function prints:
        - Shape (rows, columns)
        - Column information:
            * Data type
            * Non-null count
            * Null count
            * Null percentage
        - First 3 rows
        - Last 3 rows
        - 5 random sample rows

    Args:
        df (pd.DataFrame): The DataFrame to explore.

    Returns:
        None: Outputs exploration results directly using print() and display().
    """

    # Column info
    print("\nColumn Information:")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values,
        'Null %': (df.isnull().sum() / len(df) * 100).round(2).values
    })
    display(col_info)

    # Sample data
    print("\nFirst 3 Rows:")
    display(df.head(3))

    print("\nLast 3 Rows:")
    display(df.tail(3))

    print("\n5 Random Sample Rows:")
    display(df.sample(5, random_state=42))

def summarize_duplicates(df: pd.DataFrame) -> None:
    """
    Generate and display a summary of duplicate records for a single pandas DataFrame.

    The function prints:
        - Total number of rows
        - Number of duplicate rows (full row duplicates)
        - Percentage of duplicate rows

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        None: Outputs the duplicate summary using print() and display().
    """

    total_rows = len(df)
    duplicate_rows = df.duplicated().sum()
    duplicate_pct = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0

    summary_df = pd.DataFrame([{
        'Total Rows': total_rows,
        'Duplicate Rows': duplicate_rows,
        'Duplicate %': round(duplicate_pct, 2)
    }])

    print("=" * 80)
    print("DUPLICATE RECORDS SUMMARY")
    print("=" * 80)
    display(summary_df)


def summarize_data_types_and_stats(df: pd.DataFrame) -> None:
    """
    Analyze data types, numeric statistics, categorical distributions, 
    and datetime summaries for a given DataFrame.
    
    Args:
        df (pd.DataFrame): The dataset to analyze.

    Returns:
        None: Prints the data type and statistics summary.
    """

    # --- COLUMN TYPE SPLIT ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number, "datetime64[ns]", "datetime64"]).columns.tolist()

    print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Datetime columns ({len(datetime_cols)}): {datetime_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    # --- NUMERIC STATS ---
    if numeric_cols:
        
        # Check for potential issues
        print("\nChecking for potential issues in numeric columns:")
        check_df = pd.DataFrame(columns=["Column", "Count", "Unique Count", "Min", "Max", "Contains Negative?", "Zero Count"])

        for col in numeric_cols:
            count = df[col].count()
            unique_count = df[col].nunique()
            min_val = df[col].min()
            max_val = df[col].max()
            neg = min_val < 0
            zeros = int((df[col] == 0).sum())
            check_df.loc[len(check_df)] = [col, count, unique_count, min_val, max_val, neg, zeros]

        display(check_df)

    # --- DATETIME STATS ---
    if datetime_cols:
        print("\nDatetime Column Summary:")
        datetime_summary = pd.DataFrame(columns=["Column", "Min Date", "Max Date", "Range (Days)", "Missing Values"])
        
        for col in datetime_cols:
            min_date = df[col].min()
            max_date = df[col].max()
            range_days = (max_date - min_date).days if pd.notnull(min_date) and pd.notnull(max_date) else None
            missing = df[col].isnull().sum()
            
            datetime_summary.loc[len(datetime_summary)] = [col, min_date, max_date, range_days, missing]

        display(datetime_summary)

    # --- CATEGORICAL STATS ---
    if categorical_cols:
        print("\nCategorical Column Summary:")
        summary_df = pd.DataFrame(columns=['Column', 'Count', 'Unique Count', 'Top Value 1', 'Top Value 1 Count'])

        for col in categorical_cols:
            count = df[col].count()
            unique_count = df[col].nunique()
            value_counts = df[col].value_counts().head()
            if unique_count > 0:
                top_val = value_counts.index[0]
                top_count = value_counts.iloc[0]
            else:
                top_val, top_count = "N/A", 0
            summary_df.loc[len(summary_df)] = [col, count, unique_count, top_val, top_count]

        display(summary_df)

def detect_outliers_iqr(df: pd.DataFrame, plot=True) -> None:
    """
    Detect outliers in a DataFrame using the IQR method.
    Does NOT modify the original DataFrame (no new columns created).

    If plot=True → creates a bar chart of number of outliers per column.
    """

    print("Detecting outliers using IQR (1.5 × IQR rule)")
    print("=" * 80)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        print("No numeric columns found.")
        return pd.DataFrame()

    outlier_info = []

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = is_outlier.sum()
        outlier_pct = (outlier_count / len(df) * 100) if len(df) > 0 else 0

        outlier_info.append({
            "Column": col,
            "Outlier_Count": outlier_count,
            "Outlier_Percent": round(outlier_pct, 2),
            "Lower_Bound": round(lower_bound, 2),
            "Upper_Bound": round(upper_bound, 2)
        })

    outlier_df = pd.DataFrame(outlier_info)

    print("\nOutlier Summary:")
    display(outlier_df)
    print("\n" + "=" * 80)
    return None

# ------------------------------------------------------------------ 

def bin_distance(distance: pd.Series, bin_size: int = 4, max_range: int = 40) -> pd.Series:
    """Bin shot distance for smoother trends."""

    bins = list(range(0, max_range + bin_size, bin_size))
    labels = [f"{bins[i]}–{bins[i + 1]} ft" for i in range(len(bins) - 1)]
    return pd.cut(distance.clip(upper=max_range), bins=bins, labels=labels, include_lowest=True)



def plot_shot_distance_hist(df: pd.DataFrame) -> None:
    """Plot histogram of shot distances."""
    dist_col = next((c for c in ["SHOT_DISTANCE", "DISTANCE", "LOC_X"] if c in df.columns), None)
    if not dist_col:
        print("No distance column found – skipping histogram.")
        return

    plt.figure()
    sns.histplot(df[dist_col].dropna(), bins=40, kde=True, color="#1f77b4")
    plt.title("Distribution of Shot Distance")
    plt.xlabel(dist_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_shot_distance_fg(df: pd.DataFrame, ax=None) -> None:
    """Plot FG% by distance bins. Can draw on an external ax."""
    dist_col = next((c for c in ["SHOT_DISTANCE", "DISTANCE", "LOC_X"] if c in df.columns), None)
    outcome_col = "SHOT_MADE_FLAG" if "SHOT_MADE_FLAG" in df.columns else None

    if not dist_col or not outcome_col:
        print("Distance or outcome column missing – skipping FG% plot.")
        return

    # Binning der Distanz
    binned = pd.cut(df[dist_col].fillna(0), bins=range(0, 40, 2), right=False)
    efficiency = (
        df.assign(distance_bin=binned)
          .groupby("distance_bin")[outcome_col]
          .mean()
          .dropna()
          .reset_index()
    )
    efficiency["distance_bin"] = efficiency["distance_bin"].astype(str)

    if ax is None:
        plt.figure()
        sns.lineplot(data=efficiency, x="distance_bin", y=outcome_col, marker="o")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.title("FG% by Distance Bin")
        plt.xlabel("Distance Bin")
        plt.ylabel("FG%")
        plt.tight_layout()
        plt.show()
        plt.close()
    else:
        sns.lineplot(data=efficiency, x="distance_bin", y=outcome_col,
                     marker="o", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("FG% by Distance Bin")
        ax.set_xlabel("Distance Bin")
        ax.set_ylabel("FG%")

def plot_zone_efficiency(df: pd.DataFrame, ax=None) -> None:
    """Analyze efficiency by shot zone."""

    zone_col = "SHOT_ZONE_BASIC" 
    outcome_col = "SHOT_MADE_FLAG"

    if not zone_col or not outcome_col:
        print("Shot zone analysis skipped (zone or outcome column missing).")
        return

    zone_stats = (
        df.groupby(zone_col)[outcome_col]
        .agg(["count", "mean"])
        .rename(columns={"count": "attempts", "mean": "fg_pct"})
        .sort_values("fg_pct", ascending=False)
    )

    # Wenn kein ax übergeben wurde, aktuelle Achse verwenden
    if ax is None:
        ax = plt.gca()

    # Barplot auf ax zeichnen
    zone_stats["fg_pct"].plot(kind="bar", color="#2ca02c", ax=ax)
    ax.set_title("FG% by Shot Zone")
    ax.set_ylabel("FG%")
    ax.set_xlabel(zone_col)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")


def plot_period_effect(shot_data: pd.DataFrame, ax=None) -> None:
    """
    Visualize FG% by PERIOD.

    Requirements:
    - Columns: PERIOD, SHOT_DISTANCE, EVENT_TYPE
    - EVENT_TYPE values: 'Made Shot' or 'Missed Shot'
    """

    # Prüfen, ob die benötigten Spalten vorhanden sind
    required_cols = {'PERIOD', 'SHOT_DISTANCE', 'EVENT_TYPE'}
    if not required_cols.issubset(shot_data.columns):
        print("PERIOD oder EVENT_TYPE fehlen – Perioden-Auswertung übersprungen.")
        return

    # Filter für gültige Würfe
    df = shot_data.dropna(subset=['SHOT_DISTANCE'])
    df = df[df['EVENT_TYPE'].isin(['Made Shot', 'Missed Shot'])]

    # FG% pro Period berechnen
    period_fg = (
        df.groupby('PERIOD')
          .apply(lambda g: (g['EVENT_TYPE'] == 'Made Shot').mean())
          .reset_index(name='FG_PCT')
    )

    # Wenn kein ax übergeben wurde, aktuelle Achse verwenden
    if ax is None:
        ax = plt.gca()

    # Plot auf ax zeichnen
    sns.barplot(data=period_fg, x='PERIOD', y='FG_PCT', color='slateblue', ax=ax)
    ax.set_title('FG% by Quarter (PERIOD)')
    ax.set_xlabel('PERIOD')
    ax.set_ylabel('FG%')



def plot_time_pressure(df: pd.DataFrame, ax=None) -> None:
    """Analyze FG% under time pressure using fixed column names."""

    # Prüfen, ob alle nötigen Spalten existieren
    required_cols = ["MINUTES_REMAINING", "SECONDS_REMAINING", "PERIOD", "SHOT_MADE_FLAG"]
    if not all(col in df.columns for col in required_cols):
        print("Time-pressure analysis skipped (missing time or outcome columns).")
        return

    # Gesamtsekunden berechnen
    total_seconds = df["MINUTES_REMAINING"].fillna(0) * 60 + df["SECONDS_REMAINING"].fillna(0)
    df = df.assign(
        seconds_remaining_total=total_seconds,
        period_for_plot=df["PERIOD"]
    )

    # Zeit-Bins für Shot Clock
    df["time_bin"] = pd.cut(
        df["seconds_remaining_total"],
        bins=[0, 5, 10, 15, 24, 120, 300, 720],
        labels=["0-5s", "5-10s", "10-15s", "15-24s", "24s-2m", "2-5m", "5-12m"],
        include_lowest=True
    )

    # Berechne FG% pro Period x Time Bin
    time_eff = (
        df.groupby(["period_for_plot", "time_bin"])["SHOT_MADE_FLAG"]
        .mean()
        .reset_index()
    )

    # Period für hue in Plot als String konvertieren
    time_eff["period_for_plot"] = time_eff["period_for_plot"].astype(str)

    # Wenn kein ax übergeben wurde, aktuelle Achse verwenden
    if ax is None:
        ax = plt.gca()

    # Plot auf ax zeichnen
    sns.lineplot(
        data=time_eff,
        x="time_bin",
        y="SHOT_MADE_FLAG",
        hue="period_for_plot",
        marker="o",
        ax=ax
    )
    ax.set_title("FG% by Remaining Shot Clock and Period")
    ax.set_xlabel("Shot Clock Bin")
    ax.set_ylabel("FG%")
    ax.set_ylim(0, 1)




def add_nba_court(fig):
    """
    Draws an NBA halfcourt using Plotly shapes.
    Based on standard NBA court dimensions.
    """
    
    # --- Court Outline (Baseline, Sidelines) ---
    fig.add_shape(type="line", x0=-250, y0=-47.5, x1=250, y1=-47.5, line=dict(color="white", width=2))  # Baseline
    fig.add_shape(type="line", x0=-250, y0=-47.5, x1=-250, y1=422.5, line=dict(color="white", width=2))  # Left sideline
    fig.add_shape(type="line", x0=250, y0=-47.5, x1=250, y1=422.5, line=dict(color="white", width=2))    # Right sideline
    fig.add_shape(type="line", x0=-250, y0=422.5, x1=250, y1=422.5, line=dict(color="white", width=2))   # Halfcourt line
    
    # --- Backboard ---
    fig.add_shape(type="line", x0=-30, y0=-10, x1=30, y1=-10, line=dict(color="white", width=2))
    
    # --- Paint / Lane (free throw area) ---
    fig.add_shape(type="line", x0=-80, y0=-47.5, x1=-80, y1=142.5, line=dict(color="white", width=2))
    fig.add_shape(type="line", x0=80, y0=-47.5, x1=80, y1=142.5, line=dict(color="white", width=2))
    fig.add_shape(type="line", x0=-60, y0=-47.5, x1=-60, y1=142.5, line=dict(color="white", width=2))
    fig.add_shape(type="line", x0=60, y0=-47.5, x1=60, y1=142.5, line=dict(color="white", width=2))
    fig.add_shape(type="line", x0=-80, y0=142.5, x1=80, y1=142.5, line=dict(color="white", width=2))     # Free throw line
    
    # --- Hoop (rim) ---
    theta = np.linspace(0, 2*np.pi, 100)
    hoop_radius = 7.5
    hoop_x = hoop_radius * np.cos(theta)
    hoop_y = hoop_radius * np.sin(theta)
    fig.add_trace(go.Scatter(x=hoop_x, y=hoop_y, mode="lines", line=dict(color="orange", width=2), 
                             hoverinfo="skip", showlegend=False))
    
    # --- Restricted Area (semicircle) ---
    theta = np.linspace(0, np.pi, 100)
    restricted_radius = 40
    restricted_x = restricted_radius * np.cos(theta)
    restricted_y = restricted_radius * np.sin(theta)
    fig.add_trace(go.Scatter(x=restricted_x, y=restricted_y, mode="lines", line=dict(color="white", width=2),
                             hoverinfo="skip", showlegend=False))
    
    # --- Free Throw Circle (top) ---
    theta = np.linspace(0, np.pi, 150)
    ft_radius = 60
    ft_x = ft_radius * np.cos(theta)
    ft_y = 142.5 + ft_radius * np.sin(theta)
    fig.add_trace(go.Scatter(x=ft_x, y=ft_y, mode="lines", line=dict(color="white", width=2),
                             hoverinfo="skip", showlegend=False))
    
    # --- Free Throw Circle (bottom, dashed) ---
    theta = np.linspace(np.pi, 2*np.pi, 150)
    ft_x_dashed = ft_radius * np.cos(theta)
    ft_y_dashed = 142.5 + ft_radius * np.sin(theta)
    fig.add_trace(go.Scatter(x=ft_x_dashed, y=ft_y_dashed, mode="lines", 
                             line=dict(color="white", width=2, dash="dash"),
                             hoverinfo="skip", showlegend=False))
    
    # --- 3-Point Lines (Corner 3s) ---
    fig.add_shape(type="line", x0=-220, y0=-47.5, x1=-220, y1=92.5, line=dict(color="white", width=2))
    fig.add_shape(type="line", x0=220, y0=-47.5, x1=220, y1=92.5, line=dict(color="white", width=2))
    
    # --- 3-Point Arc ---
    # Arc with radius 237.5, from theta=22° to theta=158° (in degrees, converts to radians)
    theta_start = np.radians(22)
    theta_end = np.radians(158)
    theta = np.linspace(theta_start, theta_end, 200)
    arc_radius = 237.5
    arc_x = arc_radius * np.cos(theta)
    arc_y = arc_radius * np.sin(theta)
    fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode="lines", line=dict(color="white", width=2),
                             hoverinfo="skip", showlegend=False))
    
    # --- Halfcourt Circle (top) ---
    theta = np.linspace(np.pi, 2*np.pi, 150)
    hc_radius = 61
    hc_x = hc_radius * np.cos(theta)
    hc_y = 422.5 + hc_radius * np.sin(theta)
    fig.add_trace(go.Scatter(x=hc_x, y=hc_y, mode="lines", line=dict(color="white", width=2),
                             hoverinfo="skip", showlegend=False))

def plot_hex_shot_chart(
    df: pd.DataFrame,
    player_name: str,
    seasons: list,
    bin_size: int = 8,
    figsize: tuple = (12, 11)
) -> go.Figure:
    """
    Interactive Hex Shot Chart with Plotly.
    
    - X/Y coordinates aggregated into hexagon bins
    - Color = FG% (field goal percentage)
    - Size = number of attempts
    - Court details are added
    """
    
    # Filter by player and season(s)
    player_data = df[
        (df['PLAYER_NAME'] == player_name) &
        (df['SEASON'].isin(seasons))
    ].copy()
    
    if len(player_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data for {player_name} in {', '.join(seasons)}",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Create bins
    player_data['x_bin'] = (player_data['LOC_X'] // bin_size * bin_size).astype(int)
    player_data['y_bin'] = (player_data['LOC_Y'] // bin_size * bin_size).astype(int)
    
    # Aggregate per bin including mean SHOT_DISTANCE
    hex_stats = (
        player_data.groupby(['x_bin', 'y_bin'])
        .agg({
            'SHOT_MADE_FLAG': ['sum', 'count', 'mean'],
            'SHOT_DISTANCE': 'mean'
        })
        .reset_index()
    )
    hex_stats.columns = ['x_bin', 'y_bin', 'makes', 'attempts', 'fg_pct', 'distance']
    hex_stats['fg_pct'] = hex_stats['fg_pct'].round(3)
    hex_stats['distance'] = hex_stats['distance'].round(1)
    
    # Overall stats for title
    total_attempts = len(player_data)
    overall_fg_pct = player_data['SHOT_MADE_FLAG'].mean()
    
    # Stats for different zones
    at_rim = player_data[player_data['SHOT_DISTANCE'] <= 3]
    mid_range = player_data[(player_data['SHOT_DISTANCE'] > 3) & (player_data['SHOT_DISTANCE'] <= 16)]
    three_pt = player_data[player_data['SHOT_DISTANCE'] > 22]
    
    at_rim_pct = at_rim['SHOT_MADE_FLAG'].mean() if len(at_rim) > 0 else 0
    mid_pct = mid_range['SHOT_MADE_FLAG'].mean() if len(mid_range) > 0 else 0
    three_pct = three_pt['SHOT_MADE_FLAG'].mean() if len(three_pt) > 0 else 0
    
    # Visualization
    fig = go.Figure()
    
    # Scatter heatmap
    fig.add_trace(go.Scatter(
        x=hex_stats['x_bin'],
        y=hex_stats['y_bin'],
        mode='markers',
        marker=dict(
            size=np.sqrt(hex_stats['attempts']) * 1.5,  # Improved size scaling
            color=hex_stats['fg_pct'],
            colorscale='RdYlGn',
            cmin=0, cmax=1,
            showscale=True,
            colorbar=dict(title="FG%", thickness=15, len=0.7, x=1.02),
            line=dict(width=1, color='rgba(255,255,255,0.8)'),
            opacity=0.85,
        ),
        text=[
            f"<b>FG%: {fg:.1%}</b><br>Attempts: {att}<br>Makes: {m}<br>Distance: {d:.1f} ft"
            for fg, att, m, d in zip(
                hex_stats['fg_pct'], hex_stats['attempts'], hex_stats['makes'], hex_stats['distance']
            )
        ],
        hoverinfo='text',
        hovertemplate='%{text}<extra></extra>',
        name=''
    ))
        
    # Add court
    add_nba_court(fig)

    seasons_str = f"{', '.join(seasons)}" if len(seasons) <= 3 else f"{len(seasons)} seasons"

    subtitle = (
        f"{seasons_str} | {total_attempts} attempts | {overall_fg_pct:.1%} FG<br>"
        f"<sub>At Rim: {at_rim_pct:.1%} | Mid-Range: {mid_pct:.1%} | 3PT: {three_pct:.1%}</sub>"
    )

    fig.update_layout(
        title=f"<b>{player_name}</b> – Shot Chart (FG% Heatmap)<br>{subtitle}",
        xaxis_title="Court X (ft)",
        yaxis_title="Court Y (ft)",
        hovermode='closest',
        plot_bgcolor="#2B2B2B",
        paper_bgcolor="#2B2B2B",
        font=dict(color='white', size=11),
        xaxis=dict(
            scaleanchor="y", 
            scaleratio=1, 
            showgrid=False, 
            zeroline=False,
            range=[-280, 280]
        ),
        yaxis=dict(
            scaleanchor="x", 
            scaleratio=1, 
            showgrid=False, 
            zeroline=False,
            range=[-100, 470]  # Optimized range
        ),
        width=1050,
        height=900,
        margin=dict(l=60, r=100, t=120, b=60),
    )

    return fig

def plot_shot_count_per_season(df, season_col='SEASON', palette='Set2', figsize=(10,5), show_counts=True):
    """
    Plots the number of shots per season as a bar plot.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the shot data.
    season_col : str
        Column name for the season.
    palette : str
        Seaborn color palette.
    figsize : tuple
        Figure size.
    show_counts : bool
        Whether to show the count numbers on top of the bars.
    """
    # Sicherstellen, dass die Spalte existiert
    if season_col not in df.columns:
        print(f"Column '{season_col}' not found in DataFrame.")
        return
    
    # Shot Counts pro Saison
    season_counts = df[season_col].value_counts().sort_index()
    
    # Plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=season_counts.index, y=season_counts.values, palette=palette)
    
    plt.title('Shot Count per Season')
    plt.xlabel('Season')
    plt.ylabel('Number of Shots')
    plt.xticks(rotation=45)
    
    # Optional Zahlen auf die Balken
    if show_counts:
        for p in ax.patches:
            height = p.get_height()
            ax.text(
                p.get_x() + p.get_width()/2.,
                height + height*0.01,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9
            )
    
    plt.tight_layout()
    plt.show()

def plot_shot_type_distribution(df, shot_col='SHOT_TYPE', palette='Set2', figsize=(7,5), show_percent=True):
    """
    Plots the distribution of shot types as a bar plot with percentages on the bars.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the shot data.
    shot_col : str
        Column name for shot type (e.g., 'SHOT_TYPE').
    palette : str
        Seaborn color palette.
    figsize : tuple
        Figure size.
    show_percent : bool
        Whether to show percentage values inside the bars.
    """
    if shot_col not in df.columns:
        print(f"Column '{shot_col}' not found in DataFrame.")
        return

    plt.figure(figsize=figsize)
    ax = sns.countplot(x=shot_col, data=df, palette=palette)
    
    plt.title('Shot Type Distribution')
    plt.ylabel('Number of Shots')
    plt.xlabel(shot_col)
    plt.xticks(rotation=45)

    if show_percent:
        total = len(df)
        for p in ax.patches:
            height = p.get_height()
            percent = height / total * 100
            ax.text(
                p.get_x() + p.get_width()/2.,
                height * 0.5,           # mittig im Balken
                f'{percent:.1f}%', 
                ha='center', va='center', color='black', fontsize=10
            )
    
    plt.tight_layout()
    plt.show()

def plot_top_action_types(df, action_col='ACTION_TYPE', top_n=10, palette='Set2', figsize=(12,6)):
    """
    Plots the top N action types as a count plot.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the shot data.
    action_col : str
        Column name for the action type (e.g., 'ACTION_TYPE').
    top_n : int
        Number of top action types to display.
    palette : str
        Seaborn color palette.
    figsize : tuple
        Figure size.
    """
    if action_col not in df.columns:
        print(f"Column '{action_col}' not found in DataFrame.")
        return
    
    # Top N Action Types
    top_actions = df[action_col].value_counts().nlargest(top_n).index

    plt.figure(figsize=figsize)
    sns.countplot(x=action_col, data=df, palette=palette, order=top_actions)
    
    plt.title(f'Top {top_n} Action Types')
    plt.xlabel(action_col)
    plt.ylabel('Number of Shots')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_made_shots_per_team(df, team_col='TEAM_NAME', made_flag_col='SHOT_MADE_FLAG',
                             palette='Set3', figsize=(14,6)):
    """
    Plots the number of made shots per team with team names on the bars.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing shot data.
    team_col : str
        Column name for the team.
    made_flag_col : str
        Column name for the shot made flag (1 = made, 0 = missed).
    palette : str
        Seaborn color palette.
    figsize : tuple
        Figure size.
    """
    if team_col not in df.columns or made_flag_col not in df.columns:
        print(f"Columns '{team_col}' or '{made_flag_col}' not found in DataFrame.")
        return
    
    # Nur getroffene Würfe
    made_shots = df[df[made_flag_col] == 1]
    
    # Anzahl pro Team
    team_made = made_shots[team_col].value_counts()  # alle Teams
    
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=team_made.index, y=team_made.values, palette=palette)
    
    plt.title('Made Shots per Team')
    plt.ylabel('Number of Made Shots')
    
    # Teamnamen senkrecht auf die Balken
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height/2, 
                team_made.index[i], 
                ha='center', va='center', rotation=90, fontsize=8, color='black')
    
    # x-Achse ausblenden
    ax.set_xticks([])
    
    plt.tight_layout()
    plt.show()


def plot_shot_distance_by_type(df, shot_type_col='SHOT_TYPE', distance_col='SHOT_DISTANCE',
                               palette='Set2', figsize=(10,5)):
    """
    Plots the distribution of shot distances by shot type as a boxplot.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the shot data.
    shot_type_col : str
        Column name for the shot type (e.g., 'SHOT_TYPE').
    distance_col : str
        Column name for shot distance (e.g., 'SHOT_DISTANCE').
    palette : str
        Seaborn color palette.
    figsize : tuple
        Figure size.
    """
    if shot_type_col not in df.columns or distance_col not in df.columns:
        print(f"Columns '{shot_type_col}' or '{distance_col}' not found in DataFrame.")
        return
    
    plt.figure(figsize=figsize)
    sns.boxplot(x=shot_type_col, y=distance_col, data=df, palette=palette)
    
    plt.title('Shot Distance by Shot Type')
    plt.ylabel('Distance (Feet)')
    plt.xlabel('Shot Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_season_trends(df, season_col='SEASON', distance_col='SHOT_DISTANCE', shot_type_col='SHOT_TYPE',
                       figsize=(12,5)):
    """
    Plots average shot distance and 3P share across seasons.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing shot data.
    season_col : str
        Column name for season.
    distance_col : str
        Column name for shot distance.
    shot_type_col : str
        Column name for shot type (2PT/3PT).
    figsize : tuple
        Figure size.
    """
    needed_cols = {season_col, distance_col, shot_type_col}
    if not needed_cols.issubset(df.columns):
        print("Required columns missing – season trend skipped.")
        return
    
    # Group by season
    season_grp = df.dropna(subset=[distance_col]).groupby(season_col)
    season_df = season_grp.agg(
        avg_distance=(distance_col, 'mean'),
        shots=(distance_col, 'size'),
        threes=(shot_type_col, lambda x: (x=='3PT Field Goal').sum())
    ).reset_index()
    
    season_df['three_share'] = season_df['threes'] / season_df['shots']
    
    # Plot
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    
    ax1.plot(season_df[season_col], season_df['avg_distance'], marker='o', color='tab:blue', label='Average Distance')
    ax2.plot(season_df[season_col], season_df['three_share'], marker='s', color='tab:red', label='3P Share Across Seasons')
    
    ax1.set_ylabel('Average Distance (Feet)')
    ax2.set_ylabel('3P Share')
    ax1.set_xlabel('Season')
    ax1.set_title('Trend: Distance & 3P Share Across Seasons')
    
    # Gemeinsame Legende
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def calculate_ppp(df: pd.DataFrame, team: str, season: str, top_n: int = 10) -> pd.DataFrame:
    """
    Calculate Points Per Possession (PPP) per zone per top-N players of a team.
    
    PPP = (points scored) / attempts
    - 2pt: 2 points
    - 3pt: 3 points
    """
    
    team_data = df[
        (df['TEAM_NAME'] == team) &
        (df['SEASON'] == season)
    ].copy()
    
    if len(team_data) == 0:
        return pd.DataFrame()
    
    # Points value per shot
    team_data['points_value'] = team_data['SHOT_TYPE'].apply(
        lambda x: 3 if '3' in str(x) else 2
    )
    
    # PPP per player and zone
    ppp_data = []
    for player in team_data['PLAYER_NAME'].unique():
        for zone in team_data['SHOT_ZONE_BASIC'].unique():
            zone_data = team_data[
                (team_data['PLAYER_NAME'] == player) &
                (team_data['SHOT_ZONE_BASIC'] == zone)
            ]
            
            if len(zone_data) > 0:
                attempts = zone_data['SHOT_ATTEMPTED_FLAG'].sum()
                # Punkte: nur für gemachte Würfe zählen
                points = (zone_data['SHOT_MADE_FLAG'] * zone_data['points_value']).sum()
                ppp = points / attempts if attempts > 0 else 0
                
                ppp_data.append({
                    'PLAYER_NAME': player,
                    'SHOT_ZONE_BASIC': zone,
                    'attempts': attempts,
                    'makes': zone_data['SHOT_MADE_FLAG'].sum(),
                    'ppp': ppp
                })
    
    player_zone_stats = pd.DataFrame(ppp_data)
    
    # Only top-N players (by total attempts)
    top_players = (
        player_zone_stats.groupby('PLAYER_NAME')['attempts']
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )
    
    ppp_matrix = (
        player_zone_stats[player_zone_stats['PLAYER_NAME'].isin(top_players)]
        .pivot_table(
            index='PLAYER_NAME',
            columns='SHOT_ZONE_BASIC',
            values='ppp',
            fill_value=0
        )
    )
    
    return ppp_matrix

def plot_ppp_heatmap(
    ppp_matrix: pd.DataFrame,
    team: str,
    season: str
) -> go.Figure:
    """Plotly heatmap with PPP values and improved contrast."""
    
    if ppp_matrix.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data for {team} in {season}",
            showarrow=False,
            font=dict(size=16, color='white')
        )
        return fig
    
    fig = go.Figure(data=go.Heatmap(
        z=ppp_matrix.values,
        x=ppp_matrix.columns,
        y=ppp_matrix.index,
        colorscale='Viridis',
        text=np.round(ppp_matrix.values, 2),
        texttemplate='%{text:.2f}',
        textfont={"size": 12, "color": "white"},
        hovertemplate='<b>%{y}</b><br>Zone: %{x}<br>PPP: %{z:.2f}<extra></extra>',
        colorbar=dict(title="PPP", thickness=20, len=0.7),
    ))
    
    fig.update_layout(
        title=f"<b>{team}</b> – Points Per Possession (PPP) by Zone<br><sub>{season} | Top 10 players</sub>",
        xaxis_title="Shot Zone",
        yaxis_title="Player",
        height=550,
        width=1050,
        hovermode='closest',
        plot_bgcolor="#2B2B2B",
        paper_bgcolor="#2B2B2B",
        font=dict(color='white', size=11),
        xaxis=dict(showgrid=False, side='bottom'),
        yaxis=dict(showgrid=False),
        margin=dict(l=150, r=80, t=100, b=80),
    )
    
    return fig
