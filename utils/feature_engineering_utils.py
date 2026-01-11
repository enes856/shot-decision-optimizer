import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nba_api.stats.static import teams
from eda_utils import add_nba_court

# ==========================================
# SHOT CLOCK CALCULATION FROM PBP
# ==========================================

def calculate_shot_clock_from_pbp(input_file: str, output_file: str = None) -> pd.DataFrame:
    """
    Calculates the 'shotclock_remaining' for each shot event by detecting
    possession changes within the Play-by-Play data.

    This function implements a possession-based approximation: Shot Clock = 24s - Elapsed Time.

    Args:
        input_file (str): Path to the raw Play-by-Play (PBP) data file.
        output_file (str, optional): Path to save the resulting DataFrame. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing all PBP events with the calculated 
                      'shotclock_remaining' column added to relevant rows.
    """
    # Check if input exists
    if not os.path.exists(input_file):
        print(f"Error: File not found at: {input_file}")
        return None

    print(f"Loading data from: {input_file} ...")
    df = pd.read_parquet(input_file)

    # 1. Ensure chronological sorting
    df = df.sort_values(["gameId", "period", "actionNumber"]).reset_index(drop=True)

    print("Converting game clock (Vectorized)...")

    # 2. Clock Parsing: Convert 'PTxxMxxS' string to total seconds remaining in quarter
    time_pattern = r'PT(\d+)M(\d+\.?\d*)S'
    extracted_times = df['clock'].astype(str).str.extract(time_pattern)

    minutes = pd.to_numeric(extracted_times[0], errors='coerce').fillna(0)
    seconds = pd.to_numeric(extracted_times[1], errors='coerce').fillna(0)

    df['quarter_remaining'] = minutes * 60 + seconds

    print("Calculating shot clock (Vectorized)...")

    # --- Possession Change Logic ---

    # Filter for active events (teamId != 0 and quarter_remaining is not NaN)
    cols_needed = ['gameId', 'period', 'teamId', 'quarter_remaining', 'actionType']
    df_calc = df[cols_needed].copy()
    mask_valid = (df_calc['teamId'] != 0) & (df_calc['quarter_remaining'].notna())
    df_active = df_calc[mask_valid].copy()

    # 4. Detect possession changes
    df_active['prev_team'] = df_active.groupby(['gameId', 'period'])['teamId'].shift(1)
    df_active['prev_time'] = df_active.groupby(['gameId', 'period'])['quarter_remaining'].shift(1)

    # A new possession starts if the team ID changes
    df_active['is_new_possession'] = df_active['teamId'] != df_active['prev_team']
    df_active['possession_id'] = df_active.groupby(['gameId', 'period'])['is_new_possession'].cumsum()

    # 5. Determine start time of possession (time of the event before the team switch)
    possession_starts = df_active.groupby(['gameId', 'period', 'possession_id'])['prev_time'].transform('first')
    # Fill quarter start time (720.0 seconds) for the first possession
    possession_starts = possession_starts.fillna(720.0)

    df_active['possession_start_time'] = possession_starts

    # 6. Calculate elapsed time & shot clock
    elapsed = df_active['possession_start_time'] - df_active['quarter_remaining']

    shotclock_calc = 24.0 - elapsed
    # Clip values to ensure they stay within the valid range [0.0, 24.0]
    shotclock_calc = shotclock_calc.clip(lower=0.0, upper=24.0) 

    # Round to 1 decimal to keep sub-second precision without excessive float noise.

    # Handle edge case: If calculated shot clock exceeds quarter remaining time
    mask_end_of_quarter = shotclock_calc > df_active['quarter_remaining']
    # For these cases, set shot clock to the quarter remaining time
    shotclock_calc.loc[mask_end_of_quarter] = df_active.loc[mask_end_of_quarter, 'quarter_remaining']

    shotclock_calc = shotclock_calc.round(1)
    
    df_active['shotclock_remaining'] = shotclock_calc

    # 7. Merge data back into the main DataFrame (only for Made/Missed Shots)
    shot_mask = df_active['actionType'].isin(['Made Shot', 'Missed Shot'])
    final_values = df_active.loc[shot_mask, 'shotclock_remaining']

    df['shotclock_remaining'] = pd.NA
    df.loc[final_values.index, 'shotclock_remaining'] = final_values

    print("Calculation completed.")

    # Save output
    if output_file:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 Directory created: {output_dir}")

        print(f"Saving as {output_file}...")
        df.to_parquet(output_file)

    return df


# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================

def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """
    Loads the Parquet file and validates its existence.

    Args:
        file_path (str): Relative or absolute path to the parquet file.

    Returns:
        pd.DataFrame: The loaded pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' was not found. Check your path.")

    print(f"Loading data from: {file_path} ...")
    df = pd.read_parquet(file_path)
    print(f"Data loaded successfully: {len(df)} rows.")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic cleaning, renaming, column validation, and CRITICAL sorting.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Cleaned and sorted dataframe ready for time-series calculations.
    """
    df = df.copy()


    # Date Conversion
    if df['GAME_DATE'].dtype == 'object':
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    # Sorting is essential for shift() operations later
    df = df.sort_values(by=['GAME_DATE', 'GAME_ID', 'GAME_EVENT_ID']).reset_index(drop=True)

    return df

# ==========================================
# STATIC FEATURES
# ==========================================

def create_custom_shot_zone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a granular shot zone by combining Basic Zone and Area.
    Example: 'Mid-Range | Left Side(L)'
    """
    df['CUSTOM_SHOT_ZONE'] = df['SHOT_ZONE_BASIC'].astype(str) + " | " + df['SHOT_ZONE_AREA'].astype(str)
    return df

def calculate_shot_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines if a shot is worth 2 or 3 points based on the SHOT_TYPE text.
    """
    df['SHOT_VALUE'] = df['SHOT_TYPE'].astype(str).apply(lambda x: 3 if '3PT' in x.upper() else 2)
    return df

def calculate_shot_angle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the absolute angle of the shot relative to the basket (0-180 degrees).
    Uses LOC_X and LOC_Y coordinates.
    """
    # arctan2 returns radians, convert to degrees.
    df['SHOT_ANGLE'] = np.abs(np.arctan2(df['LOC_X'], df['LOC_Y']) * (180 / np.pi))
    return df

def calculate_time_remaining(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts minutes and seconds remaining into a single total seconds column.
    """
    df['TIME_REMAINING'] = (df['MINUTES_REMAINING'] * 60) + df['SECONDS_REMAINING']
    return df

# ==========================================
# HISTORICAL FEATURES (Time-Series)
# ==========================================

def calculate_player_season_fg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the player's Field Goal Percentage for the current season 
    UP TO the current shot.

    Logic: Uses expanding window with shift(1) to avoid data leakage.
    """
    # Group by Season AND Player (resets every season)
    season_grouper = df.groupby(['SEASON', 'PLAYER_ID'])

    # Calculate previous makes and attempts
    season_makes = season_grouper['SHOT_MADE_FLAG'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    season_attempts = season_grouper.cumcount() # cumcount starts at 0, effectively shifted

    # Calculate Percentage safely
    df['PLAYER_SEASON_FG_PCT'] = np.where(
        season_attempts > 0,
        season_makes / season_attempts,
        np.nan
    )
    return df

def calculate_player_zone_fg(df: pd.DataFrame, window: int = 50, min_periods: int = 3) -> pd.DataFrame:
    """
    Calculates the player's FG% specifically for the current CUSTOM_SHOT_ZONE.

    Args:
        window (int): Number of previous shots to consider.
        min_periods (int): Minimum shots required to calculate a value.
    """
    zone_grouper = df.groupby(['PLAYER_ID', 'CUSTOM_SHOT_ZONE'])['SHOT_MADE_FLAG']

    df['PLAYER_ZONE_FG_PCT'] = zone_grouper.transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=min_periods).mean()
    )
    return df

def calculate_hot_hand(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Calculates the number of shots made in the last N attempts (Hot Hand).
    """
    hot_hand_grouper = df.groupby(['SEASON', 'PLAYER_ID'])['SHOT_MADE_FLAG']

    df['LAST_5_SHOTS_MADE'] = hot_hand_grouper.transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).sum()
    ).fillna(0)
    return df

def calculate_long_term_form(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """
    Calculates the FG% over the last N shots (Long-term form), spanning across seasons.
    """
    player_grouper = df.groupby('PLAYER_ID')['SHOT_MADE_FLAG']
    
    df['PLAYER_LAST_100_FG_PCT'] = player_grouper.transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=20).mean()
    )
    return df

# ==========================================
# CONTEXT / OPPONENT FEATURES
# ==========================================

def add_team_abbreviations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps TEAM_ID to TEAM_ABBREVIATION using the static NBA API data.
    """
    nba_teams = teams.get_teams()
    id_to_abbrev_map = {team['id']: team['abbreviation'] for team in nba_teams}
    
    df['TEAM_ABBREVIATION'] = df['TEAM_ID'].map(id_to_abbrev_map)
    return df

def identify_opponent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines the Opponent Team Abbreviation based on Home/Visitor columns.
    """
    df['OPPONENT_TEAM'] = np.where(
        df['TEAM_ABBREVIATION'] == df['HTM'], 
        df['VTM'], 
        df['HTM']
    )
    return df

def calculate_opponent_def_strength(df: pd.DataFrame, window: int = 500) -> pd.DataFrame:
    """
    Calculates how many shots (FG%) the opponent allows on average.
    Higher value = Weaker Defense.
    """
    opp_grouper = df.groupby(['SEASON', 'OPPONENT_TEAM'])['SHOT_MADE_FLAG']
    
    df['OPP_DEF_STRENGTH'] = opp_grouper.transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=50).mean()
    )
    return df

# ==========================================
# PLAYER INFO MERGE
# ==========================================

def add_player_height(df: pd.DataFrame, player_info_path: str = '../data/raw/nba_raw_player_data.csv') -> pd.DataFrame:
    """
    Merges only the player height into the main dataframe.
    Converts Height from 'ft-in' string to centimeters (Integer).

    Args:
        df (pd.DataFrame): Main dataframe containing 'PLAYER_ID'.
        player_info_path (str): Path to the player info CSV file.

    Returns:
        pd.DataFrame: DataFrame with 'PLAYER_HEIGHT_CM' added.
    """
    if not os.path.exists(player_info_path):
        print(f"Warning: Player info file not found at {player_info_path}. Skipping height merge.")
        return df

    print(f"Loading player height info from: {player_info_path} ...")

    try:
        # Load only necessary columns
        df_info = pd.read_csv(player_info_path, usecols=['PLAYER_ID', 'HEIGHT'])
    except ValueError as e:
        print(f"Error loading columns for height: {e}. Check CSV format.")
        return df

    # --- Height Conversion (Vectorized) ---
    # Split '6-6' into two columns
    height_split = df_info['HEIGHT'].astype(str).str.split('-', expand=True)

    # Convert to numeric, coerce errors to NaN
    feet = pd.to_numeric(height_split[0], errors='coerce')
    inches = pd.to_numeric(height_split[1], errors='coerce')

    # Calculate CM: ((Feet * 12) + Inches) * 2.54
    # Round to nearest whole number and convert to Nullable Integer (Int64)
    df_info['PLAYER_HEIGHT_CM'] = (((feet * 12) + inches) * 2.54).round(0).astype('Int64')

    # Drop the original string HEIGHT column
    df_info = df_info[['PLAYER_ID', 'PLAYER_HEIGHT_CM']]

    # Deduplicate based on PLAYER_ID
    if not df_info['PLAYER_ID'].is_unique:
        df_info = df_info.drop_duplicates(subset=['PLAYER_ID'])

    print("Merging Player Height...")
    
    # Ensure ID types match
    if df['PLAYER_ID'].dtype != df_info['PLAYER_ID'].dtype:
        try:
            df_info['PLAYER_ID'] = df_info['PLAYER_ID'].astype(df['PLAYER_ID'].dtype)
        except:
            pass

    # Merge
    df = df.merge(df_info, on='PLAYER_ID', how='left')
    
    # Validation
    match_rate = df['PLAYER_HEIGHT_CM'].notna().mean()
    print(f"Player Height added. Match Rate: {match_rate:.1%}")

    return df


def add_player_position(df: pd.DataFrame, player_info_path: str = '../data/raw/nba_raw_player_data.csv') -> pd.DataFrame:
    """
    Merges only the player position into the main dataframe.

    Args:
        df (pd.DataFrame): Main dataframe containing 'PLAYER_ID'.
        player_info_path (str): Path to the player info CSV file.

    Returns:
        pd.DataFrame: DataFrame with 'PLAYER_POSITION' added.
    """
    if not os.path.exists(player_info_path):
        print(f"Warning: Player info file not found at {player_info_path}. Skipping position merge.")
        return df

    print(f"Loading player position info from: {player_info_path} ...")

    try:
        # Load only necessary columns
        df_info = pd.read_csv(player_info_path, usecols=['PLAYER_ID', 'POSITION'])
    except ValueError as e:
        print(f"Error loading columns for position: {e}. Check CSV format.")
        return df

    # Rename to match desired output
    df_info = df_info.rename(columns={'POSITION': 'PLAYER_POSITION'})

    # Deduplicate based on PLAYER_ID
    if not df_info['PLAYER_ID'].is_unique:
        df_info = df_info.drop_duplicates(subset=['PLAYER_ID'])

    print("Merging Player Position...")
    
    # Ensure ID types match
    if df['PLAYER_ID'].dtype != df_info['PLAYER_ID'].dtype:
        try:
            df_info['PLAYER_ID'] = df_info['PLAYER_ID'].astype(df['PLAYER_ID'].dtype)
        except:
            pass

    # Merge
    df = df.merge(df_info, on='PLAYER_ID', how='left')
    
    # Validation
    match_rate = df['PLAYER_POSITION'].notna().mean()
    print(f"Player Position added. Match Rate: {match_rate:.1%}")

    return df

# ==========================================
# EXTERNAL MERGE
# ==========================================

def merge_shot_clock(df: pd.DataFrame, shotclock_path: str) -> pd.DataFrame:
    """
    Merges external shot clock data with the main dataframe.
    Handles ID standardization (padding with zeros) to ensure correct matching.

    Args:
        df (pd.DataFrame): Main shots dataframe.
        shotclock_path (str): Path to the engineered shotclock parquet file.

    Returns:
        pd.DataFrame: Merged dataframe with SHOTCLOCK column.
    """
    if not os.path.exists(shotclock_path):
        print(f"Warning: Shotclock file not found at {shotclock_path}. Column will be missing.")
        return df

    print("Merging Shot Clock Data...")
    df_clock = pd.read_parquet(shotclock_path, columns=['gameId', 'actionNumber', 'shotclock_remaining'])

    # Standardize IDs for merging (Crucial Step)
    # Force Strings and pad to 10 digits (e.g., "0021900001")
    df['GAME_ID_KEY'] = df['GAME_ID'].astype(str).str.zfill(10)
    df_clock['gameId_KEY'] = df_clock['gameId'].astype(str).str.zfill(10)

    # Ensure Events are Integers
    df['GAME_EVENT_ID'] = df['GAME_EVENT_ID'].astype(int)
    df_clock['actionNumber'] = df_clock['actionNumber'].astype(int)

    # Deduplicate shotclock data to prevent row explosion
    before_dedup = len(df_clock)
    df_clock = df_clock.drop_duplicates(subset=['gameId_KEY', 'actionNumber'], keep='first')
    after_dedup = len(df_clock)

    if before_dedup > after_dedup:
        print(f"Removed {before_dedup - after_dedup} duplicate rows from shotclock data to prevent row explosion.")

    # Perform Merge
    df = df.merge(
        df_clock,
        left_on=['GAME_ID_KEY', 'GAME_EVENT_ID'],
        right_on=['gameId_KEY', 'actionNumber'],
        how='left'
    )

    # Clean up and Rename
    df = df.rename(columns={'shotclock_remaining': 'SHOTCLOCK'})
    df = df.drop(columns=['gameId', 'actionNumber', 'GAME_ID_KEY', 'gameId_KEY'])

    # Handle NaNs (matches statistics)
    matches = df['SHOTCLOCK'].notna().sum()
    print(f"Shotclock Merge Match Rate: {matches / len(df):.1%}")

    return df


### ==========================================
# PLOTTING FUNCTION
### ==========================================

def plot_custom_shot_zones(df: pd.DataFrame, sample_size: int = 50000) -> go.Figure:
    """
    Creates an interactive scatter plot of ALL CUSTOM_SHOT_ZONEs using Plotly.

    Args:
        df (pd.DataFrame): The dataframe containing 'LOC_X', 'LOC_Y', and 'CUSTOM_SHOT_ZONE'.
        sample_size (int): Max number of shots to plot to maintain browser performance. 
                           Defaults to 50000.

    Returns:
        go.Figure: The interactive Plotly figure.
    """
    # 1. Sample Data if necessary (to prevent browser crash)
    if len(df) > sample_size:
        print(f"Sampling {sample_size} shots from {len(df)} total shots...")
        df_plot = df.sample(n=sample_size, random_state=42).copy()
    else:
        df_plot = df.copy()

    # 2. Get all unique zones present in the data (sorted alphabetically)
    unique_zones = sorted(df_plot['CUSTOM_SHOT_ZONE'].unique())

    print(f"Plotting all {len(unique_zones)} Zones...")

    # 3. Create Figure
    fig = go.Figure()

    # 4. Add Traces (One scatter plot per Zone)
    for zone in unique_zones:
        subset = df_plot[df_plot['CUSTOM_SHOT_ZONE'] == zone]

        fig.add_trace(go.Scatter(
            x=subset['LOC_X'],
            y=subset['LOC_Y'],
            mode='markers',
            name=zone,
            marker=dict(
                size=5,
                opacity=0.6
            ),
            text=subset['CUSTOM_SHOT_ZONE'], # Hover text
            hoverinfo='text'
        ))

    # 5. Add the Court Lines (assuming add_nba_court is in the same file)
    add_nba_court(fig)

    # 6. Layout Settings
    fig.update_layout(
        title=f'<b>Visualization of ALL Custom Shot Zones</b><br>(Sample: {len(df_plot)} shots)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-260, 260]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-50, 430]),
        plot_bgcolor='#2B2B2B',
        paper_bgcolor='#2B2B2B',
        font=dict(color='white'),
        width=1000,
        height=900,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            title="Shot Zones"
        )
    )

    return fig