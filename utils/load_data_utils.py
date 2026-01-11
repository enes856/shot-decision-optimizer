import os
import pandas as pd
import time
from nba_api.stats.endpoints import shotchartdetail

def create_comprehensive_nba_dataset(
    start_year=2019,
    end_year=2025,
    filename="nba_raw_shot_data.parquet",
    path="."
):
    """
    Loads ONLY raw NBA shot chart data for multiple seasons and stores it as a Parquet file.
    You can specify a folder path + filename.
    No extra feature engineering, no added columns, no renaming.

    Args:
        start_year (int): first season (e.g., 2019 for 2019-20)
        end_year (int): last season (exclusive, e.g., 2025)
        filename (str): output filename (e.g., "shots.parquet")
        path (str): directory where the file will be stored
    """

    CUSTOM_HEADERS = {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.nba.com/',
        'Connection': 'keep-alive',
    }

    # Build full file path
    full_path = os.path.join(path, filename)

    all_seasons_data = []
    seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(start_year, end_year)]

    for season in seasons:
        try:
            # Fetch raw shot chart data for the entire league
            shot_data = shotchartdetail.ShotChartDetail(
                player_id=0,
                team_id=0,
                season_nullable=season,
                context_measure_simple="FGA",
                headers=CUSTOM_HEADERS,
                timeout=120
            ).get_data_frames()[0]

            # Add season label
            shot_data["SEASON"] = season

            all_seasons_data.append(shot_data)

        except Exception as e:
            print(f"ERROR for season {season}: {e}")

        # Small delay to avoid API rate limits
        time.sleep(2)

    # -----------------------------
    # Final save routine
    # -----------------------------
    if all_seasons_data:
        final_df = pd.concat(all_seasons_data, ignore_index=True)

        # Ensure folder exists
        os.makedirs(path, exist_ok=True)

        final_df.to_parquet(full_path, index=False)

        print(f"\nDONE! Saved file to: {full_path}")
        print(f"Total number of records: {len(final_df)}")

        return None
    else:
        print("No data collected.")
        return None
