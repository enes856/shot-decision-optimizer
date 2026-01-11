import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report, r2_score, mean_absolute_error, brier_score_loss, mean_squared_error
from sklearn.calibration import calibration_curve

# ==========================================
# 1. DATA PREPARATION & SPLITTING
# ==========================================

def clean_data_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops metadata columns that are not useful for prediction (IDs, names, raw dates).
    Keeps only features relevant for the model + the target variable.
    
    Args:
        df (pd.DataFrame): The engineered dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe ready for splitting.
    """
    df = df.copy()
    
    # List of columns to drop (Metadata & Redundant info)
    # Note: We keep 'SEASON' and 'GAME_DATE' for the splitting function later.
    drop_cols = [
        # Metadata / IDs
        'GRID_TYPE', 'GAME_ID', 'GAME_EVENT_ID', 'PLAYER_ID', 'PLAYER_NAME', 
        'TEAM_ID', 'TEAM_NAME', 'TEAM_ABBREVIATION', 'OPPONENT_TEAM',
        'HTM', 'VTM', 
        
        # Redundant Time/Zone info
        'MINUTES_REMAINING', 'SECONDS_REMAINING', 
        'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE', 
        'LOC_X', 'LOC_Y',
        
        # --- DATA LEAKAGE & NOISE ---
        'EVENT_TYPE',
        'SHOT_ATTEMPTED_FLAG',
        'SHOT_TYPE',
        
        # Helper Columns from Feature Engineering (if they exist)
        'SEASON_MAKES', 
        'SEASON_ATTEMPTS',
        'GAME_ID_KEY', 'gameId_KEY'
    ]
    
    # Drop existing columns only
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    df_cleaned = df.drop(columns=cols_to_drop)
    
    return df_cleaned

def get_feature_lists(df: pd.DataFrame):
    """
    Identifies numerical and categorical columns automatically.
    Excludes Target, Date, and Season columns from the feature set.
    
    Returns:
        tuple: (list_of_numerical_cols, list_of_categorical_cols)
    """
    # Exclude Target and Split columns from features
    exclude_cols = ['SHOT_MADE_FLAG', 'GAME_DATE', 'SEASON']
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove excluded columns
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    categorical_cols = [c for c in categorical_cols if c not in exclude_cols]
    
    return numeric_cols, categorical_cols

def split_data_by_season(df: pd.DataFrame, split_season: str, target_col: str = 'SHOT_MADE_FLAG'):
    """
    Splits the data based on the 'SEASON' column to prevent data leakage.
    
    Args:
        df (pd.DataFrame): The full dataset.
        split_season (str): The SEASON that marks the start of the TEST set (e.g., '2024-25').
                            - TRAIN: All seasons strictly BEFORE this season.
                            - TEST: This season AND all subsequent seasons (e.g., '2025-26').
        target_col (str): The name of the target variable.

    Returns:
        X_train, y_train, X_test, y_test
    """
    # Ensure chronological order
    df = df.sort_values(by=['GAME_DATE'])
    
    print(f"Splitting data. Split Season: {split_season}")
    
    # Ensure SEASON column exists
    if 'SEASON' not in df.columns:
        raise KeyError("Column 'SEASON' not found. Please check your data ingestion.")

    # Split logic:
    # Train = All seasons logically smaller (older) than split_season
    # Test = The split season and anything newer
    train_df = df[df['SEASON'] < split_season].copy()
    test_df = df[df['SEASON'] >= split_season].copy()
    
    if len(test_df) == 0:
        available_seasons = sorted(df['SEASON'].unique().tolist())
        raise ValueError(f"No data found for Split Season '{split_season}'. Available: {available_seasons}")
        
    print(f"Train Set Seasons: {sorted(train_df['SEASON'].unique().tolist())} ({len(train_df)} shots)")
    print(f"Test Set Seasons:  {sorted(test_df['SEASON'].unique().tolist())} ({len(test_df)} shots)")
    
    # Define columns to drop (Target + Metadata used for splitting)
    cols_to_exclude = [target_col, 'GAME_DATE', 'SEASON']
    
    # Create X and y
    X_train = train_df.drop(columns=cols_to_exclude, errors='ignore')
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=cols_to_exclude, errors='ignore')
    y_test = test_df[target_col]
    
    return X_train, y_train, X_test, y_test

# ==========================================
# 2. MODEL PIPELINE BUILDING
# ==========================================

def build_logistic_pipeline(numeric_features: list, categorical_features: list):
    """
    Creates a Scikit-Learn Pipeline that:
    1. Imputes missing values (Mean for numeric, Constant for categorical).
    2. Scales numerical features (StandardScaler).
    3. One-Hot Encodes categorical features.
    4. Trains a Logistic Regression model.
    
    Args:
        numeric_features (list): List of column names.
        categorical_features (list): List of column names.
        
    Returns:
        sklearn.pipeline.Pipeline: The untrained model pipeline.
    """
    
    # 1. Pipeline for Numerical Features
    # - Impute NaNs with Mean (crucial for 'Cold Start' players/stats)
    # - Scale features (Logistic Regression works best with scaled data)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()) 
    ])

    # 2. Pipeline for Categorical Features
    # - Handle missing categories
    # - One-Hot Encode (convert text to binary columns)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Combine Transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 4. Final Pipeline with Classifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42))
    ])
    
    return model

from sklearn.model_selection import GridSearchCV

def train_with_grid_search(model_pipeline, X_train, y_train):
    """
    Trains the model using GridSearchCV to find the optimal hyperparameters.
    It uses Cross-Validation (CV) to ensure the model is robust.
    
    Args:
        model_pipeline: The untrained Scikit-Learn pipeline.
        X_train, y_train: Training data.
        
    Returns:
        The best trained model found by the search.
    """
    # 1. Define the "Grid" of parameters to test
    # 'classifier__C' targets the 'C' parameter of the LogisticRegression step.
    # C = Inverse of regularization strength (Smaller = stronger regularization/simplification).
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100] 
    }

    # 2. Setup Grid Search
    # cv=3: Splits data into 3 parts. Trains on 4, validates on 1. Repeats 3 times.
    # scoring='neg_log_loss': The metric to optimize. Log Loss is best for probabilities (xPts).
    print("Starting Grid Search with 3-Fold Cross-Validation...")
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3,
        scoring='neg_log_loss', 
        n_jobs=-1,
        verbose=1
    )

    # 3. Train (Fit)
    grid_search.fit(X_train, y_train)

    # 4. Results
    print(f"\nBest Parameter (C): {grid_search.best_params_['classifier__C']}")
    print(f"Best CV Log Loss:   {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# ==========================================
# 3. EVALUATION & xPts
# ==========================================

def calculate_xpts(df: pd.DataFrame, prob_col: str = 'PROBABILITY_MAKE', value_col: str = 'SHOT_VALUE') -> pd.DataFrame:
    """
    Calculates Expected Points (xPts) and Actual Points for each shot.
    
    Args:
        df (pd.DataFrame): Dataframe containing predictions.
        prob_col (str): Column name for the predicted probability (0 to 1).
        value_col (str): Column name for the shot value (2 or 3).
        
    Returns:
        pd.DataFrame: Dataframe with new columns 'xPts' and 'ACTUAL_POINTS'.
    """
    df = df.copy()
    
    if prob_col not in df.columns or value_col not in df.columns:
        raise KeyError(f"Columns {prob_col} or {value_col} missing in DataFrame.")

    # xPts = Probability * Shot Value
    df['xPts'] = df[prob_col] * df[value_col]
    
    # Actual Points = Make (0/1) * Shot Value
    df['ACTUAL_POINTS'] = df['SHOT_MADE_FLAG'] * df[value_col]
    
    return df

def evaluate_model_metrics(model, X_test, y_test):
    """
    Standard ML Evaluation: Accuracy, ROC-AUC, Classification Report.
    Returns probabilities for further analysis.
    """
    print("\n" + "="*50)
    print("STANDARD MODEL METRICS")
    print("="*50)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probability of Class 1 (Make)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    logloss = log_loss(y_test, y_prob)
    
    print(f"Log Loss: {logloss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_prob

def evaluate_game_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates xPts and Actual Points per Game to evaluate 'Macro' performance.
    Plots Predicted Score vs Actual Score.
    
    Returns:
        pd.DataFrame: Game-level statistics for further analysis.
    """
    print("\n" + "="*60)
    print("GAME-LEVEL EVALUATION (Sum of xPts)")
    print("="*60)
    
    required_cols = ['GAME_ID', 'TEAM_ABBREVIATION', 'xPts', 'ACTUAL_POINTS']
    if not all(c in df.columns for c in required_cols):
        print(f"Error: Missing columns for game aggregation. Need: {required_cols}")
        return None

    game_stats = df.groupby(['GAME_ID', 'TEAM_ABBREVIATION']).agg(
        Predicted_Score=('xPts', 'sum'),
        Actual_Score=('ACTUAL_POINTS', 'sum'),
        Shot_Count=('xPts', 'count')
    ).reset_index()
    
    mae = mean_absolute_error(game_stats['Actual_Score'], game_stats['Predicted_Score'])
    mse = np.mean((game_stats['Actual_Score'] - game_stats['Predicted_Score']) ** 2)
    r2 = r2_score(game_stats['Actual_Score'], game_stats['Predicted_Score'])
    
    print(f"Games Evaluated: {len(game_stats):,}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} points per game")
    print(f"Mean Squared Error (MSE):  {mse:.2f} points per game")
    print(f"R² Score (Correlation):    {r2:.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(
        game_stats['Predicted_Score'], 
        game_stats['Actual_Score'],
        c=game_stats['Shot_Count'],
        cmap='viridis',
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )
    
    min_val = min(game_stats['Predicted_Score'].min(), game_stats['Actual_Score'].min())
    max_val = max(game_stats['Predicted_Score'].max(), game_stats['Actual_Score'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
            label='Perfect Prediction', linewidth=2, alpha=0.8)
    
    ax.set_title(f'Game Validation: Predicted (xPts) vs Actual Score\n'
                 f'MAE: {mae:.2f} pts | MSE: {mse:.2f} pts | R²: {r2:.3f}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sum of xPts (Model Prediction)', fontsize=12)
    ax.set_ylabel('Actual Score (Real Game Result)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Shot Count', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return game_stats

def evaluate_calibration(df: pd.DataFrame, n_bins: int = 10) -> None:
    """
    Plots the Calibration Curve (Reliability Diagram) combined with a distribution histogram.

    This visualization serves two purposes:
    1. **Reliability (Lines):** Checks if the predicted probabilities match reality.
       (e.g., if the model predicts 70%, do ~70% of those shots actually go in?)
       Curves are plotted separately for All Shots, 2-Pointers, and 3-Pointers to detect bias.
    2. **Distribution (Histogram):** Shows the volume of shots in each probability bin.
       This helps to determine if a point on the curve is statistically significant
       (high volume) or just noise (low volume).

    Args:
        df (pd.DataFrame): Dataframe containing 'PROBABILITY_MAKE', 'SHOT_MADE_FLAG',
                           and 'SHOT_VALUE'.
        n_bins (int): Number of bins to divide the probability range (0-1) into.
                      Default is 10 (bins of 10% width).

    Returns:
        None: Displays the plot and prints a summary table of bin statistics.
    """
    print("\n" + "="*50)
    print(f"CALIBRATION EVALUATION ({n_bins} Bins)")
    print("="*50)
    
    # Setup Plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # --- 1. HISTOGRAM (Distribution of Predictions) ---
    # We put this on ax2 (secondary y-axis) so it stays in the background
    ax2 = ax1.twinx()
    
    # Plot histogram of all predicted probabilities
    ax2.hist(
        df['PROBABILITY_MAKE'], 
        range=(0, 1), 
        bins=n_bins, 
        color='gray', 
        alpha=0.15, # Very transparent
        edgecolor='gray',
        label='Shot Count Distribution'
    )
    ax2.set_ylabel('Number of Shots (Count)', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.grid(False) # Turn off grid for histogram to keep it clean

    # --- 2. CALIBRATION CURVES ---
    # Plot Perfect Line
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    # Helper to plot one curve
    def plot_curve(subset_df, label, color):
        if len(subset_df) == 0: return
        
        y_true = subset_df['SHOT_MADE_FLAG']
        y_prob = subset_df['PROBABILITY_MAKE']
        
        # Calculate curve points
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        brier = brier_score_loss(y_true, y_prob)
        
        ax1.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f'{label} (Brier: {brier:.4f})', color=color)

    # Plot Curves
    plot_curve(df, "All Shots", "black")
    plot_curve(df[df['SHOT_VALUE'] == 2], "2-Pointers", "blue")
    plot_curve(df[df['SHOT_VALUE'] == 3], "3-Pointers", "green")
    
    # Styling ax1
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives (Actual FG%)')
    ax1.set_title('Reliability Diagram & Probability Distribution')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    plt.show()
    
    # --- 3. PRINT DATA TABLE ---
    print("\n--- Detailed Bin Statistics (All Shots) ---")
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    df['prob_bin'] = pd.cut(df['PROBABILITY_MAKE'], bins=bins, include_lowest=True)
    
    # Group by bin
    bin_stats = df.groupby('prob_bin', observed=False).agg(
        Count=('SHOT_MADE_FLAG', 'count'),
        Avg_Pred_Prob=('PROBABILITY_MAKE', 'mean'),
        Actual_FG_Pct=('SHOT_MADE_FLAG', 'mean')
    )
    
    print(bin_stats)


def evaluate_probability_distribution(df: pd.DataFrame, 
                                      feature_col: str = 'CUSTOM_SHOT_ZONE', 
                                      prob_col: str = 'PROBABILITY_MAKE', 
                                      figsize: tuple = (14, 8)) -> None:
    """
    Plots a boxplot showing the distribution (spread) of predicted probabilities 
    grouped by a specific feature. This helps visualize model confidence and 
    variance across different categories (e.g., Shot Zones).

    The categories are automatically sorted by their median predicted probability 
    (highest to lowest) for better readability.

    Args:
        df (pd.DataFrame): The dataframe containing the evaluation data. 
                           Must contain `feature_col` and `prob_col`.
        feature_col (str): The categorical column to group by 
                           (e.g., 'CUSTOM_SHOT_ZONE', 'ACTION_TYPE').
        prob_col (str): The column name containing the predicted probabilities 
                        (default: 'PROBABILITY_MAKE').
        figsize (tuple): The size of the figure (width, height).

    Returns:
        None: Displays the plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. Validation
    if feature_col not in df.columns or prob_col not in df.columns:
        print(f"Error: Columns '{feature_col}' or '{prob_col}' not found in DataFrame.")
        return

    print(f"\n" + "="*50)
    print(f"PROBABILITY DISTRIBUTION BY {feature_col}")
    print("="*50)

    # 2. Sort categories by Median Probability (High -> Low)
    # This makes the plot much easier to read than random alphabetical order
    try:
        order = df.groupby(feature_col)[prob_col].median().sort_values(ascending=False).index
    except Exception as e:
        print(f"Error sorting categories: {e}")
        return

    # 3. Plotting
    plt.figure(figsize=figsize)
    
    # Create Boxplot
    sns.boxplot(
        data=df,
        x=feature_col,
        y=prob_col,
        order=order,
        palette='viridis', # Good contrast color map
        showfliers=True    # Show outliers (dots) to see extreme predictions
    )

    # 4. Styling
    plt.title(f"Spread of Predicted Probabilities by {feature_col}", fontsize=16, fontweight='bold')
    plt.xlabel(feature_col, fontsize=12)
    plt.ylabel(f"Predicted Probability ({prob_col})", fontsize=12)
    
    # Rotate x-labels if there are many categories to prevent overlap
    plt.xticks(rotation=45, ha='right')
    
    # Add grid for easier reading of probability levels
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()