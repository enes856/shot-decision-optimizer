"""
Utility functions for XGBoost modeling: data preparation, pipeline building, and evaluation.
Includes:
1. Data cleaning and splitting by season to prevent leakage.
2. Building a Scikit-Learn pipeline with XGBoost classifier.
3. Evaluation functions for standard metrics, game-level aggregation, and calibration curves.
4. Feature importance and advanced visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, 
    r2_score, mean_absolute_error, brier_score_loss,
    roc_curve, precision_recall_curve, average_precision_score, log_loss
)
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


def split_data_by_season(df: pd.DataFrame, split_season: str, 
                         validation_season: str = None, 
                         target_col: str = 'SHOT_MADE_FLAG'):
    """
    Splits the data based on the 'SEASON' column to prevent data leakage.
    Optionally creates a validation set from a specific season.
    
    Args:
        df (pd.DataFrame): The full dataset.
        split_season (str): The SEASON that marks the start of the TEST set (e.g., '2024-25').
        validation_season (str, optional): Season to use for validation (e.g., '2023-24').
                                          If None, no validation set is created.
        target_col (str): The name of the target variable.

    Returns:
        If validation_season is None: X_train, y_train, X_test, y_test
        If validation_season provided: X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Ensure chronological order
    df = df.sort_values(by=['GAME_DATE'])
    
    print(f"Splitting data. Test Season: {split_season}")
    
    if 'SEASON' not in df.columns:
        raise KeyError("Column 'SEASON' not found. Please check your data ingestion.")

    # Define splits
    test_df = df[df['SEASON'] >= split_season].copy()
    
    if validation_season:
        print(f"Validation Season: {validation_season}")
        train_df = df[df['SEASON'] < validation_season].copy()
        val_df = df[df['SEASON'] == validation_season].copy()
        
        if len(val_df) == 0:
            raise ValueError(f"No data found for validation season '{validation_season}'.")
    else:
        train_df = df[df['SEASON'] < split_season].copy()
        val_df = None
    
    if len(test_df) == 0:
        available_seasons = sorted(df['SEASON'].unique().tolist())
        raise ValueError(f"No data found for Test Season '{split_season}'. Available: {available_seasons}")
        
    print(f"Train Set Seasons: {sorted(train_df['SEASON'].unique().tolist())} ({len(train_df):,} shots)")
    if val_df is not None:
        print(f"Val Set Seasons:   {sorted(val_df['SEASON'].unique().tolist())} ({len(val_df):,} shots)")
    print(f"Test Set Seasons:  {sorted(test_df['SEASON'].unique().tolist())} ({len(test_df):,} shots)")
    
    cols_to_exclude = [target_col, 'GAME_DATE', 'SEASON']
    
    X_train = train_df.drop(columns=cols_to_exclude, errors='ignore')
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=cols_to_exclude, errors='ignore')
    y_test = test_df[target_col]
    
    if val_df is not None:
        X_val = val_df.drop(columns=cols_to_exclude, errors='ignore')
        y_val = val_df[target_col]
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    return X_train, y_train, X_test, y_test


# ==========================================
# 2. MODEL PIPELINE BUILDING
# ==========================================

def build_xgboost_pipeline(numeric_features: list, categorical_features: list, 
                          xgb_params: dict = None):
    """
    Creates a Scikit-Learn Pipeline for an XGBoost model.
    The pipeline handles imputation, scaling, and encoding.
    
    Args:
        numeric_features (list): List of numerical column names.
        categorical_features (list): List of categorical column names.
        xgb_params (dict, optional): Hyperparameters to override XGBoost defaults.
        
    Returns:
        sklearn.pipeline.Pipeline: The untrained XGBoost model pipeline.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Improved default XGBoost parameters
    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 500,  # Will use early stopping, so this is max
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,  # Minimum loss reduction for split
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'scale_pos_weight': 1.0,  # Handle class imbalance if needed
        'early_stopping_rounds': 50,  # Early stopping patience
        'use_label_encoder': False,
        'random_state': 42,
        'n_jobs': 1
    }
    
    if xgb_params:
        default_params.update(xgb_params)

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(**default_params))
    ])
    
    return model


def fit_pipeline_with_early_stopping(pipeline, X_train, y_train, X_val=None, y_val=None, 
                                     verbose=False):
    """
    Fits a sklearn pipeline containing XGBoost with early stopping.
    
    This function handles the complexity of using early stopping with sklearn pipelines
    by pre-fitting the preprocessor and then training XGBoost with the transformed data.
    
    Args:
        pipeline: sklearn Pipeline with 'preprocessor' and 'classifier' steps.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features (optional).
        y_val: Validation labels (optional).
        verbose: Whether to print training progress.
    
    Returns:
        Fitted pipeline.
    """
    print("Fitting preprocessor on training data...")
    # Fit the preprocessor
    pipeline.named_steps['preprocessor'].fit(X_train)
    
    # Transform the data
    X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
    
    # Prepare eval_set for early stopping if validation data provided
    eval_set = None
    if X_val is not None and y_val is not None:
        X_val_transformed = pipeline.named_steps['preprocessor'].transform(X_val)
        eval_set = [(X_val_transformed, y_val)]
        xgb_classifier = pipeline.named_steps['classifier']
        early_stop_rounds = getattr(xgb_classifier, 'early_stopping_rounds', 50)
        print(f"Training XGBoost with early stopping (patience={early_stop_rounds})...")
    else:
        print("Training XGBoost without early stopping...")
    
    # Fit the classifier with early stopping
    if eval_set is not None:
        xgb_classifier = pipeline.named_steps['classifier']
        
        xgb_classifier.fit(
            X_train_transformed,
            y_train,
            eval_set=eval_set,
            verbose=verbose
        )
        
        # Report early stopping results if available
        if hasattr(xgb_classifier, 'best_iteration') and xgb_classifier.best_iteration is not None:
            best_iter = xgb_classifier.best_iteration
            n_estimators = xgb_classifier.n_estimators
            print(f"Early stopping triggered at iteration {best_iter} (out of {n_estimators} max)")
        else:
            print(f"Training completed with {xgb_classifier.n_estimators} estimators (no early stopping)")
    else:
        pipeline.named_steps['classifier'].fit(X_train_transformed, y_train)
        print(f"Training completed with {pipeline.named_steps['classifier'].n_estimators} estimators")
    
    return pipeline

def plot_hyperparameter_optimization_results(random_search):
    """
    Plots a scatter plot of the hyperparameter search results (Learning Rate vs Max Depth).
    Uses Seaborn, which is robust to small sample sizes.
    """

    results_df = pd.DataFrame(random_search.cv_results_)
    
    # Define parameters to plot
    param_x = 'param_classifier__learning_rate'
    param_y = 'param_classifier__max_depth'
    score_col = 'mean_test_score'
    
    # Check if these specific params exist
    if param_x not in results_df.columns or param_y not in results_df.columns:
        print(f"Skipping plot: Desired parameters ({param_x}, {param_y}) not found in search results.")
        return
        
    try:
        # Prepare data for plotting
        plot_df = results_df[[param_x, param_y, score_col]].copy()
        plot_df[param_x] = plot_df[param_x].astype(float)
        plot_df[param_y] = plot_df[param_y].astype(float)
        plot_df[score_col] = plot_df[score_col].astype(float)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot: Color and Size by Score
        sns.scatterplot(
            data=plot_df,
            x=param_x,
            y=param_y,
            size=score_col,
            hue=score_col,
            palette='viridis',
            sizes=(50, 200),
            alpha=0.8,
            ax=ax
        )
        
        # Mark best model
        best_idx = plot_df[score_col].idxmax()
        best_x = plot_df.loc[best_idx, param_x]
        best_y = plot_df.loc[best_idx, param_y]
        best_score = plot_df.loc[best_idx, score_col]
        
        ax.scatter(best_x, best_y, c='red', s=300, marker='*', 
                   label=f'Best (AUC={best_score:.4f})', zorder=10)
        
        ax.set_xlabel('Learning Rate', fontsize=12)
        ax.set_ylabel('Max Depth', fontsize=12)
        ax.set_title('Hyperparameter Search Landscape', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not generate plot: {e}")

def tune_xgboost_hyperparameters(pipeline, X_train, y_train, n_iter=20, cv=3, scoring='roc_auc'):
    """
    Performs randomized search for hyperparameter tuning.
    
    Args:
        pipeline: The model pipeline to tune.
        X_train: Training features.
        y_train: Training labels.
        n_iter: Number of parameter settings sampled.
        cv: Number of cross-validation folds.
        scoring: Metric to optimize. Common options: 'roc_auc' (discrimination), 'neg_log_loss' (probability accuracy).
    
    Returns:
        Best estimator from the search.
    """
    param_distributions = {
        'classifier__max_depth': [4, 5, 6, 7, 8],
        'classifier__min_child_weight': [1, 3, 5, 7],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__subsample': [0.7, 0.8, 0.9],
        'classifier__colsample_bytree': [0.7, 0.8, 0.9],
        'classifier__gamma': [0, 0.1, 0.2],
        'classifier__reg_alpha': [0, 0.1, 0.5],
        'classifier__reg_lambda': [0.5, 1.0, 2.0],
        'classifier__early_stopping_rounds': [None]
    }
    
    print(f"\nStarting RandomizedSearchCV with {n_iter} iterations and {cv}-fold CV...")
    
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"\nBest {scoring}: {random_search.best_score_:.4f}")
    print("Best parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    # Visualize results
    # plot_hyperparameter_optimization_results(random_search)
    
    return random_search.best_estimator_


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
    
    return y_prob, y_pred



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


def plot_shap_summary(model, X_test, sample_size=1000):
    classifier = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']

    # Draw a sample and transform it
    X_sample = X_test.sample(n=min(len(X_test), sample_size), random_state=42)
    X_transformed = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()

    # 2. Utilize KernelExplainer
    # We pass the predict_proba function of the classifier.
    # Since we only want the probability for "Shot Made" (Class 1), we use a lambda function.
    model_func = lambda x: classifier.predict_proba(x)[:, 1]

    # Use a summary of the transformed data as a reference ("Background")
    background = shap.kmeans(X_transformed, 5) # Representative background for faster computation
    explainer = shap.KernelExplainer(model_func, background)

    print("Computing SHAP values (this may take a minute)...")
    shap_values = explainer.shap_values(X_transformed)

    # 3. Plotting
    plt.figure(figsize=(16, 12))
    # Create a DataFrame for the plot so that feature names are displayed correctly
    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    shap.summary_plot(
        shap_values,
        X_df,
        plot_type="dot",
        show=False,
    )
    plt.title("SHAP Feature Impact on Shot Probability", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_shap_waterfall(model, X_test, instance_index=0):
    classifier = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']

    # 1. Data transformation
    X_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    model_func = lambda x: classifier.predict_proba(x)[:, 1]
    background = shap.kmeans(X_transformed, 5)
    explainer = shap.KernelExplainer(model_func, background)

    # 3. calculate SHAP values for the specific instance
    instance = X_transformed[instance_index : instance_index + 1]
    shap_values = explainer.shap_values(instance)
    expected_value = explainer.expected_value

    # shap doesnt provide an Explanation object for KernelExplainer, so we build it ourselves:
    exp = shap.Explanation(
        values=shap_values[0], 
        base_values=expected_value, 
        data=instance[0], 
        feature_names=feature_names
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, show=False)
    plt.title(f"Waterfall Plot for Instance {instance_index}", fontsize=14, pad=20)
    plt.show()


def plot_feature_importance(model, feature_names: list = None, top_n: int = 20):
    """
    Plots feature importance from the trained XGBoost model.
    
    Args:
        model: Trained pipeline containing XGBoost classifier.
        feature_names: List of feature names after preprocessing.
        top_n: Number of top features to display.
    """
    
    # Extract the XGBoost classifier from pipeline
    xgb_model = model.named_steps['classifier']
    
    # Get feature importances
    importances = xgb_model.feature_importances_
    
    # Get feature names from the preprocessor if not provided
    if feature_names is None:
        try:
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        except:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
    bars = ax.barh(range(len(importance_df)), importance_df['Importance'], color=colors)
    
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['Feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (Gain)', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features (XGBoost)', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return importance_df

def plot_shap_dependence(model, X_test, feature_to_plot, interaction_feature=None, sample_size=1000):
    classifier = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']

    X_sample = X_test.sample(n=min(len(X_test), sample_size), random_state=42)
    X_transformed = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()

    model_func = lambda x: classifier.predict_proba(x)[:, 1]
    background = shap.kmeans(X_transformed, 5)
    explainer = shap.KernelExplainer(model_func, background)

    print(f"Computing SHAP values for dependence plot (this may take a minute for {sample_size} samples)...")
    shap_values = explainer.shap_values(X_transformed)

    X_df_transformed = pd.DataFrame(X_transformed, columns=feature_names)

    try:
        feature_idx = feature_names.tolist().index(feature_to_plot)
        if interaction_feature:
            interaction_idx = feature_names.tolist().index(interaction_feature)
        else:
            interaction_idx = "auto" # SHAP finds the feature with the strongest interaction
    except ValueError as e:
        print(f"Error: Feature '{e}' not found in transformed features.")
        return

    plt.figure(figsize=(14, 10))
    shap.dependence_plot(
        ind=feature_idx,
        shap_values=shap_values,
        features=X_df_transformed,
        feature_names=feature_names,
        interaction_index=interaction_idx,
        dot_size=30,
        alpha=0.6,
        show=False
    )
    plt.title(f"SHAP Dependence Plot: {feature_to_plot} vs. SHAP Value", fontsize=14, fontweight='bold')
    plt.ylabel(f'SHAP value for {feature_to_plot}', fontsize=12)
    plt.xlabel(f'Feature value for {feature_to_plot}', fontsize=12)
    plt.tight_layout()
    plt.show()



def plot_roc_curves(y_test, y_prob_xgb, y_prob_baseline=None):
    """
    Plots ROC curves comparing XGBoost to a baseline (if provided).
    
    Args:
        y_test: True labels.
        y_prob_xgb: Predicted probabilities from XGBoost.
        y_prob_baseline: Predicted probabilities from baseline model (optional).
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot baseline (no-skill)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', 
            linewidth=2, label='No Skill (AUC=0.50)', alpha=0.7)
    
    # Plot XGBoost
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    auc_xgb = roc_auc_score(y_test, y_prob_xgb)
    ax.plot(fpr_xgb, tpr_xgb, linewidth=2.5, 
            label=f'XGBoost (AUC={auc_xgb:.4f})', color='darkblue')
    
    # Plot baseline model if provided
    if y_prob_baseline is not None:
        fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_baseline)
        auc_base = roc_auc_score(y_test, y_prob_baseline)
        ax.plot(fpr_base, tpr_base, linewidth=2.5, linestyle='--',
                label=f'Baseline (AUC={auc_base:.4f})', color='darkorange')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"XGBoost AUC: {auc_xgb:.4f}")
    if y_prob_baseline is not None:
        print(f"Baseline AUC: {auc_base:.4f}")
        print(f"Improvement: {(auc_xgb - auc_base):.4f} ({((auc_xgb/auc_base - 1)*100):.2f}%)")


def plot_precision_recall_curve(y_test, y_prob):
    """
    Plots the Precision-Recall curve.
    Useful for imbalanced datasets where ROC can be optimistic.
    
    Args:
        y_test: True labels.
        y_prob: Predicted probabilities.
    """
    print("\n" + "="*60)
    print("PRECISION-RECALL CURVE")
    print("="*60)
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    
    # Baseline is the proportion of positive class
    baseline_precision = y_test.mean()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot([0, 1], [baseline_precision, baseline_precision], 
            linestyle='--', color='gray', linewidth=2, 
            label=f'No Skill (AP={baseline_precision:.3f})', alpha=0.7)
    
    ax.plot(recall, precision, linewidth=2.5, color='darkgreen',
            label=f'XGBoost (AP={avg_precision:.4f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Average Precision Score: {avg_precision:.4f}")
    print(f"Baseline (No Skill):     {baseline_precision:.4f}")