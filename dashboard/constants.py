from pathlib import Path

DEFAULT_ARTIFACT_DIR = Path("dashboard/public_artifacts/latest")
IMAGE_ASSET_DIR = Path("dashboard/assets/images")
VIDEO_ASSET_DIR = Path("dashboard/assets/videos")
DEFAULT_FEATURE_FAMILIES_PATH = Path("dashboard/public_artifacts/latest/feature_families.json")
EDA_INPUT_DIRNAME = "eda_inputs"
EDA_INTEGRATED_BASE_FILENAME = "integrated_monthly_base.parquet"
EDA_FEATURE_TABLE_FILENAME = "feature_table.parquet"
EDA_IMPUTATION_LOG_FILENAME = "imputation_log.parquet"
DEFAULT_INTEGRATED_BASE_PATH = DEFAULT_ARTIFACT_DIR / EDA_INPUT_DIRNAME / EDA_INTEGRATED_BASE_FILENAME
DEFAULT_FEATURE_TABLE_PATH = DEFAULT_ARTIFACT_DIR / EDA_INPUT_DIRNAME / EDA_FEATURE_TABLE_FILENAME
DEFAULT_IMPUTATION_LOG_PATH = DEFAULT_ARTIFACT_DIR / EDA_INPUT_DIRNAME / EDA_IMPUTATION_LOG_FILENAME
LOCAL_INTEGRATED_BASE_PATH = Path("raw_files/integrated_monthly_base.parquet")
LOCAL_FEATURE_TABLE_PATH = Path("feature_store/income_interactions_h3_v1/feature_table.parquet")
LOCAL_IMPUTATION_LOG_PATH = Path("feature_store/income_interactions_h3_v1/imputation_log.parquet")
RESULTS_INSIGHTS_NOTEBOOK_PATH = Path("experiment_results_insights.ipynb")
PHASE_A_V3_CONFIG_PATH = Path("experiment_configs/large_phase_a_v3_pandemic_safe.yaml")
PHASE_B_V3_CONFIG_PATH = Path("experiment_configs/phase_b_autoregressive_v3_pandemic_safe.yaml")
PHASE_C_MONTHLY_CONFIG_PATH = Path("experiment_configs/phase_c_neural_monthly_finalists.yaml")

# Keep the MOV as the editable capture; use MP4 for browser and Streamlit Cloud playback.
SYSTEM_ARCH_VIDEO_PATH = IMAGE_ASSET_DIR / "Transit_System_Build.mp4"
PROJECT_OVERVIEW_VIDEO_PATH = VIDEO_ASSET_DIR / "Transit_Site_Walkthrough_v2_web.mp4"
STEP_FUNCTION_SCREENSHOT_PATH = IMAGE_ASSET_DIR / "Step_Function_Screenshot.png"
VIDEO_MIME_TYPES = {".mov": "video/quicktime", ".mp4": "video/mp4", ".webm": "video/webm"}

MODEL_FAMILY_ORDER = ["baseline", "linear", "autoregressive", "tree", "neural_net", "neural"]
MODEL_BUILD_ORDER = [
    "seasonal_naive",
    "ridge",
    "lasso",
    "elastic_net",
    "arima",
    "sarima",
    "sarimax",
    "random_forest",
    "extra_trees",
    "xgboost",
    "mlp",
    "cnn",
    "rnn",
    "gru",
    "lstm",
]
BASELINE_MODEL_FAMILIES = {"baseline"}
BASELINE_MODEL_BUILDS = {"seasonal_naive", "naive"}

REQUIRED_FILES = {
    "forecast_paths": "forecast_paths.parquet",
    "performance_over_time": "performance_over_time.parquet",
    "model_leaderboard": "model_leaderboard.parquet",
    "feature_family_summary": "feature_family_summary.parquet",
    "champion_predictions": "champion_predictions.parquet",
    "champion_selection": "champion_selection.json",
}
OPTIONAL_FILES = {
    "model_leaderboard_full": "model_leaderboard_full.parquet",
    "feature_family_summary_full": "feature_family_summary_full.parquet",
    "complexity_profile_full": "complexity_profile_full.parquet",
    "path_partition_manifest": "path_partition_manifest.json",
    "overview_top_models": "overview_top_models.parquet",
    "overview_prediction_paths": "overview_prediction_paths.parquet",
    "experiment_manifest": "experiment_manifest.json",
}
PATH_DATASET_DIRS = {
    "forecast_paths": "forecast_paths_by_build",
    "performance_over_time": "performance_over_time_by_build",
}

SCORE_RECIPES = {
    "balanced": {"label": "Balanced score", "mae_weight": 0.75, "rmse_weight": 0.25},
}
PER_BUILD_LIMIT_OPTIONS = ["Top 1", "Top 3", "Top 5", "Top 10", "Top 25", "All"]
TOTAL_LIMIT_OPTIONS = ["All", "Top 5", "Top 10", "Top 15", "Top 25", "Top 50", "Top 100"]
EVALUATION_PERIODS = {
    "pre_covid": "Pre-COVID",
    "covid_shock": "COVID shock",
    "recovery": "Recovery",
    "recent": "Recent",
}

RANK_METRIC_OPTIONS = {
    "Balanced score": ("selection_score_balanced", True),
    "MAE": ("mae", True),
    "RMSE": ("rmse", True),
    "R-squared": ("r2", False),
    "Adjusted R-squared": ("r2_adjusted", False),
    "Directional accuracy": ("diracc", False),
    "Pre-COVID MAE": ("pre_covid_mae", True),
    "Pre-COVID RMSE": ("pre_covid_rmse", True),
    "COVID shock MAE": ("covid_shock_mae", True),
    "COVID shock RMSE": ("covid_shock_rmse", True),
    "Recovery MAE": ("recovery_mae", True),
    "Recovery RMSE": ("recovery_rmse", True),
    "Recent MAE": ("recent_mae", True),
    "Recent RMSE": ("recent_rmse", True),
    "Shock penalty": ("shock_penalty", True),
    "Recovery ratio": ("recovery_ratio", True),
    "Recent recovery ratio": ("recent_recovery_ratio", True),
    "RMSE shock penalty": ("rmse_shock_penalty", True),
    "RMSE recovery ratio": ("rmse_recovery_ratio", True),
    "RMSE recent recovery ratio": ("rmse_recent_recovery_ratio", True),
}

FEATURE_POLICY_DESCRIPTIONS = {
    "none": "Use every column in the selected feature family.",
    "corr_pruned": "Within each as-of training window, drop features that are highly correlated with earlier columns.",
    "variance_pruned": "Within each as-of training window, drop near-constant features with almost no variance.",
    "mutual_info_top_20": "Rank features by mutual information with the target inside the training window and keep the top 20.",
    "mutual_info_top_30": "Rank features by mutual information with the target inside the training window and keep the top 30.",
    "lasso_selected": "Fit a Lasso selector inside the training window and keep features with nonzero coefficients.",
    "tree_top_20": "Fit a shallow Extra Trees selector inside the training window and keep the 20 most important features.",
    "tree_top_30": "Fit a shallow Extra Trees selector inside the training window and keep the 30 most important features.",
}

FEATURE_TRANSFORM_DESCRIPTIONS = {
    "identity": "Use the selected feature columns as-is.",
    "log_signed": "Add signed log1p versions of selected features, preserving direction for negative values.",
    "quadratic": "Add squared terms for selected features so linear models can fit curved relationships.",
    "cubic": "Add squared and cubed terms for selected features so linear models can fit stronger nonlinear curvature.",
    "log_signed_quadratic_cubic": "Add signed-log, squared, and cubed versions of selected features in one expanded representation.",
}

FEATURE_TRANSFORM_LABELS = {
    "identity": "No transform",
    "log_signed": "Signed log",
    "quadratic": "Quadratic",
    "cubic": "Cubic",
    "log_signed_quadratic_cubic": "Signed log + quadratic + cubic",
}
FEATURE_TRANSFORM_ORDER = [
    "identity",
    "log_signed",
    "quadratic",
    "cubic",
    "log_signed_quadratic_cubic",
]

PERIOD_RANK_WINDOWS = {
    "Pre-COVID MAE": (None, "2020-02-01"),
    "Pre-COVID RMSE": (None, "2020-02-01"),
    "COVID shock MAE": ("2020-03-01", "2021-06-01"),
    "COVID shock RMSE": ("2020-03-01", "2021-06-01"),
    "Shock penalty": ("2020-03-01", "2021-06-01"),
    "RMSE shock penalty": ("2020-03-01", "2021-06-01"),
    "Recovery MAE": ("2021-07-01", "2022-12-01"),
    "Recovery RMSE": ("2021-07-01", "2022-12-01"),
    "Recovery ratio": ("2021-07-01", "2022-12-01"),
    "RMSE recovery ratio": ("2021-07-01", "2022-12-01"),
    "Recent MAE": ("2023-01-01", None),
    "Recent RMSE": ("2023-01-01", None),
    "Recent recovery ratio": ("2023-01-01", None),
    "RMSE recent recovery ratio": ("2023-01-01", None),
}
