import pandas as pd

from constants import FEATURE_TRANSFORM_LABELS


def format_int(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.0f}"


def format_float(value, digits: int = 3) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.{digits}f}"


def display_model_family_name(value: str) -> str:
    labels = {
        "baseline": "Baseline",
        "linear": "Linear Models",
        "autoregressive": "Autoregressive Models",
        "tree": "Tree-Based Models",
        "neural_net": "Neural Nets",
        "neural": "Neural Nets",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())


def display_model_family_prefix(value: str) -> str:
    labels = {
        "baseline": "Baseline",
        "linear": "Linear",
        "autoregressive": "Autoregressive",
        "tree": "Tree",
        "neural_net": "Neural Net",
        "neural": "Neural Net",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())


def display_model_build_name(value: str) -> str:
    labels = {
        "seasonal_naive": "Seasonal naive",
        "elastic_net": "Elastic net",
        "random_forest": "Random forest",
        "extra_trees": "Extra trees",
        "xgboost": "XGBoost",
        "arima": "ARIMA",
        "sarima": "SARIMA",
        "sarimax": "SARIMAX",
        "gru": "GRU",
        "lstm": "LSTM",
        "mlp": "MLP",
        "cnn": "CNN",
        "rnn": "RNN",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())


def model_build_display_label(model_family, model_build) -> str:
    return f"{display_model_family_prefix(model_family)}: {display_model_build_name(model_build)}"


def configurations_label(value) -> str:
    count = format_int(value)
    noun = "configuration" if pd.notna(value) and int(value) == 1 else "configurations"
    return f"{count} {noun}"


def date_range_label(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df:
        return "-"
    dates = pd.to_datetime(df[column])
    return f"{dates.min().date()} to {dates.max().date()}"


def manifest_value(manifest: dict, key: str, fallback="-") -> str:
    value = manifest.get(key)
    if value is None or value == "":
        return fallback
    return str(value)


def months_label(value) -> str:
    if value is None or value == "" or value == "-":
        return "-"
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric_value} month" if numeric_value == 1 else f"{numeric_value} months"


def feature_transform_label(value) -> str:
    return FEATURE_TRANSFORM_LABELS.get(str(value), str(value).replace("_", " ").title())
