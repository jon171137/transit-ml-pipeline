"""Build a DuckDB experiment mart from model result artifacts.

This is an additive local analytics layer. The model runner remains responsible
for producing portable Parquet/JSON artifacts; this script loads those artifacts
into a queryable DuckDB file and can export dashboard-shaped Parquet files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq


RESULT_TABLES = {
    "predictions": "predictions.parquet",
    "model_runs": "model_runs.parquet",
    "metrics": "metrics.parquet",
    "feature_importance": "feature_importance.parquet",
    "feature_sets": "feature_sets.parquet",
    "feature_family_summary": "feature_family_summary.parquet",
    "complexity_profile": "complexity_profile.parquet",
}

DASHBOARD_EXPORTS = {
    "forecast_paths": "forecast_paths.parquet",
    "performance_over_time": "performance_over_time.parquet",
    "model_leaderboard": "model_leaderboard.parquet",
    "feature_family_summary_dashboard": "feature_family_summary.parquet",
    "champion_predictions": "champion_predictions.parquet",
    "overview_top_models": "overview_top_models.parquet",
    "overview_prediction_paths": "overview_prediction_paths.parquet",
    "complexity_profile_dashboard": "complexity_profile.parquet",
}

JSON_ARTIFACTS = {
    "champion_selection": "champion_selection.json",
    "experiment_manifest": "experiment_manifest.json",
    "batch_manifest": "batch_manifest.json",
}

EMPTY_TABLE_SCHEMAS = {
    "feature_importance": {
        "feature_name": "VARCHAR",
        "importance_type": "VARCHAR",
        "importance": "DOUBLE",
        "importance_abs": "DOUBLE",
        "experiment_id": "VARCHAR",
        "pipeline_run_id": "VARCHAR",
        "model_run_id": "VARCHAR",
        "model_config_id": "VARCHAR",
        "prediction_id": "VARCHAR",
        "config_id": "VARCHAR",
        "as_of_date": "VARCHAR",
        "model_family": "VARCHAR",
        "model_build": "VARCHAR",
        "model_type": "VARCHAR",
        "mode": "VARCHAR",
        "feature_family_name": "VARCHAR",
        "feature_policy": "VARCHAR",
        "feature_set_id": "VARCHAR",
        "rank": "BIGINT",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load experiment result artifacts into DuckDB and export dashboard-ready files."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing model result artifacts such as predictions.parquet and metrics.parquet.",
    )
    parser.add_argument(
        "--dashboard-dir",
        default=None,
        help="Optional existing dashboard artifact directory to ingest/export from. Defaults to none.",
    )
    parser.add_argument(
        "--duckdb-path",
        default=None,
        help="Output DuckDB path. Defaults to <results-dir>/experiments.duckdb.",
    )
    parser.add_argument(
        "--dashboard-export-dir",
        default=None,
        help="Optional folder where dashboard-shaped Parquet/JSON files should be exported.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the DuckDB file if it already exists.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def load_parquet_table(con: duckdb.DuckDBPyConnection, table_name: str, path: Path) -> int:
    if not path.exists():
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM (SELECT NULL WHERE FALSE)")
        return 0
    if table_name in EMPTY_TABLE_SCHEMAS and pq.ParquetFile(path).metadata.num_columns == 0:
        columns = ", ".join(
            f"{column_name} {column_type}"
            for column_name, column_type in EMPTY_TABLE_SCHEMAS[table_name].items()
        )
        con.execute(f"CREATE OR REPLACE TABLE {table_name} ({columns})")
        return 0
    con.execute(
        f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet({sql_string(str(path))})"
    )
    return con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]


def load_json_table(con: duckdb.DuckDBPyConnection, table_name: str, payload: dict) -> int:
    df = pd.DataFrame(
        [
            {
                "artifact_name": table_name,
                "payload_json": json.dumps(payload, sort_keys=True, default=str),
            }
        ]
    )
    con.register("_json_payload", df)
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _json_payload")
    con.unregister("_json_payload")
    return len(df)


def relation_exists(con: duckdb.DuckDBPyConnection, relation_name: str) -> bool:
    return (
        con.execute(
            """
            SELECT count(*)
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [relation_name],
        ).fetchone()[0]
        > 0
    )


def create_experiment_runs(con: duckdb.DuckDBPyConnection, manifests: dict) -> int:
    experiment_manifest = manifests.get("experiment_manifest", {})
    batch_manifest = manifests.get("batch_manifest", {})
    champion_selection = manifests.get("champion_selection", {})
    experiment_id = (
        experiment_manifest.get("experiment_id")
        or batch_manifest.get("experiment_id")
        or champion_selection.get("experiment_id")
    )
    row = {
        "experiment_id": experiment_id,
        "pipeline_run_id": experiment_manifest.get("pipeline_run_id")
        or batch_manifest.get("pipeline_run_id")
        or champion_selection.get("pipeline_run_id"),
        "target": experiment_manifest.get("target") or champion_selection.get("target"),
        "horizon": experiment_manifest.get("horizon") or champion_selection.get("horizon"),
        "as_of_start": experiment_manifest.get("as_of_start") or batch_manifest.get("as_of_start"),
        "as_of_end": experiment_manifest.get("as_of_end") or batch_manifest.get("as_of_end"),
        "model_count": batch_manifest.get("model_count"),
        "prediction_count": batch_manifest.get("prediction_count"),
        "champion_model_config_id": champion_selection.get("model_config_id"),
        "champion_selection_score": champion_selection.get("selection_score"),
        "manifest_json": json.dumps(experiment_manifest, sort_keys=True, default=str),
        "batch_manifest_json": json.dumps(batch_manifest, sort_keys=True, default=str),
        "champion_json": json.dumps(champion_selection, sort_keys=True, default=str),
    }
    df = pd.DataFrame([row])
    con.register("_experiment_runs", df)
    con.execute("CREATE OR REPLACE TABLE experiment_runs AS SELECT * FROM _experiment_runs")
    con.unregister("_experiment_runs")
    return len(df)


def create_dashboard_views(con: duckdb.DuckDBPyConnection) -> None:
    if relation_exists(con, "source_dashboard_forecast_paths"):
        con.execute("CREATE OR REPLACE VIEW forecast_paths AS SELECT * FROM source_dashboard_forecast_paths")
    else:
        con.execute(
            """
            CREATE OR REPLACE VIEW forecast_paths AS
            SELECT
                config_id,
                model_config_id,
                as_of_date,
                target_date,
                model_family,
                model_build,
                model_type,
                mode,
                feature_family_name,
                feature_policy,
                feature_set_id,
                actual,
                prediction,
                baseline_prediction,
                seasonal_naive_prediction,
                model_refit,
                error,
                abs_error,
                evaluation_period,
                shock_period_flag
            FROM predictions
            """
        )

    if relation_exists(con, "source_dashboard_performance_over_time"):
        con.execute(
            "CREATE OR REPLACE VIEW performance_over_time AS SELECT * FROM source_dashboard_performance_over_time"
        )
    else:
        con.execute("CREATE OR REPLACE VIEW performance_over_time AS SELECT * FROM predictions")

    if relation_exists(con, "source_dashboard_model_leaderboard"):
        con.execute("CREATE OR REPLACE VIEW model_leaderboard AS SELECT * FROM source_dashboard_model_leaderboard")
    else:
        con.execute(
            """
            CREATE OR REPLACE VIEW model_leaderboard AS
            SELECT
                *
            FROM metrics
            WHERE evaluation_scope = 'overall'
            ORDER BY selection_score
            """
        )

    if relation_exists(con, "source_dashboard_champion_predictions"):
        con.execute(
            "CREATE OR REPLACE VIEW champion_predictions AS SELECT * FROM source_dashboard_champion_predictions"
        )
    else:
        con.execute(
            """
            CREATE OR REPLACE VIEW champion_predictions AS
            SELECT p.*
            FROM predictions p
            JOIN experiment_runs e
              ON p.model_config_id = e.champion_model_config_id
            """
        )

    if relation_exists(con, "source_dashboard_overview_top_models"):
        con.execute("CREATE OR REPLACE VIEW overview_top_models AS SELECT * FROM source_dashboard_overview_top_models")
    else:
        con.execute(
            """
            CREATE OR REPLACE VIEW overview_top_models AS
            SELECT *
            FROM model_leaderboard
            ORDER BY selection_score
            LIMIT 5
            """
        )

    if relation_exists(con, "source_dashboard_overview_prediction_paths"):
        con.execute(
            "CREATE OR REPLACE VIEW overview_prediction_paths AS SELECT * FROM source_dashboard_overview_prediction_paths"
        )
    else:
        con.execute(
            """
            CREATE OR REPLACE VIEW overview_prediction_paths AS
            SELECT p.*
            FROM predictions p
            WHERE p.model_config_id IN (SELECT model_config_id FROM overview_top_models)
            """
        )

    if relation_exists(con, "source_dashboard_feature_family_summary_dashboard"):
        con.execute(
            """
            CREATE OR REPLACE VIEW feature_family_summary_dashboard AS
            SELECT * FROM source_dashboard_feature_family_summary_dashboard
            """
        )
    else:
        con.execute(
            """
            CREATE OR REPLACE VIEW feature_family_summary_dashboard AS
            SELECT * FROM feature_family_summary
            """
        )

    if relation_exists(con, "source_dashboard_complexity_profile_dashboard"):
        con.execute(
            """
            CREATE OR REPLACE VIEW complexity_profile_dashboard AS
            SELECT * FROM source_dashboard_complexity_profile_dashboard
            """
        )
    elif relation_exists(con, "complexity_profile"):
        con.execute(
            """
            CREATE OR REPLACE VIEW complexity_profile_dashboard AS
            SELECT * FROM complexity_profile
            """
        )


def export_table_or_view(con: duckdb.DuckDBPyConnection, object_name: str, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT * FROM {object_name}) TO {sql_string(str(output_path))} (FORMAT PARQUET)")
    return con.execute(f"SELECT count(*) FROM {object_name}").fetchone()[0]


def copy_json_artifacts(results_dir: Path, dashboard_dir: Path | None, output_dir: Path) -> dict:
    counts = {}
    for artifact_name, filename in JSON_ARTIFACTS.items():
        source = results_dir / filename
        if not source.exists() and dashboard_dir is not None:
            source = dashboard_dir / filename
        payload = read_json(source)
        write_json(output_dir / filename, payload)
        counts[artifact_name] = 1 if payload else 0
    return counts


def build_mart(
    results_dir: Path,
    dashboard_dir: Path | None,
    duckdb_path: Path,
    dashboard_export_dir: Path | None,
    replace: bool,
) -> dict:
    if duckdb_path.exists() and replace:
        duckdb_path.unlink()
    if duckdb_path.exists() and not replace:
        raise FileExistsError(f"{duckdb_path} already exists. Pass --replace to rebuild it.")

    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(duckdb_path))
    counts = {}
    manifests = {}

    try:
        for table_name, filename in RESULT_TABLES.items():
            counts[table_name] = load_parquet_table(con, table_name, results_dir / filename)

        for artifact_name, filename in JSON_ARTIFACTS.items():
            payload = read_json(results_dir / filename)
            manifests[artifact_name] = payload
            counts[artifact_name] = load_json_table(con, artifact_name, payload)

        counts["experiment_runs"] = create_experiment_runs(con, manifests)

        if dashboard_dir is not None and dashboard_dir.exists():
            for table_name, filename in DASHBOARD_EXPORTS.items():
                source = dashboard_dir / filename
                if source.exists():
                    counts[f"source_dashboard_{table_name}"] = load_parquet_table(
                        con,
                        f"source_dashboard_{table_name}",
                        source,
                    )

        create_dashboard_views(con)

        export_counts = {}
        if dashboard_export_dir is not None:
            export_counts["forecast_paths"] = export_table_or_view(
                con,
                "forecast_paths",
                dashboard_export_dir / "forecast_paths.parquet",
            )
            export_counts["performance_over_time"] = export_table_or_view(
                con,
                "performance_over_time",
                dashboard_export_dir / "performance_over_time.parquet",
            )
            export_counts["model_leaderboard"] = export_table_or_view(
                con,
                "model_leaderboard",
                dashboard_export_dir / "model_leaderboard.parquet",
            )
            export_counts["feature_family_summary"] = export_table_or_view(
                con,
                "feature_family_summary_dashboard",
                dashboard_export_dir / "feature_family_summary.parquet",
            )
            export_counts["champion_predictions"] = export_table_or_view(
                con,
                "champion_predictions",
                dashboard_export_dir / "champion_predictions.parquet",
            )
            export_counts["overview_top_models"] = export_table_or_view(
                con,
                "overview_top_models",
                dashboard_export_dir / "overview_top_models.parquet",
            )
            export_counts["overview_prediction_paths"] = export_table_or_view(
                con,
                "overview_prediction_paths",
                dashboard_export_dir / "overview_prediction_paths.parquet",
            )
            if relation_exists(con, "complexity_profile_dashboard"):
                export_counts["complexity_profile"] = export_table_or_view(
                    con,
                    "complexity_profile_dashboard",
                    dashboard_export_dir / "complexity_profile.parquet",
                )
            json_counts = copy_json_artifacts(results_dir, dashboard_dir, dashboard_export_dir)
            export_counts.update({f"json_{key}": value for key, value in json_counts.items()})

        validation = {
            "duckdb_path": str(duckdb_path),
            "results_dir": str(results_dir),
            "dashboard_dir": str(dashboard_dir) if dashboard_dir else None,
            "dashboard_export_dir": str(dashboard_export_dir) if dashboard_export_dir else None,
            "table_counts": counts,
            "export_counts": export_counts,
        }
        if dashboard_export_dir is not None:
            write_json(dashboard_export_dir / "mart_validation.json", validation)
        return validation
    finally:
        con.close()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    dashboard_dir = Path(args.dashboard_dir) if args.dashboard_dir else None
    duckdb_path = Path(args.duckdb_path) if args.duckdb_path else results_dir / "experiments.duckdb"
    dashboard_export_dir = Path(args.dashboard_export_dir) if args.dashboard_export_dir else None

    validation = build_mart(
        results_dir=results_dir,
        dashboard_dir=dashboard_dir,
        duckdb_path=duckdb_path,
        dashboard_export_dir=dashboard_export_dir,
        replace=args.replace,
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
