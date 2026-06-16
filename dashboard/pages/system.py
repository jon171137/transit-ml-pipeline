import base64
from html import escape
from pathlib import Path

import streamlit as st

from constants import STEP_FUNCTION_SCREENSHOT_PATH, SYSTEM_ARCH_VIDEO_PATH, VIDEO_MIME_TYPES
from content import SYSTEM_ARCHITECTURE, SYSTEM_ARTIFACT_FLOW, SYSTEM_OVERVIEW, SYSTEM_REASONING
from formatting import format_int


def render_looping_system_video(path: Path, title: str, description: str) -> None:
    st.markdown('<div class="system-asset-block">', unsafe_allow_html=True)
    st.subheader(title)
    st.write(description)
    if not path.exists():
        st.info(f"Expected video asset not found: `{path}`")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    mime_type = VIDEO_MIME_TYPES.get(path.suffix.lower(), "video/mp4")
    encoded_video = base64.b64encode(path.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <div class="system-arch-video-frame">
            <video class="system-arch-video" width="1692" height="1852" autoplay muted loop playsinline preload="auto">
                <source src="data:{mime_type};base64,{encoded_video}" type="{mime_type}">
                Your browser does not support embedded video playback.
            </video>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_system_screenshot(path: Path, title: str, description: str) -> None:
    st.markdown('<div class="system-asset-block">', unsafe_allow_html=True)
    text_col, image_col = st.columns([1, 1])
    with text_col:
        st.subheader(title)
        st.write(description)
    with image_col:
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.info(f"Expected screenshot asset not found: `{path}`")
    st.markdown("</div>", unsafe_allow_html=True)


def render_system_artifact_flow(experiment_manifest: dict, flat_forecast_rows: int) -> None:
    st.markdown(SYSTEM_ARTIFACT_FLOW)

    bundle = experiment_manifest.get("public_dashboard_bundle", {})
    full_path_rows = bundle.get("full_path_rows", {}) if isinstance(bundle.get("full_path_rows"), dict) else {}
    artifact_cols = st.columns(4)
    artifact_cols[0].metric(
        "Full model index",
        format_int(bundle.get("source_configurations") or bundle.get("full_metadata_configurations")),
    )
    artifact_cols[1].metric("Curated flat configs", format_int(bundle.get("selected_configurations")))
    artifact_cols[2].metric("Full forecast rows", format_int(full_path_rows.get("forecast_paths")))
    artifact_cols[3].metric("Flat forecast rows", format_int(flat_forecast_rows))

    file_rows = [
        {
            "Layer": "Full model metadata",
            "Path / files": ["model_leaderboard_full.parquet", "complexity_profile_full.parquet"],
            "Dashboard role": "Filter and compare the full public experiment index without loading every prediction row.",
        },
        {
            "Layer": "Curated compatibility paths",
            "Path / files": ["forecast_paths.parquet", "performance_over_time.parquet"],
            "Dashboard role": "Fast initial overview/champion context and backward-compatible dashboard loading.",
        },
        {
            "Layer": "Partitioned full paths",
            "Path / files": ["forecast_paths_by_build/", "performance_over_time_by_build/"],
            "Dashboard role": "On-demand forecast and rolling-error rows after Model Explorer filters are applied.",
        },
        {
            "Layer": "Bundle manifests",
            "Path / files": [
                "experiment_manifest.json",
                "public_bundle_manifest.json",
                "path_partition_manifest.json",
            ],
            "Dashboard role": "Document source experiment IDs, curation rules, counts, and partition layout.",
        },
    ]
    rows_html = [
        """
        <div class="artifact-flow-row artifact-flow-head">
            <div class="artifact-flow-cell">Layer</div>
            <div class="artifact-flow-cell">Path / files</div>
            <div class="artifact-flow-cell">Dashboard role</div>
        </div>
        """
    ]
    for row in file_rows:
        path_html = ", ".join(f"<code>{escape(path)}</code>" for path in row["Path / files"])
        rows_html.append(
            f"""
            <div class="artifact-flow-row">
                <div class="artifact-flow-cell">{escape(row["Layer"])}</div>
                <div class="artifact-flow-cell">{path_html}</div>
                <div class="artifact-flow-cell">{escape(row["Dashboard role"])}</div>
            </div>
            """
        )
    st.markdown(
        f'<div class="artifact-flow-table">{"".join(rows_html)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "The public dashboard bundle is static, but it is not limited to a tiny leaderboard. "
        "Small metadata files describe the broader model universe; larger path-level data is "
        "split by model build so the app can read only the rows needed for the current view."
    )


def render_system_page(experiment_manifest: dict, flat_forecast_rows: int) -> None:
    system_cols = st.columns(2)
    system_cols[0].markdown(SYSTEM_ARCHITECTURE)
    system_cols[1].markdown(SYSTEM_REASONING)
    st.markdown(SYSTEM_OVERVIEW)
    render_looping_system_video(
        SYSTEM_ARCH_VIDEO_PATH,
        "End-to-End System Architecture",
        (
            "A short looping walkthrough of the pipeline shape: source ingestion, S3 storage layers, "
            "ECS and Step Functions processing, local extended experiments, the DuckDB mart, and the "
            "Streamlit dashboard artifact bundle."
        ),
    )
    render_system_screenshot(
        STEP_FUNCTION_SCREENSHOT_PATH,
        "AWS Step Functions Run",
        (
            "The AWS orchestration view for the streamlined pipeline, showing parallel normalization, "
            "integration, feature-table creation, manifest writing, and smoke-scale model training "
            "running through ECS Fargate tasks."
        ),
    )
    render_system_artifact_flow(experiment_manifest, flat_forecast_rows)
