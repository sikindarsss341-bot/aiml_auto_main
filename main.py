import json
import os
import queue
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import importlib
import altair as alt
import io
import joblib

from backend.main import (
    _build_pipeline_result,
    _infer_problem_type,
    _to_serializable,
    detect_default_target,
)

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}

MODEL_STATE: dict = {}

def execute_prediction(model, features: list[str], scaling_params: dict, problem_type: str, target_mapping: dict, payload: dict) -> dict:
    """Executes prediction using the exact established ML logic."""
    row = {}
    for name in features:
        val = float(payload.get(name, 0))
        if name in scaling_params:
            mean = scaling_params[name].get("mean", 0.0)
            scale = scaling_params[name].get("scale", 1.0)
            if scale != 0:
                val = (val - mean) / scale
        row[name] = val

    df = pd.DataFrame([row], columns=features)
    raw_pred = model.predict(df)[0]
    is_regression = str(problem_type).lower() == "regression"
    pred = float(raw_pred) if is_regression else int(raw_pred)

    probs = {}
    if hasattr(model, "predict_proba") and not is_regression:
        try:
            classes = [str(c) for c in model.classes_.tolist()]
            vals = model.predict_proba(df)[0].tolist()
            probs = {cls: float(v) for cls, v in zip(classes, vals, strict=False)}
        except Exception:
            pass

    mapping = target_mapping or {}
    pred_label = mapping.get(str(pred), str(pred))
    return {"prediction": pred, "prediction_label": pred_label, "probabilities": probs}


def compute_class_distribution(series: pd.Series, target_mapping: dict = None) -> list[dict]:
    """Computes class distribution counts and percentages, highlighting 0 and 1."""
    clean = series.dropna()
    if clean.empty:
        return []
    total = len(clean)
    counts = clean.value_counts()
    dist = []
    for val, count in counts.items():
        pct = (count / total) * 100.0
        display_label = str(val)
        if target_mapping:
            for k, v in target_mapping.items():
                if str(v) == str(val):
                    display_label = f"{k} ({val})"
                    break
        dist.append({
            "val": str(val),
            "label": display_label,
            "count": int(count),
            "pct": float(pct),
        })

    def sort_key(d):
        v = d["val"]
        if v in ["0", "0.0"]:
            return (0, 0)
        if v in ["1", "1.0"]:
            return (0, 1)
        return (1, v)

    dist.sort(key=sort_key)
    return dist


def render_leaderboard_bar_chart(comp_data: list[dict], prim_metric: str, prob_type: str, chart_key: str = "chart"):
    """Renders a model comparison bar chart with y-axis scaled from 50 to 100 for classification and full, unclipped model names."""
    if not comp_data:
        st.warning("No comparison data available.")
        return

    df_chart = pd.DataFrame(comp_data)
    if "Model" not in df_chart.columns or prim_metric not in df_chart.columns:
        return

    is_classification = str(prob_type).lower() == "classification" or prim_metric in [
        "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"
    ]

    try:
        max_val = float(df_chart[prim_metric].max())
        if is_classification and max_val <= 1.05:
            df_chart["Score_Pct"] = df_chart[prim_metric] * 100.0
            chart_metric_col = "Score_Pct"
            y_axis_title = f"{prim_metric} (%)"
            min_score = float(df_chart["Score_Pct"].min())
            y_min = 50.0 if min_score >= 50.0 else max(0.0, float(int(min_score // 10) * 10))
            y_max = 100.0
            y_ticks = list(range(int(y_min), 101, 10))
            fmt_str = ".1f"
        else:
            df_chart["Score_Pct"] = df_chart[prim_metric]
            chart_metric_col = "Score_Pct"
            y_axis_title = prim_metric
            y_min = float(df_chart[prim_metric].min() * 0.9)
            y_max = float(df_chart[prim_metric].max() * 1.1)
            y_ticks = None
            fmt_str = ".2f"

        df_chart = df_chart.sort_values(by=chart_metric_col, ascending=False).reset_index(drop=True)

        col_l, col_r = st.columns([2.5, 1.5])
        with col_r:
            chart_view_mode = st.radio(
                "Graph Orientation",
                ["Vertical", "Horizontal (Full Names)"],
                index=0,
                horizontal=True,
                key=f"chart_mode_{chart_key}",
            )

        if chart_view_mode == "Horizontal (Full Names)":
            df_h = df_chart.iloc[::-1].reset_index(drop=True)
            h_bars = (
                alt.Chart(df_h)
                .mark_bar(color="#6366f1", cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
                .encode(
                    y=alt.Y(
                        "Model:N",
                        sort=None,
                        title=None,
                        axis=alt.Axis(
                            labelFontSize=12,
                            labelColor="#e6edf3",
                            labelLimit=0,
                            labelFontWeight="bold",
                            labelPadding=10,
                        ),
                    ),
                    x=alt.X(
                        f"{chart_metric_col}:Q",
                        scale=alt.Scale(domain=[y_min, y_max]),
                        title=y_axis_title,
                        axis=alt.Axis(values=y_ticks, grid=True, gridColor="#30363d") if y_ticks else alt.Axis(grid=True),
                    ),
                    tooltip=[
                        alt.Tooltip("Model:N", title="Algorithm"),
                        alt.Tooltip(f"{chart_metric_col}:Q", title=y_axis_title, format=fmt_str),
                    ],
                )
            )

            h_text = h_bars.mark_text(
                align="left",
                baseline="middle",
                dx=6,
                color="#f0f6fc",
                fontSize=11,
                fontWeight="bold",
            ).encode(text=alt.Text(f"{chart_metric_col}:Q", format=fmt_str))

            chart = (h_bars + h_text).properties(height=max(340, len(df_chart) * 40))
            st.altair_chart(chart, use_container_width=True)

        else:
            y_axis_kwargs = {
                "title": y_axis_title,
                "scale": alt.Scale(domain=[y_min, y_max]),
            }
            if y_ticks:
                y_axis_kwargs["axis"] = alt.Axis(values=y_ticks, grid=True, gridColor="#30363d")

            v_bars = (
                alt.Chart(df_chart)
                .mark_bar(color="#6366f1", cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X(
                        "Model:N",
                        sort=None,
                        title=None,
                        axis=alt.Axis(
                            labelAngle=-45,
                            labelLimit=0,
                            labelOverlap=False,
                            labelFontSize=12,
                            labelColor="#e6edf3",
                            labelFontWeight="bold",
                            labelPadding=8,
                        ),
                    ),
                    y=alt.Y(f"{chart_metric_col}:Q", **y_axis_kwargs),
                    tooltip=[
                        alt.Tooltip("Model:N", title="Algorithm"),
                        alt.Tooltip(f"{chart_metric_col}:Q", title=y_axis_title, format=fmt_str),
                    ],
                )
            )

            v_text = v_bars.mark_text(
                align="center",
                baseline="bottom",
                dy=-6,
                color="#f0f6fc",
                fontSize=11,
                fontWeight="bold",
            ).encode(text=alt.Text(f"{chart_metric_col}:Q", format=fmt_str))

            chart = (v_bars + v_text).properties(height=450, padding={"bottom": 45})
            st.altair_chart(chart, use_container_width=True)

    except Exception:
        chart_df = df_chart.set_index("Model")[[prim_metric]]
        st.bar_chart(chart_df)


def get_api_app():
    """Factory creating the FastAPI backend API for api_server.py."""
    fastapi_mod = importlib.import_module("fastapi")
    FastAPI = fastapi_mod.FastAPI
    File = fastapi_mod.File
    Form = fastapi_mod.Form
    UploadFile = fastapi_mod.UploadFile
    JSONResponse = importlib.import_module("fastapi.responses").JSONResponse

    api = FastAPI(title="AutoML Studio", description="Backend API for AutoML Studio by Sikindar")

    @api.post("/api/columns")
    async def detect_columns(file: UploadFile = File(...)):
        suffix = Path(file.filename or "data.csv").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return JSONResponse(
                content={"detail": f"Unsupported file type '{suffix or 'none'}'. Supported formats: .csv, .xlsx, .xls, .json"},
                status_code=400,
            )
        raw = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".csv") as tmp:
            tmp.write(raw)
            path = tmp.name
        try:
            if suffix in {".csv", ""}:
                df = pd.read_csv(path, nrows=0)
            elif suffix in {".xlsx", ".xls"}:
                df = pd.read_excel(path, nrows=0)
            elif suffix == ".json":
                try:
                    df = pd.read_json(path, nrows=1)
                except Exception:
                    df = pd.read_json(path)
            columns = [str(c).strip() for c in df.columns]
            if not columns:
                return JSONResponse(content={"detail": "No columns found in dataset"}, status_code=400)
            return JSONResponse(content=columns)
        except Exception as e:
            return JSONResponse(content={"detail": f"Error reading columns: {str(e)}"}, status_code=400)
        finally:
            if os.path.exists(path):
                os.remove(path)

    @api.post("/api/upload")
    async def upload_file(file: UploadFile = File(...)):
        suffix = Path(file.filename or "data.csv").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return JSONResponse(
                {"detail": f"Unsupported file type '{suffix or 'none'}'. Supported formats: .csv, .xlsx, .xls, .json"},
                status_code=400,
            )
        file_id = str(uuid.uuid4())
        dest = UPLOAD_DIR / f"{file_id}{suffix}"
        raw = await file.read()
        with open(dest, "wb") as f:
            f.write(raw)
        try:
            if suffix in {".csv", ""}:
                df = pd.read_csv(dest, nrows=0)
            elif suffix in {".xlsx", ".xls"}:
                df = pd.read_excel(dest, nrows=0)
            elif suffix == ".json":
                try:
                    df = pd.read_json(dest, nrows=1)
                except Exception:
                    df = pd.read_json(dest)
            columns = [str(c).strip() for c in df.columns]
            if not columns:
                if os.path.exists(dest):
                    os.remove(dest)
                return JSONResponse({"detail": "No columns detected in dataset"}, status_code=400)
        except Exception as e:
            if os.path.exists(dest):
                os.remove(dest)
            return JSONResponse({"detail": f"Failed to parse dataset: {str(e)}"}, status_code=400)
        return JSONResponse({"file_id": file_id, "filename": file.filename or "data.csv", "columns": columns})

    @api.get("/api/train-stream")
    async def train_stream(file_id: str, target: str | None = None):
        from fastapi.responses import StreamingResponse

        found = None
        for p in UPLOAD_DIR.iterdir():
            if p.stem == file_id:
                found = p
                break
        if found is None:
            return JSONResponse({"detail": "Uploaded file not found"}, status_code=404)

        q = queue.Queue()

        def progress_cb(progress, message):
            q.put({"type": "progress", "progress": int(progress), "message": message})

        def run_pipeline():
            try:
                result = _build_pipeline_result(
                    job_id=str(uuid.uuid4()),
                    file_path=str(found),
                    target_column=target or "",
                    progress_callback=progress_cb,
                )
                best_model_object = result.get("best_model_object")
                serializable_result = _to_serializable(dict(result))
                if isinstance(serializable_result, dict):
                    serializable_result.pop("best_model_object", None)
                MODEL_STATE.clear()
                if isinstance(serializable_result, dict):
                    MODEL_STATE.update(serializable_result)
                if best_model_object is not None:
                    MODEL_STATE["best_model_object"] = best_model_object
                q.put({"type": "result", "result": serializable_result})
            except Exception as exc:
                q.put({"type": "error", "message": str(exc)})

        thread = threading.Thread(target=run_pipeline, daemon=True)
        thread.start()

        def event_stream():
            while True:
                item = q.get()
                if item["type"] == "progress":
                    payload = json.dumps({"progress": item["progress"], "message": item["message"]})
                    yield f"event: progress\ndata: {payload}\n\n"
                elif item["type"] == "result":
                    yield f"event: result\ndata: {json.dumps(item['result'])}\n\n"
                    break
                elif item["type"] == "error":
                    yield f"event: error\ndata: {json.dumps({'message': item['message']})}\n\n"
                    break

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @api.post("/api/train")
    async def train(file: UploadFile = File(...), target_column: str = Form(None)):
        raw = await file.read()
        suffix = Path(file.filename or "data.csv").suffix.lower() or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            path = tmp.name
        try:
            try:
                df = pd.read_csv(path) if suffix in {".csv", ""} else pd.read_excel(path)
                cols = list(df.columns)
            except Exception:
                cols = []
            if not target_column or (cols and target_column not in cols):
                detected = detect_default_target(cols)
                if detected:
                    target_column = detected

            result = _build_pipeline_result(
                job_id=str(uuid.uuid4()),
                file_path=path,
                target_column=target_column,
                progress_callback=None,
            )
            summary = result.get("dashboard_summary", {})
            MODEL_STATE.clear()
            MODEL_STATE.update(result)
            return {
                "summary": {
                    "problem_type": summary.get("problem_type", result.get("problem_type", "Classification")),
                    "rows": summary.get("rows", result.get("dataset", {}).get("rows", 0)),
                    "columns": summary.get("columns", result.get("dataset", {}).get("columns", 0)),
                    "missing_values": summary.get("missing_values", result.get("dataset", {}).get("missing_values_total", 0)),
                    "best_model": summary.get("best_model", result.get("best_model_name", "—")),
                    "best_metric": summary.get("best_metric_value", "—"),
                },
                "feature_names": result.get("selected_feature_names") or result.get("feature_names", []),
                "feature_schema": result.get("feature_schema", []),
                "detected_target": target_column,
            }
        finally:
            if os.path.exists(path):
                os.remove(path)

    @api.post("/api/predict")
    async def predict(payload: dict):
        model = MODEL_STATE.get("best_model_object")
        features = MODEL_STATE.get("selected_feature_names") or MODEL_STATE.get("feature_names") or []
        if model is None or not features:
            return JSONResponse({"detail": "Train a model first."}, status_code=400)

        scaling_params = MODEL_STATE.get("scaling_params") or {}
        problem_type = str(MODEL_STATE.get("problem_type", ""))
        target_mapping = MODEL_STATE.get("target_label_mapping") or {}

        res = execute_prediction(model, features, scaling_params, problem_type, target_mapping, payload)
        return res

    return api


_api_instance = None


def __getattr__(name: str):
    global _api_instance
    if name == "app":
        if _api_instance is None:
            _api_instance = get_api_app()
        return _api_instance
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# -----------------------------------------------------------------------------
# Streamlit Frontend Application
# -----------------------------------------------------------------------------
def render_streamlit_app():
    st.set_page_config(
        page_title="AutoML Studio",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling for high-end dark dashboard feel
    st.markdown(
        """
        <style>
            /* Main container polish */
            .main .block-container {
                padding-top: 1.5rem;
                padding-bottom: 3rem;
                max-width: 1250px;
            }
            /* Metric cards styling */
            [data-testid="stMetric"] {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 10px;
                padding: 14px 18px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            }
            [data-testid="stMetricLabel"] p {
                font-size: 0.85rem !important;
                color: #8b949e !important;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            [data-testid="stMetricValue"] div {
                color: #f0f6fc !important;
                font-weight: 700;
            }
            /* Custom badges and cards */
            .custom-card {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 10px;
                padding: 16px 20px;
                margin-bottom: 16px;
            }
            .author-badge {
                display: inline-block;
                background: rgba(105, 87, 245, 0.15);
                border: 1px solid rgba(105, 87, 245, 0.4);
                color: #9d8eff;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 600;
            }
            /* Table formatting */
            .stDataFrame {
                border-radius: 8px;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # State initialization
    if "pipeline_result" not in st.session_state:
        st.session_state["pipeline_result"] = None
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = None
    if "active_file_path" not in st.session_state:
        st.session_state["active_file_path"] = None
    if "active_filename" not in st.session_state:
        st.session_state["active_filename"] = None

    # -------------------------------------------------------------------------
    # Sidebar: Data Source & Pipeline Execution
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.title("⚡ AutoML Studio")
        st.markdown(
            '<div class="author-badge">Engineered by Sikindar</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        st.subheader("1. Dataset Source")

        # Available sample datasets
        sample_datasets = {}
        titanic_csv = ROOT_DIR / "titanic.csv"
        if titanic_csv.exists():
            sample_datasets["Titanic Dataset (Classification - 891 rows)"] = titanic_csv
        for p in sorted(UPLOAD_DIR.glob("*.csv")):
            if p.name != "titanic.csv":
                sample_datasets[f"Upload: {p.name}"] = p

        has_samples = len(sample_datasets) > 0
        input_modes = ["Upload File", "Use Sample Dataset"] if has_samples else ["Upload File"]

        source_mode = st.radio(
            "Select Input Method",
            input_modes,
            index=0,
            horizontal=True,
        )

        # Track mode changes to reset state cleanly
        if "prev_source_mode" not in st.session_state:
            st.session_state["prev_source_mode"] = source_mode
        elif st.session_state["prev_source_mode"] != source_mode:
            st.session_state["prev_source_mode"] = source_mode
            st.session_state["active_file_path"] = None
            st.session_state["active_filename"] = None
            st.session_state["raw_df"] = None
            st.session_state["pipeline_result"] = None

        uploaded_file = None
        if source_mode == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload Dataset",
                type=["csv", "xlsx", "xls", "json"],
                help="Supports CSV, Excel (.xlsx, .xls), or JSON tabular files (up to 10MB)",
            )
            if uploaded_file is not None:
                # Save file if new
                if (
                    st.session_state["active_filename"] != uploaded_file.name
                    or st.session_state["active_file_path"] is None
                ):
                    file_suffix = Path(uploaded_file.name).suffix.lower()
                    save_path = UPLOAD_DIR / f"{uuid.uuid4()}{file_suffix}"
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state["active_file_path"] = str(save_path)
                    st.session_state["active_filename"] = uploaded_file.name
                    # Load dataframe
                    try:
                        if file_suffix in [".csv", ""]:
                            df = pd.read_csv(save_path)
                        elif file_suffix in [".xlsx", ".xls"]:
                            df = pd.read_excel(save_path)
                        elif file_suffix == ".json":
                            df = pd.read_json(save_path)
                        st.session_state["raw_df"] = df
                        st.session_state["pipeline_result"] = None
                    except Exception as exc:
                        st.error(f"Error reading dataset: {exc}")
            elif has_samples and st.session_state["raw_df"] is None:
                st.caption("Tip: Switch to **Use Sample Dataset** above to test with the Titanic dataset.")
        elif has_samples:
            selected_sample_label = st.selectbox(
                "Choose sample dataset",
                options=list(sample_datasets.keys()),
                help="Preloaded datasets for instant benchmarking",
            )
            if selected_sample_label:
                sample_path = sample_datasets[selected_sample_label]
                if st.session_state["active_file_path"] != str(sample_path):
                    st.session_state["active_file_path"] = str(sample_path)
                    st.session_state["active_filename"] = sample_path.name
                    try:
                        st.session_state["raw_df"] = pd.read_csv(sample_path)
                        st.session_state["pipeline_result"] = None
                    except Exception as exc:
                        st.error(f"Error reading sample dataset: {exc}")

        # Target Column selection
        target_col = None
        if st.session_state["raw_df"] is not None:
            df = st.session_state["raw_df"]
            cols = list(df.columns)
            detected_target = detect_default_target(cols)
            target_idx = cols.index(detected_target) if detected_target in cols else (len(cols) - 1 if cols else 0)

            st.subheader("2. Target & Configuration")
            target_col = st.selectbox(
                "Target Variable",
                options=cols,
                index=target_idx,
                help="Column to predict",
            )

            # Detect problem type preview
            inferred_type = _infer_problem_type(df[target_col]) if target_col in df.columns else "classification"
            st.info(f"Target: **{target_col}**  \nInferred Type: **{inferred_type.title()}**")

            # Show class distribution if classification
            if inferred_type == "classification" and target_col in df.columns:
                sidebar_dist = compute_class_distribution(df[target_col])
                if sidebar_dist:
                    st.caption("**Target Class Balance:**")
                    cards_html = ""
                    for d in sidebar_dist[:2]:
                        cards_html += f"""
                        <div style="flex: 1; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px 6px; text-align: center;">
                            <div style="color: #8b949e; font-size: 0.72rem; font-weight: 600; text-transform: uppercase;">Class {d['val']}</div>
                            <div style="color: #58a6ff; font-size: 1.3rem; font-weight: 800; margin: 3px 0;">{d['pct']:.1f}%</div>
                            <div style="color: #8b949e; font-size: 0.7rem;">{d['count']:,} rows</div>
                        </div>
                        """
                    st.markdown(
                        f'<div style="display: flex; gap: 8px; margin-bottom: 12px;">{cards_html}</div>',
                        unsafe_allow_html=True,
                    )

            with st.expander("Advanced Settings"):
                skip_svm = st.checkbox("Skip SVM/SVR for large datasets", value=len(df) > 10000)
                selected_model_option = st.selectbox(
                    "Model Filter",
                    [
                        "All Models (AutoML)",
                        "Logistic Regression",
                        "Decision Tree",
                        "Random Forest",
                        "KNN",
                        "Gradient Boosting",
                        "Naive Bayes",
                        "XGBoost",
                        "LightGBM",
                        "SVM",
                    ],
                    index=0,
                )
                selected_model_name = None if selected_model_option == "All Models (AutoML)" else selected_model_option

            st.markdown("---")
            run_btn = st.button("🚀 Run AutoML Pipeline", type="primary", use_container_width=True)

            if run_btn:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def streamlit_progress_callback(percent: int, message: str):
                    progress_bar.progress(min(max(int(percent), 0), 100))
                    status_text.text(f"[{percent}%] {message}...")

                try:
                    with st.spinner("Executing AutoML Pipeline..."):
                        result = _build_pipeline_result(
                            job_id=str(uuid.uuid4()),
                            file_path=st.session_state["active_file_path"],
                            target_column=target_col,
                            progress_callback=streamlit_progress_callback,
                            skip_svm_by_size=skip_svm,
                            selected_model_name=selected_model_name,
                        )
                        st.session_state["pipeline_result"] = result
                        MODEL_STATE.clear()
                        MODEL_STATE.update(result)
                        status_text.text("Pipeline completed successfully!")
                        st.success("AutoML Pipeline finished!")
                except Exception as exc:
                    status_text.empty()
                    st.error(f"Pipeline Execution Failed: {str(exc)}")

        st.markdown("---")
        st.caption("**AutoML Studio**  \nArchitecture: Streamlit UI + Scikit-Learn Engine  \nDeveloped by **Sikindar**")

    # -------------------------------------------------------------------------
    # Main Dashboard Tabs
    # -------------------------------------------------------------------------
    result = st.session_state["pipeline_result"]
    raw_df = st.session_state["raw_df"]

    tab_dash, tab_data, tab_prep, tab_train, tab_comp, tab_eval, tab_tune, tab_pred = st.tabs(
        [
            "📊 Dashboard",
            "📁 Dataset",
            "⚙️ Preprocessing",
            "🤖 Model Training",
            "📈 Model Comparison",
            "🎯 Evaluation",
            "⚡ Hyperparameter Tuning",
            "🔮 Prediction",
        ]
    )

    # -------------------------------------------------------------------------
    # TAB 1: DASHBOARD
    # -------------------------------------------------------------------------
    with tab_dash:
        st.header("Executive Dashboard")
        st.caption("Comprehensive AutoML performance overview and dataset health metrics.")

        if result is None:
            # Landing / Initial guidance state
            st.info("👋 Welcome to AutoML Studio! Upload a dataset or choose a sample in the sidebar, select your target column, and click **'Run AutoML Pipeline'**.")

            col1, col2, col3 = st.columns(3)
            with col1:
                with st.container(border=True):
                    st.subheader("📁 Automated Preprocessing")
                    st.caption("Cleans columns, imputes missing values, applies label & one-hot encoding, scales features, and filters redundant predictors.")
            with col2:
                with st.container(border=True):
                    st.subheader("🤖 Multi-Model Tournament")
                    st.caption("Trains 9+ algorithms in parallel across Linear, Trees, Ensembles, XGBoost, LightGBM, and SVM architectures.")
            with col3:
                with st.container(border=True):
                    st.subheader("🔮 Real-time Inference")
                    st.caption("Performs instant prediction with automated feature scaling and class confidence probability breakdown.")

            if raw_df is not None:
                st.markdown("### Loaded Dataset Quick Look")
                st.dataframe(raw_df.head(10), use_container_width=True)
        else:
            summary = result.get("dashboard_summary", {})
            prob_type = result.get("problem_type", "classification").title()
            prim_metric = result.get("primary_metric", "Accuracy")
            best_score = summary.get("best_metric_value")
            if best_score is not None:
                best_score_str = f"{best_score * 100:.2f}%" if prob_type.lower() == "classification" else f"{best_score:.4f}"
            else:
                best_score_str = "—"

            # Top KPI Metrics Row
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            kpi1.metric("Problem Type", prob_type)
            kpi2.metric("Dataset Rows", f"{summary.get('rows', 0):,}")
            kpi3.metric("Dataset Columns", summary.get("columns", 0))
            kpi4.metric("Best Model", summary.get("best_model", "—"))
            kpi5.metric(f"Top {prim_metric}", best_score_str)

            st.markdown("---")

            # Target Class Balance breakdown for classification datasets
            target_col_name = result.get("target_column")
            if prob_type.lower() == "classification" and raw_df is not None and target_col_name in raw_df.columns:
                clf_dist = compute_class_distribution(raw_df[target_col_name], result.get("target_label_mapping"))
                if clf_dist:
                    with st.container(border=True):
                        st.subheader(f"📊 Target Class Distribution ({target_col_name})")
                        cd_cols = st.columns(min(len(clf_dist), 4))
                        for idx, cd in enumerate(clf_dist[:4]):
                            with cd_cols[idx]:
                                st.metric(
                                    label=f"Class {cd['label']}",
                                    value=f"{cd['pct']:.1f}%",
                                    delta=f"{cd['count']:,} samples",
                                    delta_color="off",
                                )
                                st.progress(min(max(float(cd['pct'] / 100.0), 0.0), 1.0))

            # Top 3 Podium
            top3 = result.get("top3", [])
            st.subheader("🏆 Top Performing Models")
            if top3:
                cols = st.columns(min(len(top3), 3))
                medals = ["🥇 1st Place (Winner)", "🥈 2nd Place", "🥉 3rd Place"]
                for i, row in enumerate(top3[:3]):
                    with cols[i]:
                        with st.container(border=True):
                            st.caption(medals[i])
                            st.subheader(row.get("Model", "Unknown"))
                            metric_val = (
                                f"{float(row[prim_metric])*100:.2f}%"
                                if prob_type.lower() == "classification" and prim_metric in row
                                else str(row.get(prim_metric, "—"))
                            )
                            st.markdown(f"**{prim_metric}:** `{metric_val}`")
                            if "F1 Score" in row and prob_type.lower() == "classification":
                                st.markdown(f"**F1 Score:** `{float(row.get('F1 Score', 0))*100:.2f}%`")
                            elif "RMSE" in row and prob_type.lower() == "regression":
                                st.markdown(f"**RMSE:** `{float(row.get('RMSE', 0)):.4f}`")

            # Benchmark Chart
            comp_data = result.get("comparison", [])
            if comp_data:
                st.subheader(f"Leaderboard Comparison ({prim_metric})")
                render_leaderboard_bar_chart(comp_data, prim_metric, prob_type, chart_key="dash")

    # -------------------------------------------------------------------------
    # TAB 2: DATASET
    # -------------------------------------------------------------------------
    with tab_data:
        st.header("Dataset Inspection & Profiling")
        st.caption("Exploratory data analysis, dimensions, statistics, and column data distributions.")

        if raw_df is None:
            st.info("No dataset is currently loaded. Please upload a file via the sidebar.")
        else:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Total Rows", f"{len(raw_df):,}")
            col_m2.metric("Total Columns", len(raw_df.columns))
            missing_count = int(raw_df.isna().sum().sum())
            col_m3.metric("Missing Values", f"{missing_count:,}")
            dup_count = int(raw_df.duplicated().sum())
            col_m4.metric("Duplicate Rows", f"{dup_count:,}")

            # Target Class Distribution if classification dataset
            active_target = target_col or detect_default_target(list(raw_df.columns))
            if active_target and active_target in raw_df.columns:
                target_ser = raw_df[active_target]
                if _infer_problem_type(target_ser) == "classification":
                    t_dist = compute_class_distribution(target_ser)
                    if t_dist:
                        with st.container(border=True):
                            st.subheader(f"📊 Target Class Distribution ({active_target})")
                            cd_cols = st.columns(min(len(t_dist), 4))
                            for idx, cd in enumerate(t_dist[:4]):
                                with cd_cols[idx]:
                                    st.metric(
                                        label=f"Class {cd['val']}",
                                        value=f"{cd['pct']:.1f}%",
                                        delta=f"{cd['count']:,} samples",
                                        delta_color="off",
                                    )
                                    st.progress(min(max(float(cd['pct'] / 100.0), 0.0), 1.0))

            st.markdown("### Raw Data Preview")
            st.dataframe(raw_df.head(100), use_container_width=True)

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("### Column Data Types & Non-Null Counts")
                types_df = pd.DataFrame(
                    {
                        "Column": raw_df.columns,
                        "Type": [str(t) for t in raw_df.dtypes],
                        "Non-Null Count": raw_df.notna().sum().values,
                        "Missing Count": raw_df.isna().sum().values,
                        "Missing %": [f"{(c / len(raw_df)) * 100:.1f}%" for c in raw_df.isna().sum().values],
                    }
                )
                st.dataframe(types_df, use_container_width=True, height=350)

            with col_right:
                st.markdown("### Missing Values Distribution")
                missing_cols = raw_df.isna().sum()
                missing_cols = missing_cols[missing_cols > 0]
                if not missing_cols.empty:
                    st.bar_chart(missing_cols)
                else:
                    st.success("🎉 No missing values detected in any column of the dataset!")

            st.markdown("### Summary Statistics")
            st.dataframe(raw_df.describe(include="all").T, use_container_width=True)

            # Correlation Matrix if numeric columns exist
            numeric_cols = raw_df.select_dtypes(include=np.number)
            if numeric_cols.shape[1] > 1:
                with st.expander("Numeric Correlation Matrix"):
                    corr_df = numeric_cols.corr()
                    try:
                        st.dataframe(corr_df.style.background_gradient(cmap="coolwarm", axis=None), use_container_width=True)
                    except Exception:
                        st.dataframe(corr_df, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: PREPROCESSING
    # -------------------------------------------------------------------------
    with tab_prep:
        st.header("Preprocessing Pipeline Audit")
        st.caption("Verification of data transformations, cleaning rules, encoding, and scaling.")

        if result is None:
            st.info("Execute the AutoML pipeline in the sidebar to review preprocessing steps.")
        else:
            prep = result.get("preprocessing", {})
            ds = result.get("dataset", {})

            col1, col2 = st.columns(2)

            with col1:
                with st.expander("1. Data Cleaning & Dropped Columns", expanded=True):
                    st.write(f"**Duplicate Rows Removed:** {ds.get('duplicate_rows_removed', 0)}")
                    dropped = ds.get("dropped_columns", [])
                    reasons = ds.get("dropped_column_reasons", {})
                    if dropped:
                        st.write("**Dropped Columns:**")
                        drop_data = [{"Column": col, "Reason": reasons.get(col, "Irrelevant / High Cardinality")} for col in dropped]
                        st.dataframe(pd.DataFrame(drop_data), use_container_width=True)
                    else:
                        st.write("No columns were dropped during initial cleaning.")

                with col2:
                    with st.expander("2. Missing Value Imputation", expanded=True):
                        missing_rep = prep.get("missing_value_report", {})
                        if missing_rep and any(missing_rep.values()):
                            st.json(missing_rep)
                        else:
                            st.write("No missing values required imputation.")

            col3, col4 = st.columns(2)

            with col3:
                with st.expander("3. Categorical Feature Encoding", expanded=True):
                    enc_rep = prep.get("encoding_report", {})
                    if enc_rep and any(enc_rep.values()):
                        st.json(enc_rep)
                    else:
                        st.write("No categorical encoding transformations were needed.")

            with col4:
                with st.expander("4. Feature Selection & Filtering", expanded=True):
                    feat_rep = prep.get("feature_report", {})
                    selected_feats = result.get("selected_feature_names", [])
                    st.write(f"**Final Selected Features ({len(selected_feats)}):**")
                    st.write(", ".join([f"`{f}`" for f in selected_feats]))
                    if feat_rep and any(feat_rep.values()):
                        st.json(feat_rep)

            col5, col6 = st.columns(2)

            with col5:
                with st.expander("5. Feature Scaling (StandardScaler)", expanded=True):
                    scaling_rep = prep.get("scaling_report", {})
                    scaling_params = result.get("scaling_params", {})
                    if scaling_params:
                        df_scale = pd.DataFrame(scaling_params).T
                        st.dataframe(df_scale, use_container_width=True)
                    elif scaling_rep:
                        st.json(scaling_rep)
                    else:
                        st.write("No numerical feature scaling applied.")

            with col6:
                with st.expander("6. Train / Test Split", expanded=True):
                    prob = result.get("problem_type", "classification")
                    st.write(f"**Split Ratio:** 80% Train, 20% Test")
                    st.write(f"**Stratified Split Applied:** {'Yes (Classification)' if prob == 'classification' else 'No (Regression)'}")
                    warnings = prep.get("warnings", [])
                    if warnings:
                        st.warning("Warnings: " + "; ".join(warnings))
                    else:
                        st.success("Train/Test partition executed cleanly with zero warnings.")

    # -------------------------------------------------------------------------
    # TAB 4: MODEL TRAINING
    # -------------------------------------------------------------------------
    with tab_train:
        st.header("Trained Models & Feature Importance")
        st.caption("Trained model repository, architectural configurations, and predictive drivers.")

        if result is None:
            st.info("Execute the AutoML pipeline in the sidebar to review trained models.")
        else:
            comp_list = result.get("comparison", [])
            trained_names = [row["Model"] for row in comp_list if "Model" in row]

            st.subheader(f"Trained Algorithms ({len(trained_names)})")
            st.write(", ".join([f"`{name}`" for name in trained_names]))

            st.markdown("---")
            st.subheader("Feature Importance Analysis")

            importance_dict = result.get("feature_importance", {})
            corr_pairs = result.get("dataset", {}).get("correlation_pairs", [])

            available_sources = list(importance_dict.keys())
            if corr_pairs:
                available_sources.append("Dataset Correlation Pairs")

            if available_sources:
                selected_source = st.selectbox("Inspect Feature Drivers For:", available_sources)

                if selected_source == "Dataset Correlation Pairs":
                    df_corr = pd.DataFrame(corr_pairs)
                    st.dataframe(df_corr, use_container_width=True)
                else:
                    items = importance_dict.get(selected_source, [])
                    if items:
                        df_imp = pd.DataFrame(items)
                        col_c1, col_c2 = st.columns([1.4, 1.0])
                        with col_c1:
                            chart_df = df_imp.set_index("feature")[["importance"]]
                            st.bar_chart(chart_df)
                        with col_c2:
                            df_imp["importance_%"] = df_imp["importance"].apply(lambda v: f"{v * 100:.2f}%")
                            st.dataframe(df_imp, use_container_width=True)
                    else:
                        st.info(f"Feature importance weights not available for `{selected_source}`.")
            else:
                st.info("No feature importance data available.")

    # -------------------------------------------------------------------------
    # TAB 5: MODEL COMPARISON
    # -------------------------------------------------------------------------
    with tab_comp:
        st.header("Model Benchmark Leaderboard")
        st.caption("Direct side-by-side metric comparison across all evaluated models.")

        if result is None:
            st.info("Run the AutoML pipeline to populate the model comparison leaderboard.")
        else:
            comp_data = result.get("comparison", [])
            if comp_data:
                df_comp = pd.DataFrame(comp_data)
                prim_metric = result.get("primary_metric", "Accuracy")
                prob_type = result.get("problem_type", "classification").lower()

                # Highlight top model
                best_model_name = result.get("best_model_name", "")
                st.success(f"🏆 Top Ranked Model: **{best_model_name}**")

                # Format percentages for classification
                display_df = df_comp.copy()
                if prob_type == "classification":
                    for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
                        if metric in display_df.columns:
                            display_df[metric] = display_df[metric].apply(
                                lambda v: f"{float(v) * 100:.2f}%" if pd.notna(v) and isinstance(v, (int, float, np.number)) else v
                            )
                else:
                    for metric in ["R2", "RMSE", "MAE", "MSE"]:
                        if metric in display_df.columns:
                            display_df[metric] = display_df[metric].apply(
                                lambda v: f"{float(v):.4f}" if pd.notna(v) and isinstance(v, (int, float, np.number)) else v
                            )

                st.dataframe(display_df, use_container_width=True)

                st.subheader(f"Comparison Chart ({prim_metric})")
                render_leaderboard_bar_chart(comp_data, prim_metric, prob_type, chart_key="comp")
            else:
                st.warning("No comparison data available.")

    # -------------------------------------------------------------------------
    # TAB 6: EVALUATION
    # -------------------------------------------------------------------------
    with tab_eval:
        st.header("In-Depth Model Evaluation")
        st.caption("Detailed diagnostics, confusion matrices, and classification/regression reports.")

        if result is None:
            st.info("Run the AutoML pipeline to inspect model evaluation reports.")
        else:
            eval_report = result.get("evaluation", {})
            prob_type = result.get("problem_type", "classification").lower()

            if eval_report:
                model_choices = list(eval_report.keys())
                best_name = result.get("best_model_name", model_choices[0])
                default_idx = model_choices.index(best_name) if best_name in model_choices else 0

                selected_eval_model = st.selectbox("Select Model to Evaluate:", model_choices, index=default_idx)
                metrics = eval_report[selected_eval_model]

                # Metric Cards
                if prob_type == "classification":
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Accuracy", f"{metrics.get('Accuracy', 0) * 100:.2f}%")
                    m2.metric("Precision", f"{metrics.get('Precision', 0) * 100:.2f}%")
                    m3.metric("Recall", f"{metrics.get('Recall', 0) * 100:.2f}%")
                    m4.metric("F1 Score", f"{metrics.get('F1 Score', 0) * 100:.2f}%")
                    roc = metrics.get("ROC-AUC")
                    m5.metric("ROC-AUC", f"{roc * 100:.2f}%" if roc is not None else "N/A")

                    st.markdown("---")
                    c_left, c_right = st.columns(2)

                    with c_left:
                        st.subheader("Confusion Matrix")
                        cm = metrics.get("Confusion Matrix")
                        if cm is not None:
                            cm_arr = np.array(cm)
                            st.dataframe(pd.DataFrame(cm_arr), use_container_width=True)
                        else:
                            st.write("Confusion Matrix not available.")

                    with c_right:
                        st.subheader("Classification Report")
                        cr = metrics.get("Classification Report")
                        if cr:
                            st.code(str(cr), language="text")
                        else:
                            st.write("Classification report not available.")
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("R2 Score", f"{metrics.get('R2', 0):.4f}")
                    m2.metric("RMSE", f"{metrics.get('RMSE', 0):.4f}")
                    m3.metric("MAE", f"{metrics.get('MAE', 0):.4f}")
                    m4.metric("MSE", f"{metrics.get('MSE', 0):.4f}")
            else:
                st.warning("No evaluation metrics found.")

    # -------------------------------------------------------------------------
    # TAB 7: HYPERPARAMETER TUNING
    # -------------------------------------------------------------------------
    with tab_tune:
        st.header("Hyperparameter Tuning Summary")
        st.caption("Screening and hyperparameter optimization results for top candidate models.")

        if result is None:
            st.info("Run the AutoML pipeline to view hyperparameter tuning logs.")
        else:
            tuning_summary = result.get("tuning_summary", {})
            if tuning_summary:
                tuned_items = {k: v for k, v in tuning_summary.items() if v and v.get("was_tuned")}

                if tuned_items:
                    st.success(f"Optimized {len(tuned_items)} candidate model(s) via Cross-Validation Search.")

                    for model_name, info in tuned_items.items():
                        with st.expander(f"⚙️ {model_name} (Tuning Details)", expanded=True):
                            col_t1, col_t2 = st.columns(2)
                            b_cv = info.get("baseline_cv_score")
                            t_cv = info.get("tuned_cv_score")
                            b_str = f"{float(b_cv):.4f}" if b_cv is not None else "—"
                            t_str = f"{float(t_cv):.4f}" if t_cv is not None else "—"

                            col_t1.metric("Baseline CV Score", b_str)
                            col_t2.metric("Tuned CV Score", t_str)

                            best_params = info.get("best_params")
                            st.markdown("**Best Hyperparameters Found:**")
                            if best_params:
                                st.json(best_params)
                            else:
                                st.write("No distinct parameters adjusted.")
                else:
                    st.info("Tuning search executed, but baseline configurations already achieved superior cross-validation scores.")
            else:
                st.info("No hyperparameter tuning was triggered.")

    # -------------------------------------------------------------------------
    # TAB 8: PREDICTION
    # -------------------------------------------------------------------------
    with tab_pred:
        st.header("Interactive Real-Time Prediction")
        st.caption("Input custom feature values to generate predictions using the current top model.")

        if result is None or result.get("best_model_object") is None:
            st.info("Train models first by clicking 'Run AutoML Pipeline' in the sidebar.")
        else:
            model = result.get("best_model_object")
            best_model_name = result.get("best_model_name", "Best Model")
            target_col = result.get("target_column", "Target")
            schema = result.get("feature_schema", [])
            features = result.get("selected_feature_names") or result.get("feature_names", [])
            scaling_params = result.get("scaling_params", {})
            prob_type = result.get("problem_type", "classification")
            target_mapping = result.get("target_label_mapping") or {}

            st.info(f"**Active Inference Model:** `{best_model_name}` &nbsp;|&nbsp; **Target Variable:** `{target_col}`")

            # Auto-fill State Management
            if "prediction_form_seed" not in st.session_state:
                st.session_state["prediction_form_seed"] = 0
            if "prediction_autofill_values" not in st.session_state:
                st.session_state["prediction_autofill_values"] = {}
            if "autofill_status_msg" not in st.session_state:
                st.session_state["autofill_status_msg"] = None

            norm_cols = {str(c).strip().lower().replace(" ", "_"): c for c in raw_df.columns} if raw_df is not None else {}

            def build_autofill_dict(row: pd.Series | None):
                mapped = {}
                for item in schema:
                    fname = item.get("name")
                    ftype = item.get("type")
                    raw_col = norm_cols.get(fname)
                    raw_val = row.get(raw_col) if (row is not None and raw_col) else None

                    if ftype == "select":
                        options = item.get("options", [])
                        labels = [opt["label"] for opt in options]
                        matched_label = labels[0] if labels else ""
                        if pd.notna(raw_val):
                            for opt in options:
                                if str(opt["label"]).lower() == str(raw_val).lower():
                                    matched_label = opt["label"]
                                    break
                                elif str(opt["value"]) == str(raw_val) or str(opt["value"]) == str(int(float(raw_val)) if isinstance(raw_val, (int, float)) else ""):
                                    matched_label = opt["label"]
                                    break
                        mapped[fname] = matched_label
                    else:
                        def_val = float(item.get("default", 0.0))
                        if pd.notna(raw_val):
                            try:
                                num_v = float(raw_val)
                                min_v = float(item.get("min", -1e9)) if "min" in item else -1e9
                                max_v = float(item.get("max", 1e9)) if "max" in item else 1e9
                                mapped[fname] = max(min(num_v, max_v), min_v)
                            except Exception:
                                mapped[fname] = def_val
                        else:
                            mapped[fname] = def_val
                return mapped

            # Pre-populate on initial render if empty
            if not st.session_state["prediction_autofill_values"]:
                initial_row = raw_df.iloc[0] if (raw_df is not None and not raw_df.empty) else None
                st.session_state["prediction_autofill_values"] = build_autofill_dict(initial_row)
                if initial_row is not None:
                    actual_val = initial_row.get(target_col, "—")
                    st.session_state["autofill_status_msg"] = f"Values pre-loaded from dataset record #0 (Actual {target_col}: `{actual_val}`)"

            # Auto-Fill Action Bar
            st.markdown("##### ⚡ Auto-Fill Feature Values")
            st.caption("Auto-populate input fields with real records or typical values directly from your dataset.")
            col_af1, col_af2, col_af3, col_af4 = st.columns([1.5, 1.5, 1.5, 2.5])

            with col_af1:
                if st.button("🎲 Random Sample", use_container_width=True, help="Load features from a randomly sampled row in the dataset"):
                    if raw_df is not None and not raw_df.empty:
                        rand_idx = int(np.random.randint(0, len(raw_df)))
                        rand_row = raw_df.iloc[rand_idx]
                        st.session_state["prediction_autofill_values"] = build_autofill_dict(rand_row)
                        st.session_state["prediction_form_seed"] += 1
                        actual_val = rand_row.get(target_col, "—")
                        st.session_state["autofill_status_msg"] = f"Auto-filled with values from record #{rand_idx} (Actual {target_col}: `{actual_val}`)"
                        st.rerun()

            with col_af2:
                if st.button("⚡ First Record (#0)", use_container_width=True, help="Load features from the first record in the dataset"):
                    if raw_df is not None and not raw_df.empty:
                        first_row = raw_df.iloc[0]
                        st.session_state["prediction_autofill_values"] = build_autofill_dict(first_row)
                        st.session_state["prediction_form_seed"] += 1
                        actual_val = first_row.get(target_col, "—")
                        st.session_state["autofill_status_msg"] = f"Auto-filled with values from record #0 (Actual {target_col}: `{actual_val}`)"
                        st.rerun()

            with col_af3:
                if st.button("📊 Default Medians", use_container_width=True, help="Reset all feature inputs to median/mode default values"):
                    st.session_state["prediction_autofill_values"] = build_autofill_dict(None)
                    st.session_state["prediction_form_seed"] += 1
                    st.session_state["autofill_status_msg"] = "Reset all input values to feature medians and default modes"
                    st.rerun()

            with col_af4:
                if raw_df is not None and not raw_df.empty:
                    preview_n = min(len(raw_df), 20)
                    row_choices = ["-- Select Dataset Record --"] + [f"Record #{i} ({target_col}: {raw_df.iloc[i].get(target_col, '—')})" for i in range(preview_n)]
                    chosen_rec = st.selectbox(
                        "Record Picker",
                        options=row_choices,
                        label_visibility="collapsed",
                        key=f"rec_picker_{st.session_state['prediction_form_seed']}",
                    )
                    if chosen_rec != "-- Select Dataset Record --":
                        chosen_idx = int(chosen_rec.split("#")[1].split(" ")[0])
                        sel_row = raw_df.iloc[chosen_idx]
                        st.session_state["prediction_autofill_values"] = build_autofill_dict(sel_row)
                        st.session_state["prediction_form_seed"] += 1
                        actual_val = sel_row.get(target_col, "—")
                        st.session_state["autofill_status_msg"] = f"Auto-filled with values from record #{chosen_idx} (Actual {target_col}: `{actual_val}`)"
                        st.rerun()

            if st.session_state["autofill_status_msg"]:
                st.info(st.session_state["autofill_status_msg"])

            # Build Form dynamically based on schema
            with st.form("prediction_form"):
                st.subheader("Feature Input Values")

                form_seed = st.session_state["prediction_form_seed"]
                af_values = st.session_state["prediction_autofill_values"]

                num_cols = 3
                cols = st.columns(num_cols)
                user_inputs = {}

                for i, item in enumerate(schema):
                    col_target = cols[i % num_cols]
                    fname = item.get("name")
                    ftype = item.get("type")
                    fhelp = item.get("help", "")

                    with col_target:
                        if ftype == "select":
                            options = item.get("options", [])
                            # Format select options
                            labels = [opt["label"] for opt in options]
                            values = [opt["value"] for opt in options]
                            cur_label = af_values.get(fname, labels[0] if labels else "")
                            def_idx = labels.index(cur_label) if cur_label in labels else 0

                            selected_label = st.selectbox(
                                fname,
                                options=labels,
                                index=def_idx,
                                help=fhelp,
                                key=f"pred_input_{fname}_{form_seed}",
                            )
                            chosen_val = values[labels.index(selected_label)] if selected_label in labels else 0
                            user_inputs[fname] = chosen_val
                        else:
                            # Number input
                            def_val = float(item.get("default", 0.0))
                            cur_val = float(af_values.get(fname, def_val))
                            min_val = float(item.get("min", -1e9)) if "min" in item else None
                            max_val = float(item.get("max", 1e9)) if "max" in item else None
                            step = 1.0 if item.get("step") == 1 else None

                            val = st.number_input(
                                fname,
                                value=cur_val,
                                min_value=min_val,
                                max_value=max_val,
                                step=step,
                                help=fhelp,
                                key=f"pred_input_{fname}_{form_seed}",
                            )
                            user_inputs[fname] = val

                predict_btn = st.form_submit_button("🎯 Make Prediction", type="primary", use_container_width=True)

            if predict_btn:
                try:
                    pred_res = execute_prediction(
                        model=model,
                        features=features,
                        scaling_params=scaling_params,
                        problem_type=prob_type,
                        target_mapping=target_mapping,
                        payload=user_inputs,
                    )

                    pred_val = pred_res.get("prediction")
                    pred_label = pred_res.get("prediction_label", pred_val)
                    probs = pred_res.get("probabilities", {})

                    st.markdown("### Prediction Result")
                    st.success(f"🎯 **Predicted Result ({target_col}):** `{pred_label}`")

                    if probs:
                        st.markdown("**Prediction Confidence Probabilities:**")
                        p_cols = st.columns(len(probs))
                        for idx, (cls_name, prob_val) in enumerate(probs.items()):
                            p_cols[idx].metric(f"Class {cls_name}", f"{prob_val * 100:.1f}%")
                            p_cols[idx].progress(min(max(float(prob_val), 0.0), 1.0))
                except Exception as exc:
                    st.error(f"Inference error: {str(exc)}")

            # -----------------------------------------------------------------
            # Model Export & Download Section ("At Last")
            # -----------------------------------------------------------------
            st.markdown("---")
            st.subheader("💾 Export & Download Trained Models")
            st.caption("Download production-ready serialized model artifacts (.joblib or .pkl) for external deployment, API hosting, or offline inference.")

            # Collect available models from training tournament
            all_trained = result.get("trained_models", {})
            if not all_trained and result.get("best_model_object") is not None:
                all_trained = {best_model_name: result.get("best_model_object")}

            if all_trained:
                model_names = list(all_trained.keys())
                # Ensure the Best Model appears first as the default
                sorted_model_names = []
                if best_model_name in model_names:
                    sorted_model_names.append(best_model_name)
                for m in model_names:
                    if m not in sorted_model_names:
                        sorted_model_names.append(m)

                # Format options with distinct badges
                labels_dict = {}
                for m in sorted_model_names:
                    if m == best_model_name:
                        labels_dict[m] = f"🏆 {m} (Recommended - Best Model)"
                    else:
                        labels_dict[m] = f"⚡ {m}"

                col_dl1, col_dl2 = st.columns([2.5, 1.5])
                with col_dl1:
                    selected_download_model_name = st.selectbox(
                        "Select Model to Export",
                        options=sorted_model_names,
                        format_func=lambda m: labels_dict.get(m, m),
                        index=0,  # Default is the Best Model
                        help="Choose which algorithm's trained model artifact to download",
                    )
                with col_dl2:
                    export_format = st.selectbox(
                        "File Format",
                        options=["Joblib (.joblib)", "Pickle (.pkl)"],
                        index=0,
                        help="Joblib is recommended for Scikit-Learn models with NumPy arrays",
                    )

                selected_model_obj = all_trained.get(selected_download_model_name)

                if selected_model_obj is not None:
                    # Metric score lookup from leaderboard comparison
                    comp_lookup = {r.get("Model"): r for r in result.get("comparison", [])}
                    model_stats = comp_lookup.get(selected_download_model_name, {})
                    prim_metric_name = result.get("primary_metric", "Accuracy")
                    metric_score = model_stats.get(prim_metric_name)

                    with st.container(border=True):
                        c_meta1, c_meta2, c_meta3, c_meta4 = st.columns(4)
                        c_meta1.metric("Selected Algorithm", selected_download_model_name)
                        if metric_score is not None:
                            metric_display = (
                                f"{float(metric_score)*100:.2f}%"
                                if prob_type.lower() == "classification"
                                else f"{float(metric_score):.4f}"
                            )
                            c_meta2.metric(f"Score ({prim_metric_name})", metric_display)
                        else:
                            c_meta2.metric("Score", "Trained")
                        c_meta3.metric("Features Included", len(features))
                        c_meta4.metric("Problem Type", prob_type.title())

                        # Prepare binary model buffer
                        buf = io.BytesIO()
                        is_pkl = "pickle" in export_format.lower()
                        file_ext = "pkl" if is_pkl else "joblib"
                        clean_filename = selected_download_model_name.lower().replace(" ", "_").replace("-", "_")
                        download_filename = f"{clean_filename}_{prob_type.lower()}.{file_ext}"

                        if is_pkl:
                            import pickle
                            pickle.dump(selected_model_obj, buf)
                        else:
                            joblib.dump(selected_model_obj, buf)
                        buf.seek(0)

                        st.download_button(
                            label=f"📥 Download {selected_download_model_name} (.{file_ext})",
                            data=buf.getvalue(),
                            file_name=download_filename,
                            mime="application/octet-stream",
                            type="primary",
                            use_container_width=True,
                        )

                        with st.expander("🐍 How to load and use this model in Python"):
                            loader_lib = "pickle" if is_pkl else "joblib"
                            features_preview = str(features[:6]) if len(features) > 6 else str(features)
                            st.code(
                                f"""import {loader_lib}
import pandas as pd

# 1. Load the trained model from disk
with open("{download_filename}", "rb") as f:
    model = {loader_lib}.load(f)

# 2. Prepare sample feature data (Expected features: {features_preview})
sample_data = pd.DataFrame([
    # [Insert feature values here matching columns: {features_preview}]
], columns={features})

# 3. Generate predictions
predictions = model.predict(sample_data)
print("Prediction:", predictions)
""",
                                language="python",
                            )
            else:
                st.info("Train models by clicking 'Run AutoML Pipeline' to enable model export.")


# -----------------------------------------------------------------------------
# Runtime Execution Router
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    render_streamlit_app()