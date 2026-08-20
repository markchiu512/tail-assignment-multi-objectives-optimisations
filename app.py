import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import os
import sys
import traceback
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from databricks import sql
from databricks.sdk import WorkspaceClient
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

from run_pipeline import (
    pipeline,
    feature_engineering, fuel_prediction, lof_enrichment,
    eligibility_filter, assignment_optimisation, post_processing,
)
from src.queries import daily_input_query, ground_events_query
from src.volume_io import make_ws_client, vol_list, vol_read_bytes, vol_exists, vol_write_bytes, vol_rmdir
import config


st.set_page_config(
    page_title="Optimised Tail Allocation",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stProgress > div > div > div > div {
        background-color: #FF6900;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def _get_ws_client():
    # Try OAuth M2M first. Probe UC Volume access; if the SP lacks USE_CATALOG (dev SP), fall back to PAT.
    client_id = os.environ.get("DATABRICKS_CLIENT_ID")
    client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")
    if client_id and client_secret:
        try:
            client = make_ws_client()
            client.files.get_directory_metadata(config.VOLUME_BASE)
            return client
        except Exception:
            pass  # fall through to PAT
    token = os.environ.get("DATABRICKS_TOKEN", "").strip()
    if token:
        host = os.environ.get("DATABRICKS_HOST", "your-workspace.cloud.databricks.com")
        if not host.startswith("http"):
            host = f"https://{host}"
        return WorkspaceClient(host=host, token=token, auth_type="pat")
    raise RuntimeError("No usable Databricks credentials: OAuth M2M probe failed and DATABRICKS_TOKEN is empty")


def _vol_list(dir_path: str) -> list:
    return vol_list(_get_ws_client(), dir_path)


def _vol_read_bytes(file_path: str) -> bytes:
    return vol_read_bytes(_get_ws_client(), file_path)


def _vol_read_bytes_fallback(*candidate_paths: str) -> bytes:
    """Read the first candidate path that exists. Lets the v2 app read v1.x
    files stored under their old names/directories. Raises if none exist."""
    last_err: Exception | None = None
    for p in candidate_paths:
        try:
            return _vol_read_bytes(p)
        except Exception as e:  # noqa: PERF203 - fallback chain is short
            last_err = e
            continue
    raise last_err if last_err else FileNotFoundError("no candidate paths given")


def _vol_exists(file_path: str) -> bool:
    return vol_exists(_get_ws_client(), file_path)


def _vol_write_bytes(file_path: str, data: bytes) -> None:
    vol_write_bytes(_get_ws_client(), file_path, data)


def _vol_rmdir(dir_path: str) -> None:
    vol_rmdir(_get_ws_client(), dir_path)


@st.cache_resource
def _load_gurobi_secrets_once():
    """Load optional Gurobi credentials from the configured secret scope."""
    if config.SOLVER_BACKEND != "gurobi":
        return
    import base64
    keys = {
        "GUROBI_MANAGER":   "gurobi_manager",
        "GUROBI_ACCESS_ID": "gurobi_access_id",
        "GUROBI_SECRET":    "gurobi_secret",
        "GUROBI_APP_NAME":  "gurobi_app_name",
    }
    secret_scope = os.environ.get("DATABRICKS_SECRET_SCOPE", "aviation-optimiser-secrets")
    try:
        client = _get_ws_client()
        for env_var, secret_key in keys.items():
            resp = client.secrets.get_secret(scope=secret_scope, key=secret_key)
            if resp and resp.value:
                os.environ[env_var] = base64.b64decode(resp.value).decode("utf-8")
        os.environ.setdefault("GUROBI_GROUP", "default")
    except Exception as e:
        st.warning(f"Could not load Gurobi secrets: {e}")


_load_gurobi_secrets_once()


def _stage_inputs_to_local() -> list[str]:
    # Mirror the Volume input files polars needs into /tmp, since FUSE /Volumes isn't available in the app container.
    # Returns list of files that could not be staged (so the caller can surface a clear error).
    os.makedirs(config.INPUT_DIRECTORY, exist_ok=True)
    os.makedirs(config.INTERMEDIATE_DIRECTORY, exist_ok=True)
    os.makedirs(config.OUTPUT_DIRECTORY, exist_ok=True)

    # Stage ML model files Volume -> /tmp so Stage 2 can open() them.
    # The pooled models exceed the 10 MB/file app deploy limit and live on the Volume;
    # the container has no FUSE /Volumes mount.
    # Skip only when running locally with the real pooled joblib checked out in ./model.
    # Always patch config.ML_MODELS_DIRECTORY to the staging dir after staging so
    # stage modules get the right path regardless of how config.py resolved at import.
    _pooled_joblib = os.path.join("model", "pooled_lgbm_target_baseline_fuel_kg.joblib")
    if not os.path.isfile(_pooled_joblib):
        _staging_model_dir = config.ML_MODELS_STAGING_DIRECTORY
        os.makedirs(_staging_model_dir, exist_ok=True)
        _model_stage_errors = []
        try:
            for item in _vol_list(config.VOLUME_MODEL_DIRECTORY):
                if item.is_directory:
                    continue
                local_path = os.path.join(_staging_model_dir, item.name)
                if os.path.exists(local_path):
                    continue
                try:
                    with open(local_path, "wb") as f:
                        f.write(_vol_read_bytes(item.path))
                except Exception as _e:
                    _model_stage_errors.append(f"{item.name}: {_e}")
        except Exception as _e:
            _model_stage_errors.append(f"vol_list: {_e}")
        if _model_stage_errors:
            print(f"[WARN] model staging errors: {_model_stage_errors}", flush=True)
        # Patch config so stage modules pick up the staged path
        config.ML_MODELS_DIRECTORY = _staging_model_dir

    required = [
        f"{config.DATE_PREFIX}_Opti_Tail_With_MTOW__APM_and_Seat_Capacity.csv",
        "cost_index.csv",
    ]
    if getattr(config, 'GROUND_EVENTS_SOURCE', 'xls') == 'xls':
        required.append(config.GROUNDEVENTS_FILE)

    missing = []
    for name in required:
        vol_path = f"{config.VOLUME_INPUT_DIRECTORY}/{name}"
        local_path = f"{config.INPUT_DIRECTORY}/{name}"
        try:
            data = _vol_read_bytes(vol_path)
        except Exception:
            missing.append(name)
            continue
        with open(local_path, "wb") as f:
            f.write(data)
    # Stage existing intermediate files for this date so re-running steps 2–4 individually
    # can find their predecessor outputs in /tmp after a fresh deploy.
    try:
        for item in _vol_list(config.VOLUME_INTERMEDIATE_DIRECTORY):
            if item.is_directory:
                continue
            local_path = os.path.join(config.INTERMEDIATE_DIRECTORY, item.name)
            if os.path.exists(local_path):
                continue
            try:
                with open(local_path, "wb") as f:
                    f.write(_vol_read_bytes(item.path))
            except Exception:
                pass
    except Exception:
        pass
    # Stage Stage 5 output so Swap Approval (Stage 6) can re-run individually.
    # Do NOT stage swap_approval CSV or summary TXT — Stage 6 regenerates those.
    algo_output = f"{config.DATE_PREFIX}_assignment_optimisation.csv"
    local_algo = os.path.join(config.INTERMEDIATE_DIRECTORY, algo_output)
    if not os.path.exists(local_algo):
        try:
            with open(local_algo, "wb") as f:
                f.write(_vol_read_bytes(f"{config.VOLUME_INTERMEDIATE_DIRECTORY}/{algo_output}"))
        except Exception:
            pass
    return missing


def _stage_outputs_to_volume() -> None:
    # Upload intermediate + output files written by the pipeline to the Volume.
    for local_dir, vol_dir in [
        (config.INTERMEDIATE_DIRECTORY, config.VOLUME_INTERMEDIATE_DIRECTORY),
        (config.OUTPUT_DIRECTORY, config.VOLUME_OUTPUT_DIRECTORY),
    ]:
        if not os.path.isdir(local_dir):
            continue
        for name in os.listdir(local_dir):
            local_path = os.path.join(local_dir, name)
            if not os.path.isfile(local_path):
                continue
            with open(local_path, "rb") as f:
                _vol_write_bytes(f"{vol_dir}/{name}", f.read())
    # Guard: only run cleanup if Stage 6 ran (i.e., /tmp has the swap_approval CSV).
    suffix = '_swap_approval.csv'
    if os.path.isdir(config.OUTPUT_DIRECTORY):
        local_cleaned = {n for n in os.listdir(config.OUTPUT_DIRECTORY) if n.endswith(suffix)}
        if local_cleaned:
            try:
                for item in _vol_list(config.VOLUME_OUTPUT_DIRECTORY):
                    if (not item.is_directory and item.name.endswith(suffix)
                            and item.name not in local_cleaned):
                        try:
                            _get_ws_client().files.delete(item.path)
                        except Exception:
                            pass
            except Exception:
                pass


def query_databricks_data(mtl_date: str, date_prefix: str) -> pd.DataFrame:

    date_obj = datetime.strptime(mtl_date, "%d/%m/%Y")
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day

    # On Databricks Apps: DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET are auto-injected (OAuth M2M)
    # Locally: set DATABRICKS_TOKEN in environment or .env file
    server_hostname = os.environ.get("DATABRICKS_HOST", "your-workspace.cloud.databricks.com")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/<warehouse-id>")

    token = os.environ.get("DATABRICKS_TOKEN")
    if token:
        connection = sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=token,
        )
    else:
        # OAuth M2M path (Databricks Apps)
        from databricks.sdk.core import Config, oauth_service_principal
        def credentials_provider():
            cfg = Config(
                host=f"https://{server_hostname}",
                client_id=os.environ.get("DATABRICKS_CLIENT_ID"),
                client_secret=os.environ.get("DATABRICKS_CLIENT_SECRET"),
            )
            return oauth_service_principal(cfg)
        connection = sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            credentials_provider=credentials_provider,
        )

    query = daily_input_query(year, month, day)

    with connection.cursor() as cursor:
        cursor.execute(query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(result, columns=columns)

    connection.close()
    return df

# def query_databricks(query: str) -> pd.DataFrame:
#     with connection.cursor() as cursor:
#         cursor.execute(query)
#         result = cursor.fetchall()
#         columns = [desc[0] for desc in cursor.description]
#         return pd.DataFrame(result, columns=columns)


def _fetch_ground_events_app(mtl_date: str, date_prefix: str) -> int:
    """Fetch ground events from Databricks and write to INPUT_DIRECTORY. Returns row count."""
    date_obj = datetime.strptime(mtl_date, "%d/%m/%Y")
    year, month, day = date_obj.year, date_obj.month, date_obj.day

    server_hostname = os.environ.get("DATABRICKS_HOST", "your-workspace.cloud.databricks.com")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/<warehouse-id>")

    token = os.environ.get("DATABRICKS_TOKEN")
    if token:
        connection = sql.connect(server_hostname=server_hostname, http_path=http_path, access_token=token)
    else:
        from databricks.sdk.core import Config, oauth_service_principal
        def credentials_provider():
            cfg = Config(
                host=f"https://{server_hostname}",
                client_id=os.environ.get("DATABRICKS_CLIENT_ID"),
                client_secret=os.environ.get("DATABRICKS_CLIENT_SECRET"),
            )
            return oauth_service_principal(cfg)
        connection = sql.connect(server_hostname=server_hostname, http_path=http_path,
                                 credentials_provider=credentials_provider)

    query = ground_events_query(year, month, day)
    with connection.cursor() as cursor:
        cursor.execute(query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(result, columns=columns)
    connection.close()

    os.makedirs(config.INPUT_DIRECTORY, exist_ok=True)
    out_path = f"{config.INPUT_DIRECTORY}/{date_prefix}_ground_events_databricks.csv"
    df.to_csv(out_path, index=False)
    return len(df)


def sort_date_strings(date_list: list[str], descending: bool = True) -> list[str]:
    """Sort date strings in format like '31JAN2026' by actual date."""
    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, '%d%b%Y')
        except ValueError:
            return datetime.min  # Put unparseable dates at the end
    return sorted(date_list, key=parse_date, reverse=descending)


@st.cache_data(ttl=300)
def get_available_dates():
    """Get list of dates with available output data, sorted by date descending."""
    items = _vol_list(config.VOLUME_OUTPUT_BASE)
    date_folders = [i.name for i in items if i.is_directory and not i.name.startswith('.')]
    return sort_date_strings(date_folders, descending=True)[:5]


@st.cache_data(ttl=3600)
def load_results_data(date_prefix: str) -> pd.DataFrame | None:
    # v2: {date}_assignment_optimisation.csv in intermediate/
    # v1.x fallback: {date}_opti_tails_results.csv in output/
    try:
        return pd.read_csv(io.BytesIO(_vol_read_bytes_fallback(
            f"{config.VOLUME_INTERMEDIATE_BASE}/{date_prefix}/{date_prefix}_assignment_optimisation.csv",
            f"{config.VOLUME_OUTPUT_BASE}/{date_prefix}/{date_prefix}_opti_tails_results.csv",
        )))
    except Exception:
        return None


def reconstruct_cost_matrix(group_df: pd.DataFrame) -> np.ndarray:
    """Reconstruct n×n cost matrix from results CSV columns for one group.

    Mirrors the formula in src/opti_tails_algorithm.py::optimize_group().
    Rows = aircraft (indexed by aircraftreg), Columns = routes (indexed by Route).
    """
    group_df = group_df.reset_index(drop=True)
    perf_corr     = group_df['avg_perf_corr'].values
    fh_rate       = group_df['total_fh_rate'].values
    cycle_rate    = group_df['total_cycle_rate'].values
    baseline_fuel = group_df['total_baseline_fuel'].values
    pred_hours    = group_df['total_pred_act_hours'].values
    sectors       = group_df['total_sectors'].values

    fuel_cost  = np.outer(perf_corr, baseline_fuel) * config.FUEL_PRICE
    fh_cost    = np.outer(fh_rate, pred_hours)
    cycle_cost = np.outer(cycle_rate, sectors)
    return fuel_cost + fh_cost + cycle_cost


def render_cost_matrix_heatmap(group_df: pd.DataFrame) -> go.Figure:
    """Render cost matrix heatmap with cell-border highlights for assignments.

    Black border  = original/current assignment (diagonal).
    Blue border   = optimal assignment (solver).
    Navy border   = unchanged (current == optimal).
    """
    group_df = group_df.reset_index(drop=True)
    n = len(group_df)
    cost_matrix = reconstruct_cost_matrix(group_df)

    aircraft_labels = group_df['aircraftreg'].tolist()
    route_labels = (group_df['Route'].tolist() if 'Route' in group_df.columns
                    else [f"Route {i}" for i in range(n)])

    reg_to_idx = {reg: i for i, reg in enumerate(aircraft_labels)}

    opti_row_for_col = {}
    for j in range(n):
        opti_reg = group_df.loc[j, 'opti_aircraftreg']
        opti_row_for_col[j] = reg_to_idx.get(opti_reg, j)

    cell_text = [[f"${cost_matrix[i, j]:,.0f}" for j in range(n)] for i in range(n)]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=cost_matrix,
        x=route_labels,
        y=aircraft_labels,
        colorscale='YlOrRd',
        reversescale=False,
        text=cell_text,
        texttemplate="%{text}" if n <= 8 else "",
        colorbar=dict(title="Cost ($)"),
        hovertemplate="Aircraft: %{y}<br>Route: %{x}<br>Cost: $%{z:,.0f}<extra></extra>",
    ))

    for j in range(n):
        opti_i = opti_row_for_col[j]
        unchanged = (opti_i == j)

        if unchanged:
            fig.add_shape(
                type='rect',
                x0=j - 0.45, x1=j + 0.45,
                y0=j - 0.45, y1=j + 0.45,
                line=dict(color='#9e9e9e', width=3),
                xref='x', yref='y',
            )
        else:
            fig.add_shape(
                type='rect',
                x0=j - 0.45, x1=j + 0.45,
                y0=j - 0.45, y1=j + 0.45,
                line=dict(color='#000000', width=3),
                xref='x', yref='y',
            )
            fig.add_shape(
                type='rect',
                x0=j - 0.45, x1=j + 0.45,
                y0=opti_i - 0.45, y1=opti_i + 0.45,
                line=dict(color='#1976d2', width=3),
                xref='x', yref='y',
            )

    for color, name in [
        ('#000000', 'Original (diagonal)'),
        ('#1976d2', 'Optimal (solver)'),
        ('#9e9e9e', 'No change'),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color=color, width=3),
            name=name,
        ))

    fig.update_layout(
        title=f"Cost Matrix — {group_df['group_index'].iloc[0]}",
        xaxis_title="Route",
        yaxis_title="Aircraft",
        height=max(400, n * 55),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


@st.cache_data(ttl=3600)
def load_intermediate_data(date_prefix: str, stage: str) -> pd.DataFrame | None:
    inter = f"{config.VOLUME_INTERMEDIATE_BASE}/{date_prefix}"
    outp  = f"{config.VOLUME_OUTPUT_BASE}/{date_prefix}"
    # Each stage: v2 path first, then v1.x fallback(s). v1.x had 3 intermediate
    # files (databricks_data_interim, ml_predictions_output, opti_tail_cleaned)
    # plus the results CSV in output/.
    candidate_mapping = {
        "feature_engineering":     [f"{inter}/{date_prefix}_feature_engineering.csv",
                                     f"{inter}/{date_prefix}_databricks_data_interim.csv"],
        "fuel_prediction":         [f"{inter}/{date_prefix}_fuel_prediction.csv",
                                     f"{inter}/{date_prefix}_ml_predictions_output.csv"],
        "lof_enrichment":          [f"{inter}/{date_prefix}_lof_enrichment.csv",
                                     f"{inter}/{date_prefix}_opti_tail_cleaned.csv"],
        "eligibility_filter":      [f"{inter}/{date_prefix}_eligibility_filter.csv",
                                     f"{inter}/{date_prefix}_opti_tail_cleaned.csv"],
        "assignment_optimisation": [f"{inter}/{date_prefix}_assignment_optimisation.csv",
                                     f"{outp}/{date_prefix}_opti_tails_results.csv"],
    }
    if stage not in candidate_mapping:
        return None
    try:
        return pd.read_csv(io.BytesIO(_vol_read_bytes_fallback(*candidate_mapping[stage])))
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_input_data() -> dict:
    items = _vol_list(config.VOLUME_INPUT_DIRECTORY)
    data = {}
    flight_files = [i for i in items if not i.is_directory
                    and '_Opti_Tail_With_MTOW' in i.name and i.name.endswith('.csv')]
    if flight_files:
        data["flights"] = pd.read_csv(io.BytesIO(_vol_read_bytes(flight_files[0].path)))
    cost_index = [i for i in items if i.name == 'cost_index.csv']
    if cost_index:
        data["cost_index"] = pd.read_csv(io.BytesIO(_vol_read_bytes(cost_index[0].path)))
    return data


def run_pipeline_stage(stage_name: str, module) -> tuple[bool, str]:
    try:
        module.main()
        return True, f"{stage_name} completed successfully"
    except Exception as e:
        return False, f"Error in {stage_name}: {str(e)}\n{traceback.format_exc()}"


@st.cache_data(ttl=3600)
def load_summary_txt(date_prefix: str) -> str | None:
    items = _vol_list(f"{config.VOLUME_OUTPUT_BASE}/{date_prefix}")
    txt_files = [i for i in items if not i.is_directory and i.name.endswith('.txt')]
    if not txt_files:
        return None
    try:
        return _vol_read_bytes(txt_files[0].path).decode('utf-8')
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_cleaned_csv(date_prefix: str) -> pd.DataFrame | None:
    """Load the swap approval CSV (Stage 6 output) for a given date.
    v2: {date}_swap_approval.csv; v1.x fallback: {date}_opti_tails_results_cleaned.csv."""
    try:
        return pd.read_csv(io.BytesIO(_vol_read_bytes_fallback(
            f"{config.VOLUME_OUTPUT_BASE}/{date_prefix}/{date_prefix}_swap_approval.csv",
            f"{config.VOLUME_OUTPUT_BASE}/{date_prefix}/{date_prefix}_opti_tails_results_cleaned.csv",
        )))
    except Exception:
        return None


@st.cache_data(ttl=3600)
def create_cleaned_csvs_zip(date_prefix: str) -> bytes | None:
    import zipfile
    items = _vol_list(f"{config.VOLUME_OUTPUT_BASE}/{date_prefix}")
    suffix = "_opti_tails_results_cleaned.csv"
    combined_name = f"{date_prefix}_opti_tails_results_cleaned.csv"
    cleaned = [i for i in items if not i.is_directory and i.name.endswith(suffix) and i.name != combined_name]
    if not cleaned:
        return None
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for item in cleaned:
            zf.writestr(item.name, _vol_read_bytes(item.path))
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


@st.cache_data(ttl=3600)
def load_feedback(date_prefix: str) -> dict:
    """Load feedback JSON for a given date, returning defaults if not found."""
    path = f"{config.VOLUME_OUTPUT_BASE}/{date_prefix}/{date_prefix}_feedback.json"
    try:
        return json.loads(_vol_read_bytes(path).decode('utf-8'))
    except Exception:
        return {"group_comments": {}, "general_notes": ""}


def save_feedback(date_prefix: str, feedback: dict) -> None:
    """Write feedback JSON for a given date."""
    path = f"{config.VOLUME_OUTPUT_BASE}/{date_prefix}/{date_prefix}_feedback.json"
    content = json.dumps(feedback, indent=2).encode('utf-8')
    _get_ws_client().files.upload(path, io.BytesIO(content), overwrite=True)
    load_feedback.clear()


def _merge_feedback_columns(df: pd.DataFrame, feedback: dict) -> pd.DataFrame:
    """Add Status and Comment columns to a dataframe based on feedback data.

    Reverse adaptation: if the source CSV already has Status/Comment columns
    (e.g., a previously-downloaded copy was re-uploaded), those values are
    preserved as the baseline. JSON entries take precedence where present;
    rows whose group is not in the JSON keep the CSV value. So comments
    survive even when {date}_feedback.json is missing or incomplete.
    """
    group_comments = feedback.get("group_comments", {})

    if 'Status' not in df.columns:
        df['Status'] = "Pending"
    else:
        df['Status'] = (df['Status'].fillna("Pending").astype(str)
                          .replace({"": "Pending", "nan": "Pending"}))
    if 'Comment' not in df.columns:
        df['Comment'] = ""
    else:
        df['Comment'] = df['Comment'].fillna("").astype(str).replace({"nan": ""})

    if 'group_index' in df.columns and group_comments:
        groups = df['group_index'].astype(str)
        for group_id, entry in group_comments.items():
            mask = groups == group_id
            if not mask.any():
                continue
            if 'status' in entry:
                df.loc[mask, 'Status'] = entry['status']
            if 'comment' in entry:
                df.loc[mask, 'Comment'] = entry['comment']
    return df


def _extract_feedback_from_df(df: pd.DataFrame) -> dict:
    """Extract per-group feedback from an edited dataframe.

    If any row in a group has been changed from Pending, that status and
    comment are applied to the whole group (one change = whole group).
    """
    group_comments = {}
    if 'group_index' in df.columns:
        for group_id in df['group_index'].astype(str).unique():
            group_rows = df[df['group_index'].astype(str) == group_id]
            # Find first non-Pending status in the group (user edited one row)
            changed = group_rows[group_rows['Status'] != 'Pending']
            if not changed.empty:
                row = changed.iloc[0]
                status = row['Status']
                comment = row.get('Comment', '')
            else:
                # Check for comment-only edits
                commented = group_rows[group_rows['Comment'].astype(str).str.strip() != '']
                if not commented.empty:
                    row = commented.iloc[0]
                    status = 'Pending'
                    comment = row['Comment']
                else:
                    continue
            group_comments[group_id] = {"status": status, "comment": comment}
    return group_comments


def _render_feedback_grid(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Render an AG Grid with editable Status/Comment and row colouring.

    Returns the edited DataFrame.
    """
    gb = GridOptionsBuilder.from_dataframe(df)

    # Status column: editable dropdown, single-click to open, triangle indicator
    status_cell_renderer = JsCode("""
    function(params) {
        return params.value + ' ▾';
    }
    """)
    gb.configure_column(
        "Status", editable=True,
        cellEditor="agSelectCellEditor",
        cellEditorParams={"values": ["Pending", "Actioned", "Cannot Action"]},
        cellRenderer=status_cell_renderer,
        singleClickEdit=True,
    )
    # Comment column: editable text
    gb.configure_column("Comment", editable=True)

    # Savings column: right-aligned
    if 'savings' in df.columns:
        gb.configure_column("savings", editable=False, type=["customColumn"],
                            cellStyle={'textAlign': 'right'})

    # All other columns: read-only
    for col in df.columns:
        if col not in ("Status", "Comment", "savings"):
            gb.configure_column(col, editable=False)

    # Row colouring via JS — runs client-side, updates instantly on edit
    row_style_jscode = JsCode("""
    function(params) {
        if (params.data.Status === 'Actioned') {
            return {'backgroundColor': '#d4edda'};
        }
        if (params.data.Status === 'Cannot Action') {
            return {'backgroundColor': '#f8d7da'};
        }
        return {};
    }
    """)
    # Propagate Status/Comment edits to all rows in the same swap group
    on_cell_changed = JsCode("""
    function(event) {
        var col = event.column.colId;
        if (col === 'Status' || col === 'Comment') {
            var groupId = event.data.group_index;
            event.api.forEachNode(function(node) {
                if (node.data.group_index === groupId && node !== event.node) {
                    node.setDataValue(col, event.newValue);
                }
            });
        }
    }
    """)
    gb.configure_grid_options(
        getRowStyle=row_style_jscode,
        stopEditingWhenCellsLoseFocus=True,
        onCellValueChanged=on_cell_changed,
    )

    grid_response = AgGrid(
        df,
        gridOptions=gb.build(),
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.VALUE_CHANGED,
        fit_columns_on_grid_load=True,
        key=key,
    )

    return pd.DataFrame(grid_response["data"])


def upload_groundevents_file(uploaded_file) -> tuple[bool, str]:
    try:
        file_path = f"{config.VOLUME_INPUT_DIRECTORY}/{uploaded_file.name}"
        _get_ws_client().files.upload(file_path, io.BytesIO(uploaded_file.getbuffer()), overwrite=True)

        # config.py is a local repo file — open() is fine here
        import re
        config_path = Path("config.py")
        with open(config_path, 'r') as f:
            content = f.read()
        content = re.sub(
            r'GROUNDEVENTS_FILE\s*=\s*"[^"]*"',
            f'GROUNDEVENTS_FILE = "{uploaded_file.name}"',
            content
        )
        with open(config_path, 'w') as f:
            f.write(content)
        load_input_data.clear()
        return True, uploaded_file.name
    except Exception as e:
        return False, str(e)


def _clear_all_volume_caches() -> None:
    """Invalidate every @st.cache_data Volume loader. Used after pipeline runs, cleans, etc."""
    get_available_dates.clear()
    load_results_data.clear()
    load_intermediate_data.clear()
    load_input_data.clear()
    load_summary_txt.clear()
    load_cleaned_csv.clear()
    create_cleaned_csvs_zip.clear()
    load_feedback.clear()


def clean_current_date_files() -> tuple[bool, str]:
    import shutil
    date_prefix = config.DATE_PREFIX
    deleted = {"intermediate": False, "output": False}
    errors = []

    try:
        _vol_rmdir(f"{config.VOLUME_INTERMEDIATE_BASE}/{date_prefix}")
        deleted["intermediate"] = True
    except Exception as e:
        errors.append(f"intermediate/{date_prefix}: {e}")

    try:
        _vol_rmdir(f"{config.VOLUME_OUTPUT_BASE}/{date_prefix}")
        deleted["output"] = True
    except Exception as e:
        errors.append(f"output/{date_prefix}: {e}")

    # Also wipe the local /tmp staging dirs for this date — otherwise stale per-base
    # CSVs left over there will be re-uploaded to the Volume on the next pipeline run.
    for local_dir in [config.INTERMEDIATE_DIRECTORY, config.OUTPUT_DIRECTORY]:
        if os.path.isdir(local_dir):
            try:
                shutil.rmtree(local_dir)
            except Exception as e:
                errors.append(f"local {local_dir}: {e}")

    _clear_all_volume_caches()

    if errors:
        return False, f"Errors cleaning {date_prefix}: {'; '.join(errors)}"
    if not deleted["intermediate"] and not deleted["output"]:
        return True, f"No files found for {date_prefix}"
    parts = [k for k, v in deleted.items() if v]
    return True, f"Cleaned {date_prefix} from {' and '.join(parts)}"


_NO_CHANGE = object()  # sentinel for save_config_to_file params where None is a valid value


def save_config_to_file(
    mtl_date: str,
    fuel_price: float,
    min_ratio_threshold: int,
    filter_top_groups: int,
    groundevents_file: str,
    lease_aircraft: list[str],
    iris_uk_aircraft: list[str],
    iris_eu_aircraft: list[str],
    manual_exclusions: list[str],
    cyprus_prohibited: list[str] = None,
    cyprus_iata: list[str] = None,
    cape_verde_iata: list[str] = None,
    autoland_aircraft: list[str] = None,
    kef_iata: list[str] = None,
    eps_values: list[float] = None,
    selected_eps=_NO_CHANGE,
    auto_select_eps=_NO_CHANGE,
    auto_select_eps_method=_NO_CHANGE,
    eps_max_cost_per_fuel_kg=_NO_CHANGE,
    auto_threshold=_NO_CHANGE,
    fuel_bias=_NO_CHANGE,
    ground_events_source=_NO_CHANGE,
    solver_backend=_NO_CHANGE,
) -> bool:
    """Save configuration changes to config.py."""
    config_path = Path("config.py")

    try:
        with open(config_path, 'r') as f:
            content = f.read()

        import re

        # Update MTL_DATE_OVERRIDE (the literal MTL_DATE assignment is now derived dynamically)
        content = re.sub(
            r"MTL_DATE_OVERRIDE\s*=\s*'[^']*'",
            f"MTL_DATE_OVERRIDE = '{mtl_date}'",
            content
        )

        # Update FUEL_PRICE
        content = re.sub(
            r"FUEL_PRICE\s*=\s*[\d.]+",
            f"FUEL_PRICE = {fuel_price}",
            content
        )

        # Update MIN_SAVINGS_THRESHOLD
        content = re.sub(
            r"MIN_SAVINGS_THRESHOLD\s*=\s*\d+",
            f"MIN_SAVINGS_THRESHOLD = {min_ratio_threshold}",
            content
        )

        # Update FILTER_BY_TOP_GROUPS_INDEX
        content = re.sub(
            r"FILTER_BY_TOP_GROUPS_INDEX\s*=\s*\d+",
            f"FILTER_BY_TOP_GROUPS_INDEX = {filter_top_groups}",
            content
        )

        # Update GROUNDEVENTS_FILE
        content = re.sub(
            r'GROUNDEVENTS_FILE\s*=\s*"[^"]*"',
            f'GROUNDEVENTS_FILE = "{groundevents_file}"',
            content
        )

        # Update LEASE_AIRCRAFTREG
        content = re.sub(
            r"LEASE_AIRCRAFTREG\s*=\s*\[[^\]]*\]",
            f"LEASE_AIRCRAFTREG = {repr(lease_aircraft)}",
            content
        )

        # Update IRIS_UK_AIRCRAFTREG
        content = re.sub(
            r"IRIS_UK_AIRCRAFTREG\s*=\s*\[[^\]]*\]",
            f"IRIS_UK_AIRCRAFTREG = {repr(iris_uk_aircraft)}",
            content
        )

        # Update IRIS_EU_AIRCRAFTREG (handles multiline)
        content = re.sub(
            r"IRIS_EU_AIRCRAFTREG\s*=\s*\[.*?\]",
            f"IRIS_EU_AIRCRAFTREG = {repr(iris_eu_aircraft)}",
            content,
            flags=re.DOTALL
        )

        # Update MANUAL_DROP_AIRCRAFTREG
        content = re.sub(
            r"MANUAL_DROP_AIRCRAFTREG\s*=\s*\[[^\]]*\]",
            f"MANUAL_DROP_AIRCRAFTREG = {repr(manual_exclusions)}",
            content
        )

        # Update constraint lists if provided
        if cyprus_prohibited is not None:
            content = re.sub(
                r"CYPRUS_PROHIBITED\s*=\s*\[.*?\]",
                f"CYPRUS_PROHIBITED = {repr(cyprus_prohibited)}",
                content, flags=re.DOTALL
            )
        if cyprus_iata is not None:
            content = re.sub(
                r"CYPRUS_IATA\s*=\s*\[[^\]]*\]",
                f"CYPRUS_IATA = {repr(cyprus_iata)}",
                content
            )
        if cape_verde_iata is not None:
            content = re.sub(
                r"CAPE_VERDE_IATA\s*=\s*\[[^\]]*\]",
                f"CAPE_VERDE_IATA = {repr(cape_verde_iata)}",
                content
            )
        if autoland_aircraft is not None:
            content = re.sub(
                r"AUTOLAND_AIRCRAFTREG\s*=\s*\[.*?\]",
                f"AUTOLAND_AIRCRAFTREG = {repr(autoland_aircraft)}",
                content, flags=re.DOTALL
            )
        if kef_iata is not None:
            content = re.sub(
                r"KEF_IATA\s*=\s*\[[^\]]*\]",
                f"KEF_IATA = {repr(kef_iata)}",
                content
            )
        if eps_values is not None:
            content = re.sub(
                r"EPS_VALUES\s*=\s*\[.*?\]",
                f"EPS_VALUES = {repr(eps_values)}",
                content, flags=re.DOTALL
            )
        if selected_eps is not _NO_CHANGE:
            val_repr = "None" if selected_eps is None else repr(selected_eps)
            content = re.sub(
                r"SELECTED_EPS\s*=\s*(?:None|[\d.]+)",
                f"SELECTED_EPS = {val_repr}",
                content
            )
        if auto_select_eps is not _NO_CHANGE:
            content = re.sub(
                r"AUTO_SELECT_EPS\s*=\s*(?:True|False)",
                f"AUTO_SELECT_EPS = {repr(auto_select_eps)}",
                content
            )
        if auto_select_eps_method is not _NO_CHANGE:
            content = re.sub(
                r'AUTO_SELECT_EPS_METHOD\s*=\s*"[^"]*"',
                f'AUTO_SELECT_EPS_METHOD = "{auto_select_eps_method}"',
                content
            )
        if eps_max_cost_per_fuel_kg is not _NO_CHANGE:
            val_repr = "None" if eps_max_cost_per_fuel_kg is None else repr(float(eps_max_cost_per_fuel_kg))
            content = re.sub(
                r"EPS_MAX_COST_PER_FUEL_KG\s*=\s*(?:None|[\d.]+)",
                f"EPS_MAX_COST_PER_FUEL_KG = {val_repr}",
                content,
            )
        if auto_threshold is not _NO_CHANGE:
            content = re.sub(
                r"AUTO_THRESHOLD\s*=\s*(?:True|False)",
                f"AUTO_THRESHOLD = {repr(auto_threshold)}",
                content
            )
        if fuel_bias is not _NO_CHANGE:
            content = re.sub(
                r"FUEL_BIAS\s*=\s*[\d.]+",
                f"FUEL_BIAS = {fuel_bias}",
                content
            )
        if ground_events_source is not _NO_CHANGE:
            content = re.sub(
                r'GROUND_EVENTS_SOURCE\s*=\s*"[^"]*"',
                f'GROUND_EVENTS_SOURCE = "{ground_events_source}"',
                content
            )
        if solver_backend is not _NO_CHANGE:
            content = re.sub(
                r'SOLVER_BACKEND\s*=\s*"[^"]*"',
                f'SOLVER_BACKEND = "{solver_backend}"',
                content
            )

        with open(config_path, 'w') as f:
            f.write(content)

        return True
    except Exception as e:
        st.error(f"Failed to save config: {e}")
        return False


with st.sidebar:
    st.markdown(
        """
        <div style='padding:0.6rem 0 0.2rem 0'>
          <div style='font-size:1.15rem;font-weight:700;line-height:1.3'>Optimised Tail Allocation</div>
          <div style='font-size:0.72rem;color:#888;margin-top:3px;line-height:1.4'>
            Minimum-cost MIP assignment<br>Cost and fuel trade-off optimisation<br>Aviation planning decision support
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigation",
        ["Dashboard", "Run Pipeline", "Results Data", "Analysis", "Configuration"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Current configuration summary
    st.subheader("Current Config")
    _eps_method_label = {"frontier": "Validated frontier"}.get(
        getattr(config, 'AUTO_SELECT_EPS_METHOD', 'frontier'), getattr(config, 'AUTO_SELECT_EPS_METHOD', 'Frontier').title()
    )
    _eps_mode = f"auto ({_eps_method_label})" if getattr(config, 'AUTO_SELECT_EPS', False) else (
        f"ε={config.SELECTED_EPS}" if getattr(config, 'SELECTED_EPS', None) else "cost-optimal"
    )
    st.text(f"MTL Date:   {config.DATE_PREFIX}")
    st.text(f"Fuel Price: ${config.FUEL_PRICE}/kg")
    st.text(f"ε mode:     {_eps_mode}")

    # Available results
    available_dates = get_available_dates()
    if available_dates:
        st.markdown("---")
        st.subheader("Recent Results")
        for date in available_dates[:5]:
            st.text(f"• {date}")


# Main content
if page == "Dashboard":
    st.markdown('<p class="main-header">Dashboard</p>', unsafe_allow_html=True)

    # Latest changelog entry as a collapsible "What's New" banner
    import re as _re
    _cl_path = Path(__file__).parent / os.environ.get("CHANGELOG_FILE", "CHANGELOG.md")
    if _cl_path.exists():
        _cl_text = _cl_path.read_text(encoding="utf-8")
        _cl_match = _re.search(
            r"(## \[v([\d.]+)\] - ([\d-]+)\n)(.*?)(?=\n## \[|$)", _cl_text, _re.DOTALL
        )
        if _cl_match:
            _cl_version = _cl_match.group(2)
            _cl_date_raw = _cl_match.group(3)
            _cl_body = _cl_match.group(4).strip()
            try:
                _d = datetime.strptime(_cl_date_raw, "%Y-%m-%d")
                _cl_date_fmt = f"{_d.day} {_d.strftime('%B %Y')}"
            except ValueError:
                _cl_date_fmt = _cl_date_raw
            with st.expander(f"What's new · v{_cl_version} · {_cl_date_fmt}"):
                st.markdown(_cl_body)

    st.markdown(f"Optimisation date: **{config.DATE_PREFIX}**", unsafe_allow_html=True)

    results_df = load_results_data(config.DATE_PREFIX)

    if results_df is not None:

        total_savings = results_df['savings'].sum() if 'savings' in results_df.columns else 0
        changed_routes = results_df['changed'].sum() if 'changed' in results_df.columns else 0
        total_routes = len(results_df)
        fuel_saved = results_df['fuel_delta'].sum() if 'fuel_delta' in results_df.columns else 0

        _fuel_comp = fuel_saved * config.FUEL_PRICE
        fuel_share = _fuel_comp / total_savings * 100 if total_savings > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Saving", f"${total_savings:,.0f}")
        with col2:
            st.metric("Swaps", f"{int(changed_routes)} / {total_routes}")
        with col3:
            st.metric("Fuel Saved", f"{fuel_saved:,.0f} kg")
        with col4:
            st.metric("Fuel Share", f"{fuel_share:.0f}%")

        st.markdown("---")

        col1, _spacer, col2 = st.columns([3, 0.25, 2])

        with col1:
            st.subheader("Savings Breakdown by Base")
            _req = ['base', 'fuel_delta', 'total_fh_rate', 'opti_fh_rate',
                    'total_cycle_rate', 'opti_cycle_rate', 'total_pred_act_hours', 'total_sectors']
            if all(c in results_df.columns for c in _req):
                _bd = results_df[_req].copy()
                _bd['Fuel'] = _bd['fuel_delta'] * config.FUEL_PRICE
                _bd['FH Rate'] = (_bd['total_fh_rate'] - _bd['opti_fh_rate']) * _bd['total_pred_act_hours']
                _bd['Cycle Rate'] = (_bd['total_cycle_rate'] - _bd['opti_cycle_rate']) * _bd['total_sectors']
                _by_base = _bd.groupby('base')[['Fuel', 'FH Rate', 'Cycle Rate']].sum().reset_index()
                _by_base = _by_base.sort_values('Fuel', ascending=False)
                _melted = _by_base.melt(id_vars='base', var_name='Component', value_name='Saving ($)')
                _fig = px.bar(
                    _melted, x='base', y='Saving ($)', color='Component',
                    barmode='stack',
                    color_discrete_map={'Fuel': '#FF6900', 'FH Rate': '#003366', 'Cycle Rate': '#4A90D9'},
                    labels={'base': 'Base', 'Saving ($)': 'Saving ($)'},
                    category_orders={'Component': ['Fuel', 'FH Rate', 'Cycle Rate']},
                )
                _fig.update_layout(
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(t=40),
                )
                st.plotly_chart(_fig, width="stretch")

        with col2:
            st.subheader("Cost-Slack % Applied")
            st.markdown("<br>", unsafe_allow_html=True)
            _sweep_path = f"{config.VOLUME_INTERMEDIATE_DIRECTORY}/{config.DATE_PREFIX}_eps_sweep_summary.json"
            try:
                if _vol_exists(_sweep_path):
                    _sd = json.loads(_vol_read_bytes(_sweep_path).decode('utf-8'))
                    if _sd:
                        _first = next(iter(_sd.values()), {})
                        _auto = getattr(config, 'AUTO_SELECT_EPS', False)
                        # Prefer what the pipeline actually applied (stored in JSON),
                        # fall back to SELECTED_EPS from config, then 0.0
                        _json_auto_eps = (_first.get('auto_selected_eps')
                                          or _first.get('eps_star')
                                          or _first.get('kneedle_eps_star'))
                        _applied_eps = (
                            _json_auto_eps if (_auto and _json_auto_eps is not None)
                            else (getattr(config, 'SELECTED_EPS', None) or 0.0)
                        )
                        _stored_method = _first.get('auto_select_method', 'actual_cost_concave_hull')
                        _method_label = {"actual_cost_concave_hull": "Validated frontier", "marginal_cost_limit": "Marginal-cost limit"}.get(
                            _stored_method, _stored_method.title()
                        )
                        _mode = f"auto · {_method_label}" if _auto else "manual"
                        st.markdown(f"**ε = {float(_applied_eps) * 100:.2f}%** &nbsp;·&nbsp; *{_mode}*",
                                    unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        _eps0 = next((v for k, v in _sd.items() if abs(float(k)) < 1e-9), {})
                        _eps_row = min(_sd.items(),
                                       key=lambda kv: abs(float(kv[0]) - float(_applied_eps)))[1]
                        _extra_fuel_kg = (_eps0.get('total_fuel_cost_kg', 0)
                                          - _eps_row.get('total_fuel_cost_kg', 0))
                        _cost_overhead = (_eps_row.get('total_opti_cost', 0)
                                          - _eps0.get('total_opti_cost', 0))
                        _extra_fuel_usd = (_eps0.get('total_fuel_cost_usd', 0)
                                           - _eps_row.get('total_fuel_cost_usd', 0))
                        st.metric("Extra Fuel Saved vs Cost-Optimal (ε=0) assignments ",
                                  f"+{_extra_fuel_kg:,.0f} kg",
                                  help="Additional fuel saved by applying this ε over the Cost-Optimal (ε=0) assignment")
                        st.metric("Fuel Value of Saving",
                                  f"+${_extra_fuel_usd:,.0f}",
                                  help="Dollar value of the extra fuel saved")
                        st.metric("Cost Overhead vs ε=0",
                                  f"+${_cost_overhead:,.0f}",
                                  help="Extra total cost incurred to achieve the fuel saving")
                    else:
                        st.caption("Sweep data empty.")
                else:
                    st.caption("No ε-sweep data for this date.")
                    st.caption("Run pipeline with `EPS_VALUES` set to see fuel/cost trade-off.")
            except Exception:
                st.caption("Sweep data unavailable.")

    else:
        st.info("No results available for the current date. Run the pipeline to generate results.")

        # Show input data summary instead
        input_data = load_input_data()
        if "flights" in input_data:
            st.subheader("Input Data Summary")
            flights_df = input_data["flights"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(flights_df))
            with col2:
                if 'tailnumber' in flights_df.columns:
                    st.metric("Unique Aircraft", flights_df['tailnumber'].nunique())
            with col3:
                if 'aircrafttype' in flights_df.columns:
                    st.metric("Aircraft Types", flights_df['aircrafttype'].nunique())


elif page == "Run Pipeline":
    st.markdown('<p class="main-header">Run Pipeline</p>', unsafe_allow_html=True)

    st.markdown("""
    The optimization pipeline consists of 6 stages:
    1. **Feature Engineering** - Query raw Databricks data, builds per-flight features and distances fallback logic
    2. **Fuel Prediction** - Predicts fuel burn and airbrone hour using LightGBM model (More info in Changlog v2.0.0 or documentation ml section 6.2)
    3. **LoF Enrichment** - Aggregates per-flight predictions to per-tail LoF level
    4. **Eligibility Filter** - Filters maintenance tails, injects spares
    5. **Assignment Optimisation** - Solves minimum-cost tail assignment via Gurobi or Pyomo MIP (HiGHS)
    6. **Post-Processing** - Decomposes swap cycles, applies threshold, generates swap list and summary
    """)

    st.markdown("---")

    # Pipeline execution options
    col1, col2 = st.columns([2, 1])

    with col1:
        run_mode = st.radio(
            "Execution Mode",
            ["Full Pipeline", "Individual Stages"],
            horizontal=True
        )

    with col2:
        st.info(f"Output will be saved to:\n`{config.VOLUME_OUTPUT_DIRECTORY}/`")

    if run_mode == "Full Pipeline":
        st.markdown("### Run Full Pipeline")

        col1, col2 = st.columns(2)

        with col1:
            run_pipeline_btn = st.button("Run Full Pipeline", type="primary", width="stretch")

        with col2:
            clean_files_btn = st.button(f"Clean {config.DATE_PREFIX}", type="secondary", width="stretch")

        if clean_files_btn:
            st.session_state['confirm_clean'] = True

        if st.session_state.get('confirm_clean'):
            st.warning(
                f"This will permanently delete **all** {config.DATE_PREFIX} files from the Volume — "
                f"intermediate data, output CSVs, ε-sweep results, and summary files. "
                f"This cannot be undone."
            )
            _cc, _cx, _ = st.columns([1, 1, 3])
            with _cc:
                if st.button("Yes, delete all", type="primary", width="stretch", key="confirm_clean_yes"):
                    st.session_state.pop('confirm_clean', None)
                    with st.spinner(f"Cleaning {config.DATE_PREFIX} files..."):
                        success, message = clean_current_date_files()
                    if success:
                        st.success(f" {message}")
                    else:
                        st.error(f" Failed to clean files: {message}")
            with _cx:
                if st.button("Cancel", type="secondary", width="stretch", key="confirm_clean_cancel"):
                    st.session_state.pop('confirm_clean', None)
                    st.rerun()

        if run_pipeline_btn:
            with st.spinner("Fetching data from Databricks SQL..."):
                try:
                    df = query_databricks_data(config.MTL_DATE, config.DATE_PREFIX)
                    csv_bytes = df.to_csv(index=False).encode('utf-8')
                    _vol_write_bytes(
                        f"{config.VOLUME_INPUT_DIRECTORY}/{config.DATE_PREFIX}_Opti_Tail_With_MTOW__APM_and_Seat_Capacity.csv",
                        csv_bytes,
                    )
                    st.success(f"Fetched {len(df)} rows from Databricks")
                except Exception as e:
                    st.error(f"Failed to fetch Databricks data: {e}")
                    st.stop()

            if getattr(config, 'GROUND_EVENTS_SOURCE', 'xls') == 'databricks':
                with st.spinner("Fetching ground events from Databricks..."):
                    try:
                        ge_rows = _fetch_ground_events_app(config.MTL_DATE, config.DATE_PREFIX)
                        st.success(f"Ground events fetched: {ge_rows} rows")
                    except Exception as e:
                        st.error(f"Failed to fetch ground events: {e}")
                        st.stop()

            with st.spinner("Staging input files from Volume to local..."):
                try:
                    missing = _stage_inputs_to_local()
                except Exception as e:
                    st.error(f"Failed to stage inputs: {e}")
                    st.stop()
                if missing:
                    st.error(
                        f"Required input files missing on Volume (under `{config.VOLUME_INPUT_DIRECTORY}/`): "
                        + ", ".join(missing)
                    )
                    st.stop()

            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.container()

            total_stages = len(pipeline)

            for i, (step_name, module) in enumerate(pipeline):
                status_text.text(f"Running: {step_name}...")

                success, message = run_pipeline_stage(step_name, module)

                with log_container:
                    if success:
                        st.success(f" {message}")
                    else:
                        st.error(f" {message}")
                        break

                progress_bar.progress((i + 1) / total_stages)

            if success:
                with st.spinner("Uploading pipeline outputs to Volume..."):
                    try:
                        _stage_outputs_to_volume()
                    except Exception as e:
                        st.error(f"Pipeline ran but failed to upload outputs to Volume: {e}")
                _clear_all_volume_caches()
                status_text.text("Pipeline completed successfully!")
                # Warn if any groups were skipped due to infeasible constraints
                _skipped_vol_path = f"{config.VOLUME_INTERMEDIATE_DIRECTORY}/{config.DATE_PREFIX}_skipped_groups.json"
                if _vol_exists(_skipped_vol_path):
                    try:
                        _skipped = json.loads(_vol_read_bytes(_skipped_vol_path).decode('utf-8'))
                        if _skipped:
                            _names = ", ".join(g['group_index'] for g in _skipped)
                            st.warning(f"{len(_skipped)} group(s) skipped due to infeasible constraints: {_names}")
                    except Exception:
                        pass

                # ε-constraint sweep trade-off table
                _sweep_path = f"{config.VOLUME_INTERMEDIATE_DIRECTORY}/{config.DATE_PREFIX}_eps_sweep_summary.json"
                if _vol_exists(_sweep_path):
                    try:
                        _sweep = json.loads(_vol_read_bytes(_sweep_path).decode('utf-8'))
                        if _sweep:
                            st.subheader("Cost Trade-off Table")
                            _cost_at_eps0 = next(iter(_sweep.values()), {}).get('total_opti_cost', 0)
                            _sweep_rows = []
                            for _eps_str, _s in _sweep.items():
                                _eps_val = float(_eps_str)
                                _fuel_saving_kg = _s.get('baseline_fuel_kg', 0) - _s.get('total_fuel_cost_kg', 0)
                                _cost_tradeoff = _s['total_opti_cost'] - _cost_at_eps0
                                _fuel_saving_usd = _fuel_saving_kg * config.FUEL_PRICE
                                _sweep_rows.append({
                                    'ε': f"{_eps_val*100:.2f}%",
                                    'Total Cost ($)': f"${_s['total_opti_cost']:,.0f}",
                                    'Cost vs ε=0 ($)': f"+${_cost_tradeoff:,.0f}" if _cost_tradeoff > 0 else "$0",
                                    'Fuel Saving vs Baseline (kg)': f"{_fuel_saving_kg:+,.0f}",
                                    'Fuel Saving ($)': f"${_fuel_saving_usd:,.0f}",
                                    'Swaps': _s['n_changed'],
                                })
                            st.dataframe(pd.DataFrame(_sweep_rows), width="stretch", hide_index=True)
                    except Exception:
                        pass

    else:  # Individual Stages
        st.markdown("### Run Individual Stages")

        stages = [
            ("Feature Engineering",     feature_engineering,     "Joins raw Databricks data, builds per-leg features and airway distances"),
            ("Fuel Prediction",         fuel_prediction,         "Predicts fuel burn per leg using pooled LightGBM model"),
            ("LoF Enrichment",          lof_enrichment,          "Aggregates per-leg predictions to per-tail LoF level; joins FH/cycle rates from cost index"),
            ("Eligibility Filter",      eligibility_filter,      "Filters maintenance tails, injects TOPS spare aircraft, identifies LGW-S terminal"),
            ("Assignment Optimisation", assignment_optimisation, "Solves minimum-cost tail assignment via Pyomo MIP (Gurobi/HiGHS) with optional eps-sweep"),
            ("Post-Processing",         post_processing,         "Decomposes independent swap cycles, applies savings threshold, writes swap list and summary"),
        ]

        for stage_name, module, description in stages:
            with st.expander(f"**{stage_name}**", expanded=False):
                st.markdown(description)

                if st.button(f"Run {stage_name}", key=f"run_{stage_name}"):
                    with st.spinner(f"Staging inputs and running {stage_name}..."):
                        try:
                            missing = _stage_inputs_to_local()
                        except Exception as e:
                            st.error(f"Failed to stage inputs: {e}")
                            st.stop()
                        if missing:
                            st.error(
                                f"Required input files missing on Volume (under `{config.VOLUME_INPUT_DIRECTORY}/`): "
                                + ", ".join(missing)
                            )
                            st.stop()
                        success, message = run_pipeline_stage(stage_name, module)
                        if success:
                            try:
                                _stage_outputs_to_volume()
                            except Exception as e:
                                st.warning(f"Stage ran but upload to Volume failed: {e}")
                            _clear_all_volume_caches()

                    if success:
                        st.success(message)
                        if stage_name == "Assignment Optimisation":
                            _skipped_vol_path = f"{config.VOLUME_INTERMEDIATE_DIRECTORY}/{config.DATE_PREFIX}_skipped_groups.json"
                            if _vol_exists(_skipped_vol_path):
                                try:
                                    _skipped = json.loads(_vol_read_bytes(_skipped_vol_path).decode('utf-8'))
                                    if _skipped:
                                        _names = ", ".join(g['group_index'] for g in _skipped)
                                        st.warning(f"{len(_skipped)} group(s) skipped due to infeasible constraints: {_names}")
                                except Exception:
                                    pass
                    else:
                        st.error(message)


elif page == "Results Data":
    st.markdown('<p class="main-header">Results Data</p>', unsafe_allow_html=True)

    data_source = st.selectbox(
        "Select Data Source",
        ["Input Data", "Output Results"],
        index=1
    )

    if data_source == "Input Data":
        input_data = load_input_data()

        if input_data:
            dataset = st.selectbox("Select Dataset", list(input_data.keys()))

            if dataset:
                df = input_data[dataset]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

                # Column filter
                st.subheader("Column Filter")
                selected_columns = st.multiselect(
                    "Select columns to display",
                    df.columns.tolist(),
                    default=df.columns.tolist()[:10]
                )

                # Row filter
                st.subheader("Data Preview")
                n_rows = st.slider("Number of rows to display", 10, 1000, 100)

                st.dataframe(df[selected_columns].head(n_rows), width="stretch")

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download Full Dataset",
                    csv,
                    f"{dataset}.csv",
                    "text/csv"
                )
        else:
            st.warning("No input data found. Please ensure data files are in the input directory.")

    elif data_source == "Intermediate Data":
        available_dates = get_available_dates()
        items = _vol_list(config.VOLUME_INTERMEDIATE_BASE)
        int_dates = sort_date_strings(
            [i.name for i in items if i.is_directory and not i.name.startswith('.')],
            descending=True
        )[:5]

        if int_dates:
            selected_date = st.selectbox("Select Date", int_dates)

            stage = st.selectbox(
                "Select Stage",
                ["feature_engineering", "fuel_prediction", "lof_enrichment",
                 "eligibility_filter", "assignment_optimisation"],
                format_func=lambda x: {
                    "feature_engineering":     "Stage 1: Feature Engineering",
                    "fuel_prediction":         "Stage 2: Fuel Prediction",
                    "lof_enrichment":          "Stage 3: LoF Enrichment",
                    "eligibility_filter":      "Stage 4: Eligibility Filter",
                    "assignment_optimisation": "Stage 5: Assignment Optimisation",
                }.get(x, x)
            )

            df = load_intermediate_data(selected_date, stage)

            if df is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

                st.dataframe(df.head(100), width="stretch")
            else:
                st.warning(f"No intermediate data found for {selected_date} - {stage}")
        else:
            st.info("No intermediate data available. Run the pipeline first.")

    else:
        available_dates = get_available_dates()

        if available_dates:
            selected_date = st.selectbox("Select Date", available_dates)

            tab1, tab2, tab3, tab4 = st.tabs(["Cleaned Results", "Summary Report (TXT)", "Full Results Data (CSV)", "ε-Sweep Trade-off"])

            with tab1:
                st.subheader("Cleaned Results (Filtered by Savings)")
                st.markdown(f"These files contain only changed routes filtered by `MIN_SAVINGS_THRESHOLD` (current: **${config.MIN_SAVINGS_THRESHOLD}**).")
                st.markdown("Adjust threshold in Configuration and re-run pipeline.")

                cleaned_df = load_cleaned_csv(selected_date)

                if cleaned_df is not None and not cleaned_df.empty:
                    feedback = load_feedback(selected_date)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Routes", len(cleaned_df))
                    with col2:
                        st.metric("Bases", cleaned_df['base'].nunique() if 'base' in cleaned_df.columns else "—")
                    with col3:
                        if 'group_index' in cleaned_df.columns:
                            st.metric("Groups", cleaned_df['group_index'].nunique())
                    with col4:
                        if 'aircraftreg' in cleaned_df.columns:
                            st.metric("Aircraft", cleaned_df['aircraftreg'].nunique())

                    st.markdown("---")

                    merged_df = _merge_feedback_columns(cleaned_df, feedback)
                    edited_df = _render_feedback_grid(merged_df, key="grid_all_bases")

                    if st.button("Save Feedback", key="save_all_bases", type="primary"):
                        group_comments = _extract_feedback_from_df(edited_df)
                        feedback["group_comments"] = group_comments
                        save_feedback(selected_date, feedback)
                        st.success("Feedback saved.")

                    csv_data = edited_df.to_csv(index=False)
                    st.download_button(
                        "Download Swap Approval CSV",
                        csv_data,
                        f"{selected_date}_swap_approval.csv",
                        "text/csv",
                    )

                    st.markdown("---")

                    # General Notes section
                    st.subheader(f"General Notes — {selected_date}")
                    general_notes = st.text_area(
                        "Freeform notes",
                        value=feedback.get("general_notes", ""),
                        key=f"general_notes_{selected_date}",
                        height=120,
                    )
                    if st.button("Save Notes", key=f"save_notes_{selected_date}"):
                        feedback["general_notes"] = general_notes
                        save_feedback(selected_date, feedback)
                        st.success("Notes saved.")

                    st.markdown("---")
                else:
                    st.warning("No cleaned CSV found.")
                    st.info("Cleaned results are generated when `MIN_SAVINGS_THRESHOLD` filters groups. "
                        "Set a high value (e.g., 9999) for the initial stage, no cleaned file will be produced. "
                        "Check the Summary Report (TXT) to see group savings, then adjust the threshold and re-run.")

            with tab2:
                summary_txt = load_summary_txt(selected_date)

                if summary_txt:
                    st.subheader("Optimization Summary Report")
                    st.code(summary_txt, language=None)

                    # Download button for txt
                    st.download_button(
                        "Download Summary Report",
                        summary_txt,
                        f"{selected_date}_swap_approval_summary.txt",
                        "text/plain"
                    )
                else:
                    st.warning(f"No summary TXT file found for {selected_date}")

            with tab3:
                df = load_results_data(selected_date)

                if df is not None:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Line of Works", len(df))
                    with col2:
                        st.metric("Reassignments", int(df['changed'].sum()) if 'changed' in df.columns else "N/A")
                    with col3:
                        st.metric("Total Savings", f"${df['savings'].sum():,.2f}" if 'savings' in df.columns else "N/A")
                    with col4:
                        st.metric("Fuel Saved", f"{df['fuel_delta'].sum():,.0f} kg" if 'fuel_delta' in df.columns else "N/A")

                    # Filters
                    # st.subheader("Filters")
                    # col1, col2, col3 = st.columns(3)

                    # with col1:
                    #     if 'base' in df.columns:
                    #         bases = st.multiselect("Base", df['base'].unique().tolist())

                    # with col2:
                    #     if 'aircrafttype' in df.columns:
                    #         aircraft_types = st.multiselect("Aircraft Type", df['aircrafttype'].unique().tolist())

                    # with col3:
                    #     if 'changed' in df.columns:
                    #         show_changed_only = st.checkbox("Show changed routes only")

                    # # Apply filters
                    # filtered_df = df.copy()
                    # if bases:
                    #     filtered_df = filtered_df[filtered_df['base'].isin(bases)]
                    # if aircraft_types:
                    #     filtered_df = filtered_df[filtered_df['aircrafttype'].isin(aircraft_types)]
                    # if show_changed_only and 'changed' in df.columns:
                    #     filtered_df = filtered_df[filtered_df['changed'] == True]

                    st.subheader(f"Results ({len(df)} rows)")
                    st.dataframe(df, width="stretch")

                    # Download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download all results (Unfiltered)",
                        csv,
                        f"{selected_date}_assignment_optimisation.csv",
                        "text/csv"
                    )
                else:
                    st.warning(f"No CSV results found for {selected_date}")

            with tab4:
                st.subheader("Cost Trade-off Table")
                st.markdown(
                    "Each row is a solved daily assignment: minimise fuel under one global total-cost "
                    "budget C ≤ C*(1+ε), then break ties by lower realised cost and fewer swaps."
                )
                _sweep_summary_path = f"{config.VOLUME_INTERMEDIATE_BASE}/{selected_date}/{selected_date}_eps_sweep_summary.json"
                if _vol_exists(_sweep_summary_path):
                    try:
                        _sweep_data = json.loads(_vol_read_bytes(_sweep_summary_path).decode('utf-8'))
                        if _sweep_data:
                            _eps0_tab = min(_sweep_data.items(), key=lambda item: abs(float(item[0])))[1]
                            _cost_at_eps0_tab = _eps0_tab.get('total_opti_cost', 0)
                            _fuel_at_eps0_tab = _eps0_tab.get('total_fuel_cost_kg', 0)
                            _tbl = []
                            for _eps_str, _s in _sweep_data.items():
                                _eps_val = float(_eps_str)
                                _fuel_saving = _s.get('baseline_fuel_kg', 0) - _s.get('total_fuel_cost_kg', 0)
                                _cost_tradeoff = _s['total_opti_cost'] - _cost_at_eps0_tab
                                _fuel_saving_usd = _fuel_saving * config.FUEL_PRICE
                                _tbl.append({
                                    'ε': f"{_eps_val*100:.2f}%",
                                    'Total Optimised Cost ($)': f"${_s['total_opti_cost']:,.0f}",
                                    'Cost vs ε=0 ($)': f"+${_cost_tradeoff:,.0f}" if _cost_tradeoff > 0 else "$0",
                                    'Fuel Saving vs Baseline (kg)': f"{_fuel_saving:+,.0f}",
                                    'Fuel Saving ($)': f"${_fuel_saving_usd:,.0f}",
                                    'Swaps': _s['n_changed'],
                                    'Extra Fuel Saved vs ε=0 (kg)': f"{(_fuel_at_eps0_tab - _s.get('total_fuel_cost_kg', 0)):+,.0f}",
                                    'Pareto': bool(_s.get('is_pareto', True)),
                                    'Concave Hull': bool(_s.get('is_concave_hull', False)),
                                })
                            st.dataframe(pd.DataFrame(_tbl), width="stretch", hide_index=True)

                            # Show auto-selected ε info
                            _fv = next(iter(_sweep_data.values()), {})
                            _kapplied = _fv.get('auto_selected_eps')
                            _fv_method = _fv.get('auto_select_method', 'actual_cost_concave_hull')
                            _fv_method_label = {"actual_cost_concave_hull": "Validated frontier", "marginal_cost_limit": "Marginal-cost limit"}.get(
                                _fv_method, _fv_method.title()
                            )
                            if _kapplied is not None:
                                st.success(
                                    f"{_fv_method_label}: solved ε = **{_kapplied*100:.2f}%** applied. "
                                    f"{_fv.get('selection_reason', '')}"
                                )
                            elif _fv.get('selection_reason'):
                                st.info(
                                    "No defensible interior knee was selected; the cost-optimal assignment was retained. "
                                    f"{_fv.get('selection_reason')}"
                                )
                            elif _kapplied is not None:
                                st.info(f"Applied ε = {_kapplied*100:.2f}% (manual override)")

                            # Selection geometry: realised dollars, not epsilon, on x-axis.
                            _eps0_chart = min(_sweep_data.items(), key=lambda item: abs(float(item[0])))[1]
                            _fuel_at_0_chart = _eps0_chart.get('total_fuel_cost_kg', 0)
                            _cost_at_0_chart = _eps0_chart.get('total_opti_cost', 0)
                            _chart_df = pd.DataFrame({
                                'ε (%)': [float(k)*100 for k in _sweep_data],
                                'Extra Fuel Saved vs ε=0 (kg)': [
                                    _fuel_at_0_chart - v['total_fuel_cost_kg']
                                    for v in _sweep_data.values()
                                ],
                                'Cost vs ε=0 ($)': [
                                    v['total_opti_cost'] - _cost_at_0_chart
                                    for v in _sweep_data.values()
                                ],
                            })
                            fig_sweep = px.line(
                                _chart_df, x='Cost vs ε=0 ($)', y='Extra Fuel Saved vs ε=0 (kg)',
                                title="Solved Global Cost/Fuel Points",
                                labels={
                                    'Cost vs ε=0 ($)': 'Realised extra total cost ($)',
                                    'Extra Fuel Saved vs ε=0 (kg)': 'Additional fuel saved (kg)',
                                },
                                markers=True,
                                hover_data={'ε (%)': ':.3f'},
                            )
                            fig_sweep.update_traces(textposition='top center')
                            st.plotly_chart(fig_sweep, width="stretch")

                            # Apply selected ε as canonical assignment
                            if selected_date == config.DATE_PREFIX:
                                st.markdown("---")
                                st.subheader("Apply Selected ε as Assignment")
                                st.caption(
                                    "Replaces the standard cost-optimal results CSV with the chosen ε solution "
                                    "and re-runs post-processing only. "
                                )
                                _apply_eps = st.selectbox(
                                    "Choose ε to apply",
                                    options=list(_sweep_data.keys()),
                                    format_func=lambda k: f"ε = {float(k)*100:.2f}%",
                                    key=f"apply_eps_{selected_date}",
                                )
                                if st.button("Apply and re-run post-processing", type="primary", key=f"apply_btn_{selected_date}"):
                                    with st.spinner(f"Applying ε={_apply_eps} and re-running Post-Processing..."):
                                        try:
                                            _sweep_csv_path = f"{config.VOLUME_INTERMEDIATE_BASE}/{selected_date}/{selected_date}_eps_sweep_results.csv"
                                            _sweep_full = pd.read_csv(io.BytesIO(_vol_read_bytes(_sweep_csv_path)))
                                            _chosen = _sweep_full[
                                                np.isclose(_sweep_full['epsilon'], float(_apply_eps))
                                            ].drop(columns=['epsilon'])
                                            if _chosen.empty:
                                                st.error(f"No rows found for ε={_apply_eps} in sweep CSV.")
                                            else:
                                                _std_path = f"{config.VOLUME_INTERMEDIATE_BASE}/{selected_date}/{selected_date}_assignment_optimisation.csv"
                                                _vol_write_bytes(_std_path, _chosen.to_csv(index=False).encode('utf-8'))

                                                # Force re-stage of the freshly-written assignment CSV (delete local copy first)
                                                _local_algo = os.path.join(config.INTERMEDIATE_DIRECTORY, f"{selected_date}_assignment_optimisation.csv")
                                                if os.path.exists(_local_algo):
                                                    os.remove(_local_algo)

                                                missing = _stage_inputs_to_local()
                                                if missing:
                                                    st.error(f"Missing inputs: {missing}")
                                                else:
                                                    success, message = run_pipeline_stage("Post-Processing", post_processing)
                                                    if success:
                                                        _stage_outputs_to_volume()
                                                        _clear_all_volume_caches()
                                                        st.success(
                                                            f"Applied ε={_apply_eps}. "
                                                            f"All results data now reflect this assignment."
                                                        )
                                                        st.rerun()
                                                    else:
                                                        st.error(f"Post-Processing failed: {message}")
                                        except Exception as e:
                                            st.error(f"Failed to apply: {e}")
                            else:
                                st.markdown("---")
                                st.info(
                                    f"Apply is only available for the active pipeline date "
                                    f"(`{config.DATE_PREFIX}`). Switch `MTL_DATE` in Config to apply ε for a different date."
                                )
                        else:
                            st.info("Sweep summary is empty.")
                    except Exception as e:
                        st.warning(f"Could not load sweep summary: {e}")
                else:
                    st.info("No ε-sweep results found for this date. Re-run the pipeline with EPS_VALUES set.")

        else:
            st.info("No output results available. Run the pipeline first.")

elif page == "Analysis":
    st.markdown('<p class="main-header">Analysis</p>', unsafe_allow_html=True)

    available_dates = get_available_dates()

    if not available_dates:
        st.info("No results available. Run the pipeline first to generate results.")
    else:
        selected_date = st.selectbox("Select Date for Analysis", available_dates)

        df = load_results_data(selected_date)

        if df is not None:
            st.subheader("Cost Matrix")
            st.markdown(
                "Reconstructed n×n cost matrix for a selected group. "
                "**Black border** = original assignment (diagonal). "
                "**Blue border** = optimal assignment (solver). "
                "**Grey border** = no change. "
                "Cell colour: green = cheaper, red = more expensive."
            )

            if 'group_index' in df.columns:
                # Sort groups by total savings descending
                group_savings = (df.groupby('group_index')['savings'].sum()
                                 .sort_values(ascending=False))
                groups = group_savings.index.tolist()
                group_labels = [f"{g}  (${group_savings[g]:,.0f})" for g in groups]

                selected_idx = st.selectbox(
                    "Select Group",
                    range(len(groups)),
                    format_func=lambda i: group_labels[i],
                    key="cost_matrix_group",
                )
                selected_group = groups[selected_idx]

                group_df = df[df['group_index'] == selected_group]
                n_group = len(group_df)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Group Size (n)", n_group)
                with col2:
                    st.metric("Current Cost", f"${group_df['current_cost'].sum():,.2f}")
                with col3:
                    st.metric("Savings", f"${group_df['savings'].sum():,.2f}")

                required_cols = {
                    'avg_perf_corr', 'total_fh_rate', 'total_cycle_rate',
                    'total_baseline_fuel', 'total_pred_act_hours', 'total_sectors',
                    'opti_aircraftreg'
                }
                if required_cols.issubset(df.columns):
                    fig = render_cost_matrix_heatmap(group_df)
                    st.plotly_chart(fig, width="stretch")
                else:
                    missing = required_cols - set(df.columns)
                    st.warning(f"Full results CSV missing columns needed for reconstruction: {missing}")


elif page == "Configuration":
    st.markdown('<p class="main-header">Configuration</p>', unsafe_allow_html=True)

    st.markdown("Edit configuration values below. Changes will be saved to `config.py`.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Date Settings")
        current_mtl_date = datetime.strptime(config.MTL_DATE, '%d/%m/%Y').date()
        new_mtl_date = st.date_input(
            "MTL Date",
            value=current_mtl_date,
            format="DD/MM/YYYY"
        )

        derived_prefix = new_mtl_date.strftime('%d%b%Y').upper()
        st.text(f"Date Prefix (derived from MTL Date): {derived_prefix}")

        st.subheader("Variables")
        new_fuel_price = st.number_input(
            "Fuel Price ($/kg)",
            value=config.FUEL_PRICE,
            format="%.3f",
            step=0.001,
            min_value=0.0
        )
        new_filter_top_groups = st.number_input(
            "Filter Top Groups Index",
            value=config.FILTER_BY_TOP_GROUPS_INDEX,
            step=1,
            min_value=1
        )



        st.subheader("Directories")
        st.text(f"ML Models: {config.ML_MODELS_DIRECTORY}")

        st.markdown("**Manual Exclusions**")
        st.caption("Enter one aircraft registration per line:")
        current_manual = "\n".join(config.MANUAL_DROP_AIRCRAFTREG)
        new_manual_text = st.text_area(
            "Manual Exclusions",
            value=current_manual,
            height=120,
            key="manual_exclusions",
            label_visibility="collapsed"
        )
        new_manual_exclusions = [
            reg.strip().upper() for reg in new_manual_text.strip().split("\n") if reg.strip()
        ]

    with col2:
        st.markdown("**Cost Trade-off Sweep**")
        st.caption("ε values for fuel/cost trade-off. Empty = disabled.")
        _default_eps = ", ".join(str(e) for e in config.EPS_VALUES)
        new_eps_text = st.text_input("EPS_VALUES", value=_default_eps, key="eps_values")
        try:
            new_eps_values = [float(x.strip()) for x in new_eps_text.split(",") if x.strip()]
        except ValueError:
            st.warning("Invalid EPS_VALUES — enter comma-separated numbers")
            new_eps_values = config.EPS_VALUES

        st.markdown("**MIP Solver**")
        _solver_opts = {
            "gurobi": "Gurobi (requires gurobi-secrets)",
            "highs":  "HiGHS — Pyomo (open-source)",
        }
        _current_solver = getattr(config, 'SOLVER_BACKEND', 'highs')
        try:
            _solver_idx = list(_solver_opts.keys()).index(_current_solver)
        except ValueError:
            _solver_idx = 0
        _new_solver = st.selectbox(
            "SOLVER_BACKEND",
            options=list(_solver_opts.keys()),
            format_func=lambda k: _solver_opts[k],
            index=_solver_idx,
            key="solver_backend_select",
        )
        new_solver_gurobi = (_new_solver == "gurobi")

        st.markdown("**Auto-select ε**")
        st.caption(
            "Validates globally solved cost/fuel points, removes duplicates and dominated "
            "solutions, and selects an actual upper-concave-hull vertex. If no prominent "
            "knee exists, the cost-optimal assignment is retained."
        )
        new_auto_select_eps = st.toggle(
            "AUTO_SELECT_EPS",
            value=getattr(config, 'AUTO_SELECT_EPS', True),
            key="auto_select_eps",
        )
        if new_auto_select_eps:
            _method_opts = {
                "frontier": "Validated frontier — actual cost/fuel concave hull",
            }
            _current_method = getattr(config, 'AUTO_SELECT_EPS_METHOD', 'frontier')
            try:
                _method_idx = list(_method_opts.keys()).index(_current_method)
            except ValueError:
                _method_idx = 0
            new_auto_select_eps_method = st.selectbox(
                "AUTO_SELECT_EPS_METHOD",
                options=list(_method_opts.keys()),
                format_func=lambda k: _method_opts[k],
                index=_method_idx,
                key="auto_select_eps_method",
            )
            _current_wtp = getattr(config, 'EPS_MAX_COST_PER_FUEL_KG', None)
            _use_wtp = st.toggle(
                "Use an explicit maximum extra cost per kg saved",
                value=_current_wtp is not None,
                key="eps_use_wtp",
                help=(
                    "Selects the last concave-frontier segment whose incremental extra total cost "
                    "per additional kg of fuel saved is within this limit."
                ),
            )
            if _use_wtp:
                new_eps_max_cost_per_fuel_kg = st.number_input(
                    "Maximum incremental cost ($/kg)",
                    min_value=0.0,
                    value=float(_current_wtp if _current_wtp is not None else 10.0),
                    step=0.5,
                    key="eps_max_cost_per_fuel_kg",
                )
            else:
                new_eps_max_cost_per_fuel_kg = None
            # Show the actual solved point from the last run if available.
            _sweep_path = f"{config.VOLUME_INTERMEDIATE_DIRECTORY}/{config.DATE_PREFIX}_eps_sweep_summary.json"
            if _vol_exists(_sweep_path):
                try:
                    _sd = json.loads(_vol_read_bytes(_sweep_path).decode('utf-8'))
                    _fv = next(iter(_sd.values()), {})
                    _applied = _fv.get('auto_selected_eps')
                    _run_method = _fv.get('auto_select_method', 'actual_cost_concave_hull')
                    _run_label = {"actual_cost_concave_hull": "Validated frontier", "marginal_cost_limit": "Marginal-cost limit"}.get(
                        _run_method, _run_method.title()
                    )
                    if _applied is not None:
                        st.info(
                            f"Last run ({_run_label}): solved ε = **{_applied*100:.2f}%** applied"
                        )
                    else:
                        st.info(f"Last run: no defensible knee; cost-optimal assignment retained. {_fv.get('selection_reason', '')}")
                except Exception:
                    pass
            new_selected_eps = _NO_CHANGE  # auto mode — don't overwrite SELECTED_EPS
        else:
            new_auto_select_eps_method = getattr(config, 'AUTO_SELECT_EPS_METHOD', 'frontier')
            new_eps_max_cost_per_fuel_kg = getattr(config, 'EPS_MAX_COST_PER_FUEL_KG', None)
            st.caption("Manual ε — choose from the EPS_VALUES preset list:")
            _eps_opts = [None] + list(new_eps_values)

            def _fmt_eps_opt(x):
                return "None (cost-optimal)" if x is None else f"ε = {x:.4f} ({x*100:.2f}%)"

            try:
                _sel_idx = _eps_opts.index(config.SELECTED_EPS)
            except ValueError:
                _sel_idx = 0
            new_selected_eps = st.selectbox(
                "SELECTED_EPS",
                options=_eps_opts,
                format_func=_fmt_eps_opt,
                index=_sel_idx,
                key="selected_eps_manual",
            )

        st.markdown("---")
        st.markdown("**Savings Threshold**")
        new_auto_threshold = st.toggle(
            "AUTO_THRESHOLD",
            value=getattr(config, 'AUTO_THRESHOLD', True),
            key="auto_threshold",
        )
        if not new_auto_threshold:
            new_min_ratio = st.number_input(
                "Min Savings Threshold ($)",
                value=config.MIN_SAVINGS_THRESHOLD,
                step=1,
                min_value=0,
                key="min_savings_manual",
            )
        else:
            new_min_ratio = config.MIN_SAVINGS_THRESHOLD
            st.caption(f"Manual threshold (fallback): ${config.MIN_SAVINGS_THRESHOLD}. Auto-select ε will override this.")

        st.markdown("**Fuel Bias**")
        st.caption("Weighting bias on fuel value in composite benefit scoring.")
        new_fuel_bias = st.number_input(
            "FUEL_BIAS",
            value=float(getattr(config, 'FUEL_BIAS', 1.5)),
            format="%.2f",
            step=0.1,
            min_value=1.0,
            max_value=5.0,
            key="fuel_bias",
        )

    st.markdown("---")

    # Ground Events File Upload Section
    st.subheader("Ground Events File")

    new_ground_events_source = st.toggle(
        "Auto-ingest ground events from Databricks TOPS Data",
        value=(getattr(config, 'GROUND_EVENTS_SOURCE', 'xls') == 'databricks'),
        key="ground_events_source_toggle",
        help="ON = query silver_stream_tops.ground_activity_details. "
             "OFF = use manually uploaded Ground Event file (.xls or .xlsx).",
    )

    if new_ground_events_source:
        st.info(
            "Ground events will be now fetched automatically from "
            "`silver_stream_tops.ground_activity_details` on each pipeline run. "
        )
    else:
        st.info("Please upload the most updated ground event (.xls or .xlsx) before use")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.text(f"Current file: {config.GROUNDEVENTS_FILE}")

            current_file_path = f"{config.VOLUME_INPUT_DIRECTORY}/{config.GROUNDEVENTS_FILE}"
            if _vol_exists(current_file_path):
                st.success("File exists in input folder")
            else:
                st.warning("File not found in input folder")

        with col2:
            uploaded_groundevents = st.file_uploader(
                "Upload new Ground Events file",
                type=["xls", "xlsx"],
                help="Upload the TOPS Ground Events Maintenance file (.xls or .xlsx)"
            )

            if uploaded_groundevents is not None:
                if st.button("Save Uploaded File", type="primary"):
                    success, result = upload_groundevents_file(uploaded_groundevents)
                    if success:
                        st.success(f"File '{result}' saved successfully! Restart app to apply.")
                        st.rerun()
                    else:
                        st.error(f"Failed to save file: {result}")

    new_lease_aircraft = list(config.LEASE_AIRCRAFTREG)

    st.markdown("---")

    st.subheader("Route Constraints")
    st.markdown(
        "These lists define which aircraft are eligible or prohibited for specific route types. "
    )

    with st.expander("IRIS Fleet — UK", expanded=False):
        new_iris_uk_text = st.text_area(
            "IRIS UK Fleet", value="\n".join(config.IRIS_UK_AIRCRAFTREG),
            height=120, key="iris_uk", label_visibility="collapsed"
        )
        new_iris_uk_aircraft = [
            reg.strip().upper() for reg in new_iris_uk_text.strip().split("\n") if reg.strip()
        ]

    with st.expander("IRIS Fleet — EU", expanded=False):
        new_iris_eu_text = st.text_area(
            "IRIS EU Fleet", value="\n".join(config.IRIS_EU_AIRCRAFTREG),
            height=120, key="iris_eu", label_visibility="collapsed"
        )
        new_iris_eu_aircraft = [
            reg.strip().upper() for reg in new_iris_eu_text.strip().split("\n") if reg.strip()
        ]

    with st.expander("Cape Verde IATA Codes", expanded=False):
        new_cv_text = st.text_area(
            "Cape Verde IATAs", value="\n".join(config.CAPE_VERDE_IATA),
            height=80, key="cv_iata", label_visibility="collapsed"
        )
        new_cape_verde_iata = [x.strip().upper() for x in new_cv_text.strip().split("\n") if x.strip()]

    with st.expander("Cyprus IATA Codes", expanded=False):
        new_cy_iata_text = st.text_area(
            "Cyprus IATAs", value="\n".join(config.CYPRUS_IATA),
            height=80, key="cyprus_iata", label_visibility="collapsed"
        )
        new_cyprus_iata = [x.strip().upper() for x in new_cy_iata_text.strip().split("\n") if x.strip()]

    with st.expander("KEF IATA Codes", expanded=False):
        new_kef_text = st.text_area(
            "KEF IATAs", value="\n".join(config.KEF_IATA),
            height=60, key="kef_iata", label_visibility="collapsed"
        )
        new_kef_iata = [x.strip().upper() for x in new_kef_text.strip().split("\n") if x.strip()]

    with st.expander("Cyprus Prohibited Aircraft", expanded=False):
        new_cy_text = st.text_area(
            "Cyprus Prohibited", value="\n".join(config.CYPRUS_PROHIBITED),
            height=160, key="cyprus_prohibited", label_visibility="collapsed"
        )
        new_cyprus_prohibited = [r.strip().upper() for r in new_cy_text.strip().split("\n") if r.strip()]

    with st.expander("Autoland Aircraft", expanded=False):
        new_auto_text = st.text_area(
            "Autoland Aircraft", value="\n".join(config.AUTOLAND_AIRCRAFTREG),
            height=160, key="autoland_aircraft", label_visibility="collapsed"
        )
        new_autoland_aircraft = [r.strip().upper() for r in new_auto_text.strip().split("\n") if r.strip()]

    st.markdown("---")

    # Check if there are changes
    new_mtl_date_str = new_mtl_date.strftime('%d/%m/%Y')
    config_changed = (
        new_mtl_date_str != config.MTL_DATE or
        new_fuel_price != config.FUEL_PRICE or
        new_min_ratio != config.MIN_SAVINGS_THRESHOLD or
        new_filter_top_groups != config.FILTER_BY_TOP_GROUPS_INDEX or
        new_lease_aircraft != config.LEASE_AIRCRAFTREG or
        new_iris_uk_aircraft != config.IRIS_UK_AIRCRAFTREG or
        new_iris_eu_aircraft != config.IRIS_EU_AIRCRAFTREG or
        new_manual_exclusions != config.MANUAL_DROP_AIRCRAFTREG or
        new_cape_verde_iata != config.CAPE_VERDE_IATA or
        new_cyprus_iata != config.CYPRUS_IATA or
        new_kef_iata != config.KEF_IATA or
        new_cyprus_prohibited != config.CYPRUS_PROHIBITED or
        new_autoland_aircraft != config.AUTOLAND_AIRCRAFTREG or
        new_eps_values != config.EPS_VALUES or
        new_auto_select_eps != getattr(config, 'AUTO_SELECT_EPS', True) or
        new_auto_select_eps_method != getattr(config, 'AUTO_SELECT_EPS_METHOD', 'frontier') or
        new_eps_max_cost_per_fuel_kg != getattr(config, 'EPS_MAX_COST_PER_FUEL_KG', None) or
        (not new_auto_select_eps and new_selected_eps is not _NO_CHANGE and new_selected_eps != config.SELECTED_EPS) or
        new_auto_threshold != getattr(config, 'AUTO_THRESHOLD', True) or
        new_fuel_bias != float(getattr(config, 'FUEL_BIAS', 1.5)) or
        new_ground_events_source != (getattr(config, 'GROUND_EVENTS_SOURCE', 'xls') == 'databricks') or
        new_solver_gurobi != (getattr(config, 'SOLVER_BACKEND', 'highs') == 'gurobi')
    )

    if config_changed:
        st.warning("You have unsaved changes.")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Save Changes", type="primary"):
            if not config_changed:
                st.info("No changes to save.")
            elif save_config_to_file(
                mtl_date=new_mtl_date_str,
                fuel_price=new_fuel_price,
                min_ratio_threshold=new_min_ratio,
                filter_top_groups=new_filter_top_groups,
                groundevents_file=config.GROUNDEVENTS_FILE,
                lease_aircraft=new_lease_aircraft,
                iris_uk_aircraft=new_iris_uk_aircraft,
                iris_eu_aircraft=new_iris_eu_aircraft,
                manual_exclusions=new_manual_exclusions,
                cyprus_prohibited=new_cyprus_prohibited,
                cyprus_iata=new_cyprus_iata,
                cape_verde_iata=new_cape_verde_iata,
                autoland_aircraft=new_autoland_aircraft,
                kef_iata=new_kef_iata,
                eps_values=new_eps_values,
                selected_eps=new_selected_eps,
                auto_select_eps=new_auto_select_eps,
                auto_select_eps_method=new_auto_select_eps_method,
                eps_max_cost_per_fuel_kg=new_eps_max_cost_per_fuel_kg,
                auto_threshold=new_auto_threshold,
                fuel_bias=new_fuel_bias,
                ground_events_source="databricks" if new_ground_events_source else "xls",
                solver_backend="gurobi" if new_solver_gurobi else "highs",
            ):
                # Reload config and all pipeline stage modules so that the
                # current process picks up the new values (e.g. FUEL_PRICE)
                # without requiring an app restart. run_pipeline loads stages
                # at import time via importlib, so we reload it too which
                # re-executes each stage's module-level `from config import ...`.
                import importlib, sys
                importlib.reload(sys.modules["config"])
                importlib.reload(sys.modules["run_pipeline"])
                # Re-run Gurobi secret injection with the new backend setting.
                _load_gurobi_secrets_once.clear()
                _load_gurobi_secrets_once()
                st.success("Configuration saved.")
                st.rerun()

    st.markdown("---")
    st.info("To change directory paths, please contact me directly.")


# Footer
st.markdown("---")

with st.expander("Changelog"):
    changelog_path = Path(__file__).parent / os.environ.get("CHANGELOG_FILE", "CHANGELOG.md")
    if changelog_path.exists():
        st.markdown(changelog_path.read_text(encoding="utf-8"))
    else:
        st.info("CHANGELOG.md not found.")

st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "Optimised Tail Allocation &nbsp;·&nbsp; Sanitised portfolio edition"
    "</div>",
    unsafe_allow_html=True
)
