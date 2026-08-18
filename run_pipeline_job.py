"""
Daily Job entrypoint — wraps run_pipeline.py for Databricks Lakeflow Jobs.

Forces MTL_DATE = today + 3 days (Europe/London) via env var, then:
1. Fetches the daily input CSV from Databricks SQL Warehouse (same query as app.py)
2. Mirrors the Volume <-> /tmp staging that app.py does for the Streamlit deployment,
   since serverless Job compute can't be relied on for FUSE /Volumes access.

Run locally:  python run_pipeline_job.py
"""
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from src.queries import daily_input_query, ground_events_query
from src.volume_io import make_ws_client, vol_read_bytes, vol_write_bytes


def _get_run_dates() -> list[str]:
    """Return ordered list of dates for this job invocation.

    Priority 1: MTL_DATE env var already set (single date — scheduler or external caller).
    Priority 2: MTL_DATE_OVERRIDE in config.py non-empty (single date — manual test).
    Priority 3: Day-of-week logic (Europe/London):
        Friday  → D+3, D+4, D+5  (covers weekend + Monday before next scheduled run)
        Mon–Thu → D+3 only
    """
    existing = os.environ.get('MTL_DATE', '').strip()
    if existing:
        print(f'[Job] MTL_DATE from environment: {existing}')
        return [existing]

    import re
    _cfg_path = Path(__file__).with_name('config.py')
    try:
        with open(_cfg_path, encoding='utf-8') as _f:
            _content = _f.read()
        _m = re.search(r"MTL_DATE_OVERRIDE\s*=\s*['\"]([^'\"]*)['\"]", _content)
        if _m:
            override = _m.group(1).strip()
            if override:
                print(f'[Job] MTL_DATE_OVERRIDE={override} from config.py')
                return [override]
    except Exception as exc:
        print(f'[Job] could not read config.py for override: {exc}', file=sys.stderr)

    today = datetime.now(ZoneInfo('Europe/London'))
    offsets = [3, 4, 5] if today.weekday() == 4 else [3]  # 4 = Friday
    dates = [(today.date() + timedelta(days=off)).strftime('%d/%m/%Y') for off in offsets]
    print(f'[Job] {today.strftime("%A")} — running for: {", ".join(dates)}')
    return dates


def _get_ws_client():
    return make_ws_client()


def _get_sql_token() -> str:
    """Get PAT for SQL Warehouse: environment first, then a configured secret scope."""
    token = os.environ.get('DATABRICKS_TOKEN', '').strip()
    if token:
        return token
    try:
        import base64
        from databricks.sdk import WorkspaceClient
        ws = WorkspaceClient()
        scope = os.environ.get('DATABRICKS_SECRET_SCOPE', 'aviation-optimiser-secrets')
        resp = ws.secrets.get_secret(scope=scope, key='databricks_token')
        if resp and resp.value:
            return base64.b64decode(resp.value).decode('utf-8')
    except Exception as e:
        print(f'[Job] Warning: could not read token from secret scope: {e}', file=sys.stderr)
    return ''


def _open_sql_connection(cfg):
    """Return an open Databricks SQL connection using PAT or OAuth M2M."""
    from databricks import sql
    server_hostname = os.environ.get('DATABRICKS_HOST', 'your-workspace.cloud.databricks.com')
    if server_hostname.startswith('https://'):
        server_hostname = server_hostname[len('https://'):]
    http_path = os.environ.get('DATABRICKS_HTTP_PATH', '/sql/1.0/warehouses/<warehouse-id>')
    token = _get_sql_token()
    if token:
        return sql.connect(server_hostname=server_hostname, http_path=http_path, access_token=token)
    from databricks.sdk.core import Config, oauth_service_principal
    def credentials_provider():
        sdk_cfg = Config(
            host=f'https://{server_hostname}',
            client_id=os.environ.get('DATABRICKS_CLIENT_ID'),
            client_secret=os.environ.get('DATABRICKS_CLIENT_SECRET'),
        )
        return oauth_service_principal(sdk_cfg)
    return sql.connect(server_hostname=server_hostname, http_path=http_path,
                       credentials_provider=credentials_provider)


def _fetch_ground_events(cfg) -> None:
    """Fetch ground events from silver_stream_tops.ground_activity_details and write to INPUT_DIRECTORY."""
    import pandas as pd
    date_obj = datetime.strptime(cfg.MTL_DATE, '%d/%m/%Y')
    year, month, day = date_obj.year, date_obj.month, date_obj.day

    connection = _open_sql_connection(cfg)
    query = ground_events_query(year, month, day)
    with connection.cursor() as cursor:
        cursor.execute(query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(result, columns=columns)
    connection.close()

    out_path = f'{cfg.INPUT_DIRECTORY}/{cfg.DATE_PREFIX}_ground_events_databricks.csv'
    df.to_csv(out_path, index=False)
    print(f'[Job] Ground events fetch: {len(df)} rows -> {out_path}')


def _fetch_sql_input(client, cfg) -> None:
    """Fetch daily input CSV from Databricks SQL Warehouse and upload to Volume."""
    import pandas as pd

    date_obj = datetime.strptime(cfg.MTL_DATE, '%d/%m/%Y')
    year, month, day = date_obj.year, date_obj.month, date_obj.day

    connection = _open_sql_connection(cfg)
    query = daily_input_query(year, month, day)

    with connection.cursor() as cursor:
        cursor.execute(query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(result, columns=columns)
    connection.close()

    csv_bytes = df.to_csv(index=False).encode('utf-8')
    vol_path = f'{cfg.VOLUME_INPUT_DIRECTORY}/{cfg.DATE_PREFIX}_Opti_Tail_With_MTOW__APM_and_Seat_Capacity.csv'
    vol_write_bytes(client, vol_path, csv_bytes)
    print(f'[Job] SQL fetch: {cfg.DATE_PREFIX}_Opti_Tail_With_MTOW__APM_and_Seat_Capacity.csv uploaded ({len(df)} rows)')


def _load_gurobi_secrets(client) -> None:
    """Read Gurobi CM credentials from secret scope and set as env vars."""
    import base64
    keys = {
        "GUROBI_MANAGER":   "gurobi_manager",
        "GUROBI_ACCESS_ID": "gurobi_access_id",
        "GUROBI_SECRET":    "gurobi_secret",
        "GUROBI_APP_NAME":  "gurobi_app_name",
    }
    for env_var, secret_key in keys.items():
        try:
            scope = os.environ.get("DATABRICKS_SECRET_SCOPE", "aviation-optimiser-secrets")
            resp = client.secrets.get_secret(scope=scope, key=secret_key)
            if resp and resp.value:
                os.environ[env_var] = base64.b64decode(resp.value).decode("utf-8")
        except Exception as e:
            print(f"[Job] Warning: could not load secret {secret_key}: {e}", file=sys.stderr)
    os.environ.setdefault("GUROBI_GROUP", "default")


def _stage_models(client, cfg) -> None:
    """Stage ML model files from the Volume into the local model dir.
    Job compute can't be relied on for FUSE /Volumes, and Stage 2 loads models
    via open()/joblib.load() on a real path. Skip when using a local ./model."""
    if cfg.ML_MODELS_DIRECTORY != cfg.ML_MODELS_STAGING_DIRECTORY:
        return
    os.makedirs(cfg.ML_MODELS_DIRECTORY, exist_ok=True)
    from src.volume_io import vol_list
    for item in vol_list(client, cfg.VOLUME_MODEL_DIRECTORY):
        if item.is_directory:
            continue
        local_path = os.path.join(cfg.ML_MODELS_DIRECTORY, item.name)
        if os.path.exists(local_path):
            continue
        with open(local_path, 'wb') as f:
            f.write(vol_read_bytes(client, item.path))


def _stage_inputs(client, cfg) -> list[str]:
    os.makedirs(cfg.INPUT_DIRECTORY, exist_ok=True)
    os.makedirs(cfg.INTERMEDIATE_DIRECTORY, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIRECTORY, exist_ok=True)

    required = [
        f'{cfg.DATE_PREFIX}_Opti_Tail_With_MTOW__APM_and_Seat_Capacity.csv',
        'cost_index.csv',
    ]
    ground_events_source = getattr(cfg, 'GROUND_EVENTS_SOURCE', 'xls')
    if ground_events_source == 'xls':
        required.append(cfg.GROUNDEVENTS_FILE)

    missing = []
    for name in required:
        vol_path = f'{cfg.VOLUME_INPUT_DIRECTORY}/{name}'
        local_path = f'{cfg.INPUT_DIRECTORY}/{name}'
        try:
            data = vol_read_bytes(client, vol_path)
            with open(local_path, 'wb') as f:
                f.write(data)
        except Exception as e:
            missing.append(f'{name} ({e})')

    if ground_events_source == 'databricks':
        try:
            _fetch_ground_events(cfg)
        except Exception as e:
            missing.append(f'ground_events_databricks ({e})')

    return missing


def _stage_outputs(client, cfg) -> None:
    for local_dir, vol_dir in [
        (cfg.INTERMEDIATE_DIRECTORY, cfg.VOLUME_INTERMEDIATE_DIRECTORY),
        (cfg.OUTPUT_DIRECTORY, cfg.VOLUME_OUTPUT_DIRECTORY),
    ]:
        if not os.path.isdir(local_dir):
            continue
        for name in os.listdir(local_dir):
            local_path = os.path.join(local_dir, name)
            if not os.path.isfile(local_path):
                continue
            with open(local_path, 'rb') as f:
                vol_write_bytes(client, f'{vol_dir}/{name}', f.read())


def _run_single_date(date_str: str, client) -> int:
    """Run the full pipeline for one date. Returns 0 on success, non-zero on error."""
    import importlib
    import tempfile
    import shutil

    # Create staging dir first — config.py reads OPTI_STAGING_ROOT at module
    # level to set INPUT/OUTPUT/INTERMEDIATE_DIRECTORY, so it must be in env
    # before config is reloaded.
    staging_root = tempfile.mkdtemp(prefix='opti_tails_job_')
    os.environ['MTL_DATE'] = date_str
    os.environ['OPTI_STAGING_ROOT'] = staging_root
    print(f'[Job] Staging root: {staging_root}')

    # Reload config (picks up new MTL_DATE + OPTI_STAGING_ROOT) then reload
    # run_pipeline so every stage module re-executes its module-level
    # `from config import ...` against the freshly-reloaded config.
    import sys
    import config as _cfg_mod
    import run_pipeline as _rp_mod
    importlib.reload(_cfg_mod)
    importlib.reload(_rp_mod)
    import config
    from run_pipeline import pipeline, save_config
    print(f'[Job] DATE_PREFIX = {config.DATE_PREFIX}')

    if config.SOLVER_BACKEND == "gurobi":
        print("[Job] SOLVER_BACKEND=gurobi — loading Gurobi secrets...")
        _load_gurobi_secrets(client)

    exit_code = 0
    try:
        print('[Job] Fetching daily input CSV from SQL Warehouse...')
        try:
            _fetch_sql_input(client, config)
        except Exception as e:
            print(f'[Job] ERROR fetching SQL input: {e}', file=sys.stderr)
            traceback.print_exc()
            return 1

        print('[Job] Staging Volume inputs to /tmp...')
        missing = _stage_inputs(client, config)
        if missing:
            print(f'[Job] ERROR: missing required inputs: {missing}', file=sys.stderr)
            return 1

        print('[Job] Staging ML models from Volume to /tmp...')
        try:
            _stage_models(client, config)
        except Exception as e:
            print(f'[Job] ERROR staging models: {e}', file=sys.stderr)
            traceback.print_exc()
            return 1

        save_config()

        for step_name, module in pipeline:
            print(f'\n[Job] Running: {step_name}...')
            try:
                module.main()
                print(f'[Job] {step_name} completed')
            except Exception as e:
                print(f'[Job] ERROR in {step_name}: {e}', file=sys.stderr)
                traceback.print_exc()
                print('[Job] Uploading partial outputs before exit...')
                _stage_outputs(client, config)
                return 2

        print('\n[Job] Uploading outputs to Volume...')
        _stage_outputs(client, config)
        print(f'[Job] {date_str} done.')
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        # Clear OPTI_STAGING_ROOT so the next date gets a fresh temp dir
        os.environ.pop('OPTI_STAGING_ROOT', None)

    return exit_code


def main() -> int:
    dates = _get_run_dates()

    client = _get_ws_client()

    for date_str in dates:
        print(f'\n{"=" * 60}')
        print(f'[Job] MTL_DATE = {date_str}')
        print(f'{"=" * 60}')
        rc = _run_single_date(date_str, client)
        if rc != 0:
            print(f'[Job] FAILED for {date_str} (exit code {rc})', file=sys.stderr)
            return rc

    return 0


if __name__ == '__main__':
    rc = main()
    if rc != 0:
        raise RuntimeError(f'Pipeline failed with exit code {rc}')
