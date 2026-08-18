"""Databricks Volume I/O primitives shared between app.py and run_pipeline_job.py."""

import io
import os

from databricks.sdk import WorkspaceClient


def make_ws_client() -> WorkspaceClient:
    """Create a WorkspaceClient from available env credentials: OAuth M2M → PAT → SDK default."""
    host = os.environ.get("DATABRICKS_HOST", "").strip()
    if not host:
        return WorkspaceClient()
    if not host.startswith("http"):
        host = f"https://{host}"
    client_id = os.environ.get("DATABRICKS_CLIENT_ID")
    client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")
    token = os.environ.get("DATABRICKS_TOKEN", "").strip()

    if client_id and client_secret:
        return WorkspaceClient(host=host, client_id=client_id, client_secret=client_secret, auth_type="oauth-m2m")
    if token:
        return WorkspaceClient(host=host, token=token, auth_type="pat")
    return WorkspaceClient()


def vol_list(client: WorkspaceClient, dir_path: str) -> list:
    try:
        return list(client.files.list_directory_contents(dir_path))
    except Exception:
        return []


def vol_read_bytes(client: WorkspaceClient, file_path: str) -> bytes:
    return client.files.download(file_path).contents.read()


def vol_exists(client: WorkspaceClient, file_path: str) -> bool:
    try:
        client.files.get_metadata(file_path)
        return True
    except Exception:
        return False


def vol_write_bytes(client: WorkspaceClient, file_path: str, data: bytes) -> None:
    client.files.upload(file_path, io.BytesIO(data), overwrite=True)


def vol_rmdir(client: WorkspaceClient, dir_path: str) -> None:
    for item in vol_list(client, dir_path):
        if item.is_directory:
            vol_rmdir(client, item.path)
        else:
            client.files.delete(item.path)
    try:
        client.files.delete_directory(dir_path)
    except Exception:
        pass
