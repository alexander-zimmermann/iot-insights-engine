"""Settings for the iot-insights-engine jobs (anomaly detection +
forecast pulls). MCP_-prefixed env vars kept as the de-facto homelab
convention — same SealedSecrets and Kyverno-clone topology the MCP
server uses already inject these into the namespace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MCP_", env_file=".env", extra="ignore")

    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    db_host: str
    db_port: int = 5432
    db_name: str
    # Settings() validates db_username/password as required at construct
    # time even though most jobs only ever open the *write* connection —
    # this keeps the SealedSecret topology identical between the MCP
    # server and the jobs image.
    db_username: str = ""
    db_password: str = Field(default="", repr=False)
    db_username_file: str | None = None
    db_password_file: str | None = None

    db_write_username: str = ""
    db_write_password: str = Field(default="", repr=False)
    db_write_username_file: str | None = None
    db_write_password_file: str | None = None

    # NATS — jobs publish anomaly events on `anomaly.<uc>.<severity>`.
    nats_servers: str | None = None
    nats_user: str | None = None
    nats_password: str = Field(default="", repr=False)
    nats_password_file: str | None = None
    nats_creds_file: str | None = None
    nats_nkey_seed_file: str | None = None

    # S3 (rustfs) — jobs persist trained models as joblib pickles.
    s3_endpoint: str | None = None
    s3_access_key: str = ""
    s3_secret_key: str = Field(default="", repr=False)
    s3_access_key_file: str | None = None
    s3_secret_key_file: str | None = None
    s3_bucket: str = "iot-mcp-bridge-models"
    s3_region: str = "us-east-1"

    # SMTP — weekly-report job sends Markdown via the cluster-internal
    # relay (no auth, BGP-advertised LoadBalancer).
    smtp_host: str = "smtprelay.smtprelay.svc.cluster.local"
    smtp_port: int = 25
    smtp_from: str = "admin@zimmermann.sh"
    smtp_to: str = "admin@zimmermann.sh"

    # Forecast.Solar — PV-production forecast HTTPS API. Personal-Plus
    # tier supports up to 2 planes in a single request, so the
    # homelab's east+west roof fits one hourly call. Planes are
    # JSON-encoded so adding a 3rd plane is a config change, not a
    # code change.
    forecast_solar_api_key: str = Field(default="", repr=False)
    forecast_solar_api_key_file: str | None = None
    forecast_solar_lat: float | None = None
    forecast_solar_lon: float | None = None
    forecast_solar_planes: str = "[]"
    forecast_solar_base_url: str = "https://api.forecast.solar"
    # forecast.solar returns naive local timestamps in the account's
    # configured timezone — must match the account setting so the job
    # can convert to UTC before writing to `mcp_forecasts.forecast_for`
    # (TIMESTAMPTZ).
    forecast_solar_timezone: str = "Europe/Berlin"

    @model_validator(mode="after")
    def _resolve_db_secret_files(self) -> Settings:
        if self.db_username_file:
            self.db_username = Path(self.db_username_file).read_text(encoding="utf-8").strip()
        if self.db_password_file:
            self.db_password = Path(self.db_password_file).read_text(encoding="utf-8").strip()
        if not self.db_username:
            raise ValueError("MCP_DB_USERNAME or MCP_DB_USERNAME_FILE is required")
        if not self.db_password:
            raise ValueError("MCP_DB_PASSWORD or MCP_DB_PASSWORD_FILE is required")
        return self

    @model_validator(mode="after")
    def _resolve_optional_secret_files(self) -> Settings:
        if self.db_write_username_file:
            self.db_write_username = (
                Path(self.db_write_username_file).read_text(encoding="utf-8").strip()
            )
        if self.db_write_password_file:
            self.db_write_password = (
                Path(self.db_write_password_file).read_text(encoding="utf-8").strip()
            )
        if self.nats_password_file:
            self.nats_password = Path(self.nats_password_file).read_text(encoding="utf-8").strip()
        if self.s3_access_key_file:
            self.s3_access_key = Path(self.s3_access_key_file).read_text(encoding="utf-8").strip()
        if self.s3_secret_key_file:
            self.s3_secret_key = Path(self.s3_secret_key_file).read_text(encoding="utf-8").strip()
        if self.forecast_solar_api_key_file:
            self.forecast_solar_api_key = (
                Path(self.forecast_solar_api_key_file).read_text(encoding="utf-8").strip()
            )
        return self

    @property
    def db_dsn(self) -> str:
        user = quote(self.db_username, safe="")
        password = quote(self.db_password, safe="")
        return f"postgresql://{user}:{password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def db_write_dsn(self) -> str:
        if not self.db_write_username or not self.db_write_password:
            raise ValueError(
                "MCP_DB_WRITE_USERNAME / MCP_DB_WRITE_PASSWORD "
                "(or *_FILE variants) are required"
            )
        user = quote(self.db_write_username, safe="")
        password = quote(self.db_write_password, safe="")
        return f"postgresql://{user}:{password}@{self.db_host}:{self.db_port}/{self.db_name}"


def load_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
