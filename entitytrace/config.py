from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

OPENAI_DEFAULT_TIMEOUT_SECONDS = 420.0
OPENAI_DEFAULT_CONCURRENCY = 8
GENERATOR_PROVIDERS = ("deepseek", "openai")


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    return values


@dataclass(frozen=True)
class Service:
    name: str
    url: str
    model: str
    key: str = field(repr=False)
    provider: str = "deepseek"
    extra_body: dict[str, Any] = field(default_factory=dict, repr=False)
    timeout_seconds: float | None = field(default=None, repr=False)

    def public(self) -> dict:
        return {"service": self.name, "url": self.url, "model": self.model}


def _config_error(message: str) -> ValueError:
    exc = ValueError(message)
    exc.et_category = "config"  # type: ignore[attr-defined]
    return exc


def _check_openai_base_url(url: str) -> str:
    if not url:
        raise _config_error("未配置 ENTITYTRACE_OPENAI_BASE_URL，请检查本机配置或环境变量")
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise _config_error("ENTITYTRACE_OPENAI_BASE_URL 必须是包含主机名的 http 或 https 地址")
    return url


def _parse_extra_body(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _config_error("ENTITYTRACE_OPENAI_EXTRA_BODY 不是合法 JSON") from exc
    if not isinstance(parsed, dict):
        raise _config_error("ENTITYTRACE_OPENAI_EXTRA_BODY 必须是 JSON 对象")
    return parsed


def _parse_timeout_seconds(raw: str) -> float:
    if not raw:
        return OPENAI_DEFAULT_TIMEOUT_SECONDS
    try:
        val = float(raw)
    except ValueError as exc:
        raise _config_error("ENTITYTRACE_OPENAI_TIMEOUT_SECONDS 必须是大于 0 的数值") from exc
    if val <= 0:
        raise _config_error("ENTITYTRACE_OPENAI_TIMEOUT_SECONDS 必须大于 0")
    return val


DECISION_MODES = ("auto", "on", "off")


def resolve_decision_mode(
    explicit: str | None = None,
    config_path: Path | dict[str, str] | None = None,
) -> str:
    if isinstance(config_path, Path):
        local = read_env(config_path)
    elif isinstance(config_path, dict):
        local = config_path
    else:
        local = {}
    if explicit is not None:
        mode = explicit.strip().lower()
    else:
        env_val = os.environ.get("ENTITYTRACE_DECISION_MODE", "").strip().lower()
        local_val = local.get("ENTITYTRACE_DECISION_MODE", "").strip().lower()
        mode = env_val or local_val or "auto"
    if mode not in DECISION_MODES:
        raise _config_error(f"不支持的 Decision 模式：{mode or '(空)'}（可选：auto、on、off）")
    return mode


def is_typesafe_decision_service(service: Service | None) -> bool:
    if service is None or not getattr(service, "key", ""):
        return False
    if getattr(service, "name", None) != "decision":
        return False
    model = getattr(service, "model", None)
    if not (isinstance(model, str) and model.startswith("jev-") and len(model) > 4):
        return False
    url = str(getattr(service, "url", "")).rstrip("/")
    return url.startswith("https://api.typesafe.ai") or getattr(service, "provider", None) == "typesafe"


def load_services(
    path: Path = Path(".env.models.local"),
    require_decision: bool = False,
    generator: str | None = None,
    *,
    decision_mode: str | None = None,
) -> dict[str, Service]:
    local = read_env(path)
    mode = resolve_decision_mode(decision_mode, local)
    def setting(name: str, default: str = "") -> str:
        return os.environ.get(name, local.get(name, default)).strip()
    provider = (generator or setting("GENERATOR_PROVIDER", "deepseek")).strip()
    if provider not in GENERATOR_PROVIDERS:
        raise _config_error(f"不支持的生成服务商：{provider or '(空)'}（可选：deepseek、openai）")
    services: dict[str, Service] = {}
    jev_model = ""
    if require_decision:
        jev_model = setting("TYPESAFE_MODEL", "jev-1.13.0")
        if jev_model == "jev-latest":
            jev_model = "jev-1.13.0"
        services["decision"] = Service("decision", setting("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1").rstrip("/"),
                                       jev_model, setting("TYPESAFE_API_KEY"))
    elif mode != "off":
        ts_key = setting("TYPESAFE_API_KEY")
        if mode == "on" or ts_key:
            jev_model = setting("TYPESAFE_MODEL", "jev-latest") or "jev-latest"
            services["decision"] = Service("decision", setting("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1").rstrip("/"),
                                           jev_model, ts_key)
    if provider == "openai":
        base = _check_openai_base_url(setting("ENTITYTRACE_OPENAI_BASE_URL").rstrip("/"))
        model = setting("ENTITYTRACE_OPENAI_MODEL")
        if not model:
            raise _config_error("未配置 ENTITYTRACE_OPENAI_MODEL，请检查本机配置或环境变量")
        api_key = setting("ENTITYTRACE_OPENAI_API_KEY")
        if not api_key:
            raise _config_error("未配置 ENTITYTRACE_OPENAI_API_KEY，请检查本机配置或环境变量")
        extra_body = _parse_extra_body(setting("ENTITYTRACE_OPENAI_EXTRA_BODY"))
        timeout_seconds = _parse_timeout_seconds(setting("ENTITYTRACE_OPENAI_TIMEOUT_SECONDS"))
        services["discovery"] = Service(
            "discovery",
            base,
            model,
            api_key,
            provider="openai",
            extra_body=extra_body,
            timeout_seconds=timeout_seconds,
        )
    else:
        services["discovery"] = Service("discovery", setting("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
                                        setting("DEEPSEEK_MODEL", "deepseek-flash"), setting("DEEPSEEK_API_KEY"))
    allowed = {"decision": {"https://api.typesafe.ai/v1"},
               "discovery": {"https://api.deepseek.com", "https://api.deepseek.com/v1"}}
    for name, service in services.items():
        if not service.key:
            if name == "decision":
                if require_decision:
                    raise ValueError("groups-v2 需要 TypeSafe/Jev 凭证，请配置 TYPESAFE_API_KEY")
                raise _config_error("已启用 --decision on，但未配置 TYPESAFE_API_KEY")
            raise ValueError(f"未配置 {name} 模型密钥，请检查本机配置文件")
        if service.provider == "deepseek" and service.url not in allowed[name]:
            if name == "decision" and not require_decision:
                raise _config_error(f"{name} 的服务地址不是已核对的官方地址")
            raise ValueError(f"{name} 的服务地址不是已核对的官方地址")
    if require_decision and (not jev_model.startswith("jev-") or jev_model == "jev-latest"):
        raise ValueError("Decision 模型需要明确版本")
    if "decision" in services and not (jev_model.startswith("jev-") and len(jev_model) > 4):
        raise _config_error("Decision 模型版本非法（须以 jev- 开头）")
    return services
