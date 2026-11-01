from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


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

    def public(self) -> dict:
        return {"service": self.name, "url": self.url, "model": self.model}


def load_services(path: Path, require_decision: bool = False) -> dict[str, Service]:
    local = read_env(path)
    def setting(name: str, default: str = "") -> str:
        return os.environ.get(name, local.get(name, default)).strip()
    services: dict[str, Service] = {}
    jev_model = ""
    if require_decision:
        jev_model = setting("TYPESAFE_MODEL", "jev-1.13.0")
        if jev_model == "jev-latest":
            jev_model = "jev-1.13.0"
        services["decision"] = Service("decision", setting("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1").rstrip("/"),
                                       jev_model, setting("TYPESAFE_API_KEY"))
    services["discovery"] = Service("discovery", setting("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
                                    setting("DEEPSEEK_MODEL", "deepseek-flash"), setting("DEEPSEEK_API_KEY"))
    allowed = {"decision": {"https://api.typesafe.ai/v1"},
               "discovery": {"https://api.deepseek.com", "https://api.deepseek.com/v1"}}
    for name, service in services.items():
        if not service.key:
            if name == "decision":
                raise ValueError("groups-v2 需要 TypeSafe/Jev 凭证，请配置 TYPESAFE_API_KEY")
            raise ValueError(f"未配置 {name} 模型密钥，请检查本机配置文件")
        if service.url not in allowed[name]:
            raise ValueError(f"{name} 的服务地址不是已核对的官方地址")
    if require_decision and (not jev_model.startswith("jev-") or jev_model == "jev-latest"):
        raise ValueError("Decision 模型需要明确版本")
    return services
