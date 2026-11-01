from __future__ import annotations

import collections
import io
import json
import math
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from typing import Callable

from .config import Service
from .storage import Store
from .util import digest, dumps, utc_now


class ModelError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status, self.code = status, code


class BudgetError(RuntimeError):
    """在 HTTP 客户端之外传递预算或连续断网停止，不能变成格式错误或模型答案。"""

    def __init__(self, message: str, category: str = "budget", stop_reason: str | None = None):
        super().__init__(message)
        self.category = category
        self.stop_reason = stop_reason


ExperimentBudget = BudgetError


class RequestBudget:
    def __init__(self, max_requests: int | None = None, max_seconds: float | None = None,
                 opener=None, max_network_failures: int = 9):
        if max_requests is not None and max_requests < 0:
            raise ValueError("请求预算不能为负数")
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError("时间预算必须大于 0")
        if max_network_failures < 1:
            raise ValueError("连续网络失败上限必须至少为 1")
        self.max_requests, self.max_seconds = max_requests, max_seconds
        self.started = time.perf_counter()
        self.count = 0
        self.stop_reason: str | None = None
        self.max_network_failures = max_network_failures
        self.consecutive_network_failures = 0
        self.network_errors: collections.Counter[str] = collections.Counter()
        self.lock = threading.Lock()
        self.opener = opener or urllib.request.urlopen

    def consume(self) -> float | None:
        with self.lock:
            remaining = None if self.max_seconds is None else self.max_seconds - (time.perf_counter() - self.started)
            if self.stop_reason == "network_outage":
                raise BudgetError("连续网络失败，停止新请求并保留断点", category="network", stop_reason="network_outage")
            if (self.max_requests is not None and self.count >= self.max_requests) or (remaining is not None and remaining <= 0):
                self.stop_reason = (
                    "request_budget"
                    if (self.max_requests is not None and self.count >= self.max_requests)
                    else "time_budget"
                )
                raise BudgetError("全文请求数或时间预算耗尽；保留未完成任务以便恢复", category="budget", stop_reason=self.stop_reason)
            self.count += 1
            return remaining

    def __call__(self, request, timeout: float | None = None):
        remaining = self.consume()
        effective_timeout = timeout if remaining is None else (remaining if timeout is None else min(timeout, remaining))
        try:
            # 在预算包装层读完响应，连接阶段和读取阶段失败都能计入停止条件。
            with self.opener(request, timeout=effective_timeout) as response:
                body = response.read()
        except urllib.error.HTTPError:
            # HTTP 错误由生产客户端解析、重试或触发硬额度保护。
            with self.lock:
                self.consecutive_network_failures = 0
            raise
        except OSError as exc:
            with self.lock:
                self.consecutive_network_failures += 1
                self.network_errors[type(exc).__name__] += 1
                if self.consecutive_network_failures >= self.max_network_failures:
                    self.stop_reason = "network_outage"
            raise
        with self.lock:
            self.consecutive_network_failures = 0
        return io.BytesIO(body)

    def summary(self) -> dict:
        with self.lock:
            return {
                "max_requests": self.max_requests,
                "max_seconds": self.max_seconds,
                "max_network_failures": self.max_network_failures,
                "network_attempts": self.count,
                "stop_reason": self.stop_reason,
                "network_errors": dict(self.network_errors),
            }


def choice_question(instructions: str, criteria: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def check_choices(raw: dict, questions: dict, model: str) -> dict:
    if not isinstance(raw, dict) or raw.get("model") != model or not isinstance(raw.get("answers"), dict):
        raise ModelError("Decision 返回结构或模型版本不一致")
    answers = raw["answers"]
    if set(answers) != set(questions):
        raise ModelError("Decision 问题与回答数量不一致")
    for key, question in questions.items():
        answer = answers[key]
        if not isinstance(answer, dict):
            raise ModelError("Decision 回答格式错误")
        probs, confidence = answer.get("probabilities"), answer.get("confidence")
        if not isinstance(probs, dict) or set(probs) != set(question["criteria"]):
            raise ModelError("Decision 概率选项不完整")
        if answer.get("type") != "choice" or answer.get("choice") not in probs:
            raise ModelError("Decision 选择不合法")
        if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in [*probs.values(), confidence]):
            raise ModelError("Decision 分数范围不合法")
        if abs(sum(probs.values()) - 1) > 0.025:
            raise ModelError("Decision 概率总和异常")
        if probs[answer["choice"]] + 1e-7 < max(probs.values()):
            raise ModelError("Decision 选择与最高概率不一致")
    return answers


class Models:
    def __init__(self, store: Store, run_id: str, invocation: str, services: dict[str, Service],
                 discovery_concurrency: int = 6, decision_concurrency: int = 3,
                 timeout: float = 120, attempts: int = 3, opener=None,
                 budget: RequestBudget | None = None):
        self.store, self.run_id, self.invocation, self.services = store, run_id, invocation, services
        self.timeout, self.attempts = timeout, attempts
        self.budget = budget
        if budget is not None:
            if opener is not None and budget.opener is urllib.request.urlopen:
                budget.opener = opener
            self.opener = budget
        else:
            self.opener = opener or urllib.request.urlopen
        self.gates = {"discovery": threading.BoundedSemaphore(discovery_concurrency),
                      "decision": threading.BoundedSemaphore(decision_concurrency)}
        self.lock = threading.Lock()
        self.cooldown = {name: 0.0 for name in services}
        self.active = {name: 0 for name in services}
        self.peaks = {name: 0 for name in services}
        self.halted = {}

    def _request(self, service_name: str, path: str, payload: dict, stage: str, validate: Callable) -> tuple[dict, str]:
        service = self.services[service_name]
        request_id = digest(["http-v1", service.public(), path, payload])
        cached = self.store.cache(request_id)
        if cached is not None:
            validate(cached)
            self.store.call(self.run_id, self.invocation, stage, request_id, service_name, 0, "cached", 0, {})
            return cached, request_id
        error = "模型请求失败"
        for attempt in range(1, self.attempts + 1):
            if self.budget is not None and self.budget.stop_reason == "network_outage":
                raise BudgetError("连续网络失败，停止新请求并保留断点", category="network", stop_reason="network_outage")
            error_status, error_code = None, None
            with self.lock:
                if service_name in self.halted:
                    raise ModelError(*self.halted[service_name])
                delay = self.cooldown[service_name] - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            retryable, retry_after = True, 0.0
            begin = time.monotonic()
            detail: dict = {}
            raw = None
            with self.gates[service_name]:
                with self.lock:
                    if service_name in self.halted:
                        raise ModelError(*self.halted[service_name])
                    self.active[service_name] += 1
                    self.peaks[service_name] = max(self.peaks[service_name], self.active[service_name])
                    active = self.active[service_name]
                try:
                    request = urllib.request.Request(service.url + path, data=dumps(payload).encode("utf-8"),
                        headers={"Authorization": "Bearer " + service.key, "Content-Type": "application/json"}, method="POST")
                    with self.opener(request, timeout=self.timeout) as response:
                        raw = json.loads(response.read().decode("utf-8"))
                    validate(raw)
                    # Authorization 头从不写入日志；不保存模型内部推理字段。
                    for choice in raw.get("choices", []):
                        choice.get("message", {}).pop("reasoning_content", None)
                    detail = {"usage": raw.get("usage", {}), "active_requests": active}
                except urllib.error.HTTPError as exc:
                    error_status = exc.code
                    error = f"{service_name} 请求失败（HTTP {exc.code}）"
                    body = exc.read().decode("utf-8", errors="replace").lower()
                    try:
                        problem = json.loads(body)
                        code = problem.get("detail", {}).get("error_type") if isinstance(problem.get("detail"), dict) else None
                        if isinstance(code, str) and len(code) <= 80 and code.replace("_", "").isalnum():
                            error_code = code
                    except (ValueError, TypeError, AttributeError):
                        pass
                    hard_quota = any(word in body for word in ["insufficient", "balance", "billing", "余额", "资源包", "usage limit", "quota_exceeded"])
                    retryable = exc.code in {429, 500, 502, 503, 504, 529} and not hard_quota
                    header = exc.headers.get("Retry-After", "")
                    try:
                        retry_after = float(header)
                    except ValueError:
                        try:
                            retry_after = parsedate_to_datetime(header).timestamp() - time.time()
                        except (TypeError, ValueError, OverflowError):
                            retry_after = 0
                    detail = {"http_status": exc.code, "hard_quota": hard_quota, "error_type": error_code}
                    if hard_quota:
                        error += "，余额或硬额度不足，本次运行停止该服务的新请求；成功缓存仍可复用"
                        with self.lock:
                            self.halted[service_name] = (error, error_status, error_code)
                    raw = None
                except (urllib.error.URLError, TimeoutError, socket.timeout, OSError):
                    error = f"{service_name} 网络失败或超时"
                    error_code = "network_error"
                    raw = None
                except (ValueError, KeyError, TypeError, ModelError) as exc:
                    error = f"{service_name} 返回格式不符合约定"
                    detail = {"validation_error": str(exc)[:160] if isinstance(exc, ModelError) else type(exc).__name__}
                    raw = None
                finally:
                    with self.lock:
                        self.active[service_name] -= 1
            seconds = time.monotonic() - begin
            self.store.call(self.run_id, self.invocation, stage, request_id, service_name, attempt,
                            "success" if raw is not None else "failed", seconds, detail)
            if raw is not None:
                self.store.execute("INSERT OR REPLACE INTO request_cache VALUES(?,?,?,?,?,?)",
                    (request_id, service_name, service.model, dumps(payload), dumps(raw), utc_now()))
                return raw, request_id
            if self.budget is not None and self.budget.stop_reason == "network_outage":
                raise BudgetError("连续网络失败，停止新请求并保留断点", category="network", stop_reason="network_outage")
            if not retryable or attempt == self.attempts:
                raise ModelError(error, error_status, error_code)
            wait = min(60.0, max(2 ** (attempt - 1), retry_after))
            with self.lock:
                self.cooldown[service_name] = max(self.cooldown[service_name], time.monotonic() + wait)
        raise ModelError(error)

    def choices(self, state: dict, questions: dict, stage: str) -> tuple[dict, str]:
        model = self.services["decision"].model
        payload = {"model": model, "state": state, "questions": questions}
        raw, request_id = self._request("decision", "/systemone", payload, stage,
                                      lambda value: check_choices(value, questions, model))
        return raw["answers"], request_id

    def generate(self, system: str, state: dict, stage: str, validate: Callable,
                 max_tokens: int = 8000, temperature: float | None = None) -> tuple[dict, str]:
        payload = {"model": self.services["discovery"].model,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": dumps(state)}],
                   "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"}, "max_tokens": max_tokens}
        if temperature is not None:
            payload["temperature"] = temperature
        def check(raw):
            try:
                choice = raw["choices"][0]
                if choice["finish_reason"] != "stop":
                    raise ModelError("生成输出被截断，未记为完成")
                validate(json.loads(choice["message"]["content"]))
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise ModelError("生成返回不是有效 JSON") from exc
        raw, request_id = self._request("discovery", "/chat/completions", payload, stage, check)
        return json.loads(raw["choices"][0]["message"]["content"]), request_id


def bounded_map(items: list, worker: Callable, workers: int):
    """始终至多 workers 个任务在途，某任务完成后立即补位。"""
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for _ in range(workers):
            item = next(iterator, None)
            if item is not None:
                pending[pool.submit(worker, item)] = item
        while pending:
            future = next(as_completed(pending))
            item = pending.pop(future)
            try:
                yield item, future.result(), None
            except Exception as exc:
                yield item, None, exc
            following = next(iterator, None)
            if following is not None:
                pending[pool.submit(worker, following)] = following
