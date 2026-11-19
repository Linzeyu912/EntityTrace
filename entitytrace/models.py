from __future__ import annotations

import collections
import io
import json
import math
import queue
import socket
import statistics
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
        except urllib.error.HTTPError as exc:
            # HTTP 502/503/504/529 计入连续网络失败；其余 HTTP 错误由生产客户端解析、重试或触发硬额度保护。
            with self.lock:
                if exc.code in {502, 503, 504, 529}:
                    self.consecutive_network_failures += 1
                    self.network_errors[f"HTTP {exc.code}"] += 1
                    if self.consecutive_network_failures >= self.max_network_failures:
                        self.stop_reason = "network_outage"
                else:
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
    if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
        raise ModelError("Decision 返回结构或模型版本不一致")
    served = raw.get("model")
    if model == "jev-latest":
        if not (isinstance(served, str) and served.startswith("jev-") and len(served) > 4):
            raise ModelError("Decision 返回结构或模型版本不一致")
    elif served != model:
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


def _strip_code_fence(content: str) -> str:
    """去掉首尾 Markdown 代码围栏；仅通用兼容路径启用。"""
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_tolerant_json(content: str, *, strip_fence: bool = False) -> object:
    """解析模型输出中首个 JSON 值；dict 后跟零散 `,\"key\":...}` 片段时合并补齐。

    合并规则：新键直接补入；已有键两侧均为 list 时拼接；其余已有键不覆盖。
    非 dict 的首值要求其后为空，否则抛出 `json.JSONDecodeError`。
    """
    s = _strip_code_fence(content) if strip_fence else content.strip()
    obj, pos = json.JSONDecoder().raw_decode(s)
    if not isinstance(obj, dict):
        if s[pos:].strip():
            raise json.JSONDecodeError("Extra data", s, pos)
        return obj
    rest = s[pos:].strip()
    while rest:
        if rest.startswith(","):
            rest = "{" + rest[1:]
        extra, p = json.JSONDecoder().raw_decode(rest)
        if not isinstance(extra, dict):
            raise json.JSONDecodeError("Extra non-object data", rest, 0)
        for k, v in extra.items():
            if k not in obj:
                obj[k] = v
            elif isinstance(obj[k], list) and isinstance(v, list):
                obj[k] = obj[k] + v
        rest = rest[p:].strip()
    return obj


class Models:
    def __init__(self, store: Store, run_id: str, invocation: str, services: dict[str, Service],
                 discovery_concurrency: int = 6, decision_concurrency: int = 3,
                 timeout: float = 120, attempts: int = 3, opener=None,
                 budget: RequestBudget | None = None,
                 hedge_enabled: bool = True, hedge_min_seconds: float = 20.0,
                 hedge_min_samples: int = 8, hedge_max_ratio: float = 0.05):
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
        self.hedge_enabled = hedge_enabled
        self.hedge_min_seconds = hedge_min_seconds
        self.hedge_min_samples = hedge_min_samples
        self.hedge_max_ratio = hedge_max_ratio
        self._stage_latencies: dict[str, list[float]] = collections.defaultdict(list)
        self._sent_count = 0
        self._hedged_count = 0

    def _hedge_delay(self, stage: str) -> float:
        """生成请求慢请求对冲阈值：样本不足时用固定下限，否则取 3 倍成功耗时中位数。"""
        with self.lock:
            samples = list(self._stage_latencies.get(stage, []))
        if len(samples) < self.hedge_min_samples:
            return self.hedge_min_seconds
        return max(self.hedge_min_seconds, 3.0 * statistics.median(samples))

    def _try_reserve_hedge(self, service_name: str) -> bool:
        """对冲准入判断：仅 discovery 生成请求、未停机、不在冷却、占比未超限且预算仍有剩余。"""
        if not (self.hedge_enabled and service_name == "discovery"):
            return False
        with self.lock:
            if service_name in self.halted:
                return False
            if self.cooldown[service_name] > time.monotonic():
                return False
            if not (self._sent_count > 0
                    and (self._hedged_count + 1) <= self._sent_count * self.hedge_max_ratio):
                return False
            if self.budget is not None:
                with self.budget.lock:
                    if self.budget.stop_reason == "network_outage":
                        return False
                    if (self.budget.max_requests is not None
                            and self.budget.count >= self.budget.max_requests):
                        return False
                    if (self.budget.max_seconds is not None
                            and (self.budget.max_seconds - (time.perf_counter() - self.budget.started)) <= 0):
                        return False
            self._hedged_count += 1
            return True

    def _request(self, service_name: str, path: str, payload: dict, stage: str, validate: Callable) -> tuple[dict, str]:
        service = self.services[service_name]
        request_id = digest(["http-v1", service.public(), path, payload])
        cached = self.store.cache(request_id)
        if cached is not None:
            validate(cached)
            self.store.call(self.run_id, self.invocation, stage, request_id, service_name, 0, "cached", 0, {})
            return cached, request_id
        error = "模型请求失败"
        drop_temperature = False
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
            send_payload = (
                {k: v for k, v in payload.items() if k != "temperature"}
                if (drop_temperature and "temperature" in payload)
                else payload
            )
            temperature_sent = send_payload.get("temperature")
            detail: dict = {"temperature_sent": temperature_sent, "served_model": None}
            raw = None
            hedging = service_name == "discovery" and self.hedge_enabled
            # _execute_http 在子线程内严禁写 store；结果统一经 out_q 由当前尝试线程串行落库。
            # 返回元组：(is_hedge, success, raw, error_kind, detail, info, req_seconds)
            #   error_kind ∈ None/"budget"/"http"/"network"/"format"；info 仅 http 附带。
            def _execute_http(is_hedge: bool, out_q: queue.Queue | None):
                req_begin = time.monotonic()
                r = None
                try:
                    request = urllib.request.Request(service.url + path, data=dumps(send_payload).encode("utf-8"),
                        headers={"Authorization": "Bearer " + service.key, "Content-Type": "application/json"}, method="POST")
                    with self.opener(request, timeout=self.timeout) as response:
                        r = json.loads(response.read().decode("utf-8"))
                    served_model = r.get("model") if isinstance(r, dict) and isinstance(r.get("model"), str) else None
                    validate(r)
                    # Authorization 头从不写入日志；不保存模型内部推理字段。
                    for choice in r.get("choices", []):
                        choice.get("message", {}).pop("reasoning_content", None)
                    det = {
                        "usage": r.get("usage", {}),
                        "active_requests": active,
                        "temperature_sent": temperature_sent,
                        "served_model": served_model,
                        "hedge": is_hedge,
                    }
                    result = (is_hedge, True, r, None, det, {}, time.monotonic() - req_begin)
                except BudgetError as exc:
                    result = (is_hedge, False, None, "budget",
                              {"temperature_sent": temperature_sent, "served_model": None, "hedge": is_hedge},
                              {"exc": exc}, time.monotonic() - req_begin)
                except urllib.error.HTTPError as exc:
                    err = f"{service_name} 请求失败（HTTP {exc.code}）"
                    body = exc.read().decode("utf-8", errors="replace").lower()
                    code = None
                    try:
                        problem = json.loads(body)
                        cand = problem.get("detail", {}).get("error_type") if isinstance(problem.get("detail"), dict) else None
                        if isinstance(cand, str) and len(cand) <= 80 and cand.replace("_", "").isalnum():
                            code = cand
                    except (ValueError, TypeError, AttributeError):
                        pass
                    hq = any(word in body for word in ["insufficient", "balance", "billing", "余额", "资源包", "usage limit", "quota_exceeded"])
                    rep = exc.code in {429, 500, 502, 503, 504, 529} and not hq
                    header = exc.headers.get("Retry-After", "")
                    ra = 0.0
                    try:
                        ra = float(header)
                    except ValueError:
                        try:
                            ra = parsedate_to_datetime(header).timestamp() - time.time()
                        except (TypeError, ValueError, OverflowError):
                            ra = 0
                    if hq:
                        err += "，余额或硬额度不足，本次运行停止该服务的新请求；成功缓存仍可复用"
                    det = {
                        "http_status": exc.code,
                        "hard_quota": hq,
                        "error_type": code,
                        "temperature_sent": temperature_sent,
                        "served_model": None,
                        "hedge": is_hedge,
                    }
                    result = (is_hedge, False, None, "http", det,
                              {"error": err, "status": exc.code, "code": code, "hard_quota": hq,
                               "retryable": rep, "retry_after": ra}, time.monotonic() - req_begin)
                except (urllib.error.URLError, TimeoutError, socket.timeout, OSError):
                    result = (is_hedge, False, None, "network",
                              {"temperature_sent": temperature_sent, "served_model": None, "hedge": is_hedge},
                              {"error": f"{service_name} 网络失败或超时"}, time.monotonic() - req_begin)
                except (ValueError, KeyError, TypeError, ModelError) as exc:
                    served_model = r.get("model") if isinstance(r, dict) and isinstance(r.get("model"), str) else None
                    det = {
                        "validation_error": str(exc)[:160] if isinstance(exc, ModelError) else type(exc).__name__,
                        "temperature_sent": temperature_sent,
                        "served_model": served_model,
                        "hedge": is_hedge,
                    }
                    if isinstance(r, dict):
                        try:
                            content = r["choices"][0]["message"]["content"]
                        except (KeyError, IndexError, TypeError):
                            content = None
                        if isinstance(content, str):
                            det["content_head"] = content[:200]
                            det["content_tail"] = content[-200:]
                    if isinstance(r, dict) and isinstance(r.get("usage"), dict) and r["usage"]:
                        det["usage"] = r["usage"]
                    result = (is_hedge, False, None, "format", det,
                              {"error": f"{service_name} 返回格式不符合约定"}, time.monotonic() - req_begin)
                if out_q is not None:
                    out_q.put(result)
                else:
                    return result

            pending_calls: list[tuple[str, float, dict]] = []
            budget_exc: BudgetError | None = None
            with self.gates[service_name]:
                with self.lock:
                    if service_name in self.halted:
                        raise ModelError(*self.halted[service_name])
                    self.active[service_name] += 1
                    self.peaks[service_name] = max(self.peaks[service_name], self.active[service_name])
                    active = self.active[service_name]
                    if hedging:
                        self._sent_count += 1
                try:
                    if not hedging:
                        result = _execute_http(False, None)
                    else:
                        out_q: queue.Queue = queue.Queue()
                        in_flight = 1
                        hedge_delay = self._hedge_delay(stage)
                        threading.Thread(target=_execute_http, args=(False, out_q), daemon=True).start()
                        try:
                            result = out_q.get(timeout=hedge_delay)
                        except queue.Empty:
                            if self._try_reserve_hedge(service_name):
                                in_flight = 2
                                threading.Thread(target=_execute_http, args=(True, out_q), daemon=True).start()
                            result = out_q.get()
                        while True:
                            is_hedge, success, r, kind, det, info, req_sec = result
                            if success:
                                raw = r
                                detail = det
                                with self.lock:
                                    self._stage_latencies[stage].append(req_sec)
                                pending_calls.append(("success", req_sec, detail))
                                if in_flight == 2:
                                    loser_detail = {"hedge": not is_hedge, "discarded": True,
                                                    "temperature_sent": temperature_sent, "served_model": None}
                                    pending_calls.append(("discarded", 0.0, loser_detail))
                                in_flight = 0
                                break
                            if kind == "budget":
                                if not is_hedge:
                                    budget_exc = info["exc"]
                                    break
                                # 对冲请求竞态耗尽预算：不打断仍在途的主请求。
                                in_flight -= 1
                                if in_flight <= 0:
                                    break
                                result = out_q.get()
                                continue
                            pending_calls.append(("failed", req_sec, det))
                            if kind == "format":
                                drop_temperature = True
                            if kind == "http":
                                error, error_status, error_code = info["error"], info["status"], info["code"]
                                retryable, retry_after = info["retryable"], info["retry_after"]
                                if info.get("hard_quota"):
                                    with self.lock:
                                        self.halted[service_name] = (error, error_status, error_code)
                                    retryable = False
                                    break
                            else:
                                error = info["error"]
                                if kind == "network":
                                    error_code = "network_error"
                            in_flight -= 1
                            if in_flight <= 0:
                                break
                            result = out_q.get()
                finally:
                    with self.lock:
                        self.active[service_name] -= 1
            if hedging:
                for call_status, call_sec, call_det in pending_calls:
                    self.store.call(self.run_id, self.invocation, stage, request_id, service_name,
                                    attempt, call_status, call_sec, call_det)
                if budget_exc is not None:
                    raise budget_exc
            else:
                is_hedge, success, r, kind, det, info, req_sec = result
                if success:
                    raw = r
                    detail = det
                    with self.lock:
                        self._stage_latencies[stage].append(req_sec)
                elif kind == "budget":
                    raise info["exc"]
                else:
                    detail = det
                    if kind == "format":
                        drop_temperature = True
                    if kind == "http":
                        error, error_status, error_code = info["error"], info["status"], info["code"]
                        retryable, retry_after = info["retryable"], info["retry_after"]
                        if info.get("hard_quota"):
                            with self.lock:
                                self.halted[service_name] = (error, error_status, error_code)
                    else:
                        error = info["error"]
                        if kind == "network":
                            error_code = "network_error"
                self.store.call(self.run_id, self.invocation, stage, request_id, service_name, attempt,
                                "success" if raw is not None else "failed", req_sec, detail)
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
        service = self.services["discovery"]
        is_openai = service.provider == "openai"
        payload = {"model": service.model,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": dumps(state)}],
                   "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"}, "max_tokens": max_tokens}
        if is_openai:
            del payload["thinking"]
            for k, v in (service.extra_body or {}).items():
                if v is None:
                    payload.pop(k, None)
                else:
                    payload[k] = v
        if temperature is not None:
            payload["temperature"] = temperature
        validated: dict[int, object] = {}
        def check(raw):
            try:
                choice = raw["choices"][0]
                if choice["finish_reason"] != "stop":
                    raise ModelError("生成输出被截断，未记为完成")
                content = choice["message"]["content"]
                parsed = parse_tolerant_json(content, strip_fence=is_openai)
                validate(parsed)
                validated[id(raw)] = parsed
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise ModelError("生成返回不是有效 JSON") from exc
        raw, request_id = self._request("discovery", "/chat/completions", payload, stage, check)
        return validated[id(raw)], request_id


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
