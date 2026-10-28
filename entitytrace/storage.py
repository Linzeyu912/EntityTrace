from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .util import dumps, utc_now


class Store:
    """锁只覆盖短 SQL 事务；模型请求永远在调用方、事务外执行。"""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS sources(
          id TEXT PRIMARY KEY, metadata TEXT NOT NULL, text TEXT NOT NULL, raw BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS runs(
          id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id),
          config TEXT NOT NULL, state TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tasks(
          run_id TEXT NOT NULL REFERENCES runs(id), id TEXT NOT NULL, stage TEXT NOT NULL,
          start INTEGER, end INTEGER, status TEXT NOT NULL, result TEXT, error TEXT,
          PRIMARY KEY(run_id,id));
        CREATE INDEX IF NOT EXISTS task_stage ON tasks(run_id,stage,status);
        CREATE TABLE IF NOT EXISTS records(
          run_id TEXT NOT NULL REFERENCES runs(id), kind TEXT NOT NULL, id TEXT NOT NULL,
          start INTEGER, data TEXT NOT NULL, PRIMARY KEY(run_id,kind,id));
        CREATE INDEX IF NOT EXISTS record_position ON records(run_id,kind,start);
        CREATE TABLE IF NOT EXISTS request_cache(
          id TEXT PRIMARY KEY, service TEXT NOT NULL, model TEXT NOT NULL,
          request TEXT NOT NULL, response TEXT NOT NULL, created TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS calls(
          id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, invocation TEXT NOT NULL, stage TEXT NOT NULL,
          request_id TEXT NOT NULL, service TEXT NOT NULL, attempt INTEGER NOT NULL,
          status TEXT NOT NULL, seconds REAL NOT NULL, detail TEXT NOT NULL, created TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS calls_run ON calls(run_id,invocation,stage);
        CREATE TABLE IF NOT EXISTS human_answers(
          run_id TEXT NOT NULL, review_id TEXT NOT NULL, data TEXT NOT NULL,
          PRIMARY KEY(run_id,review_id));
        """)
        self.db.commit()

    def close(self) -> None:
        with self.lock:
            self.db.close()

    def execute(self, sql: str, args: tuple = ()) -> None:
        with self.lock, self.db:
            self.db.execute(sql, args)

    def rows(self, sql: str, args: tuple = ()) -> list[dict]:
        with self.lock:
            return [dict(row) for row in self.db.execute(sql, args).fetchall()]

    def source(self, source_id: str) -> dict:
        values = self.rows("SELECT * FROM sources WHERE id=?", (source_id,))
        if not values:
            raise ValueError("原文版本不存在")
        result = values[0]
        result["metadata"] = json.loads(result["metadata"])
        return result

    def run(self, run_id: str) -> dict:
        values = self.rows("SELECT * FROM runs WHERE id=?", (run_id,))
        if not values:
            raise ValueError("运行编号不存在")
        value = values[0]
        value["config"] = json.loads(value["config"])
        value["state"] = json.loads(value["state"])
        return value

    def state(self, run_id: str, **updates: Any) -> None:
        with self.lock, self.db:
            state = self.run(run_id)["state"]
            state.update(updates)
            self.db.execute("UPDATE runs SET state=?,updated=? WHERE id=?", (dumps(state), utc_now(), run_id))

    def put(self, run_id: str, kind: str, record: dict) -> None:
        self.execute("INSERT OR REPLACE INTO records VALUES(?,?,?,?,?)",
                     (run_id, kind, record["id"], record.get("start"), dumps(record)))

    def records(self, run_id: str, kind: str) -> list[dict]:
        return [json.loads(r["data"]) for r in self.rows(
            "SELECT data FROM records WHERE run_id=? AND kind=? ORDER BY start,id", (run_id, kind))]

    def task(self, run_id: str, task_id: str) -> dict | None:
        values = self.rows("SELECT * FROM tasks WHERE run_id=? AND id=?", (run_id, task_id))
        if not values:
            return None
        value = values[0]
        value["result"] = json.loads(value["result"]) if value["result"] else None
        return value

    def finish_task(self, run_id: str, task: dict, result: dict, records: list[tuple[str, dict]] = ()) -> None:
        # 结果、提及和完成状态在同一事务中落地，失败会整体回滚。
        with self.lock, self.db:
            for kind, record in records:
                self.db.execute("INSERT OR REPLACE INTO records VALUES(?,?,?,?,?)",
                                (run_id, kind, record["id"], record.get("start"), dumps(record)))
            self.db.execute("INSERT OR REPLACE INTO tasks VALUES(?,?,?,?,?,?,?,?)",
                            (run_id, task["id"], task["stage"], task.get("start"), task.get("end"),
                             "completed", dumps(result), None))

    def fail_task(self, run_id: str, task: dict, error: str) -> None:
        self.execute("INSERT OR REPLACE INTO tasks VALUES(?,?,?,?,?,?,?,?)",
                     (run_id, task["id"], task["stage"], task.get("start"), task.get("end"), "failed", None, error))

    def cache(self, request_id: str) -> dict | None:
        rows = self.rows("SELECT response FROM request_cache WHERE id=?", (request_id,))
        return json.loads(rows[0]["response"]) if rows else None

    def call(self, run_id: str, invocation: str, stage: str, request_id: str,
             service: str, attempt: int, status: str, seconds: float, detail: dict) -> None:
        self.execute("INSERT INTO calls(run_id,invocation,stage,request_id,service,attempt,status,seconds,detail,created) VALUES(?,?,?,?,?,?,?,?,?,?)",
                     (run_id, invocation, stage, request_id, service, attempt, status, seconds, dumps(detail), utc_now()))
