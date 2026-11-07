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
        CREATE TABLE IF NOT EXISTS query_snapshots(
          run_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
          meta_json TEXT NOT NULL, entities_json TEXT NOT NULL, reviews_json TEXT NOT NULL,
          updated TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS query_entity_mentions(
          run_id TEXT NOT NULL, entity_id TEXT NOT NULL,
          entity_json TEXT NOT NULL, mentions_json TEXT NOT NULL,
          PRIMARY KEY(run_id,entity_id));
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

    def _invalidate_query_snapshot(self, run_id: str) -> None:
        self.db.execute("DELETE FROM query_snapshots WHERE run_id=?", (run_id,))
        self.db.execute("DELETE FROM query_entity_mentions WHERE run_id=?", (run_id,))

    def query_fingerprint(self, run_id: str) -> str:
        import hashlib
        from .util import digest

        with self.lock:
            run_row = self.db.execute("SELECT source_id, state FROM runs WHERE id=?", (run_id,)).fetchone()
            if run_row is None:
                raise ValueError("运行编号不存在")
            src_row = self.db.execute(
                "SELECT metadata, CAST(text AS BLOB) AS text_bytes, raw FROM sources WHERE id=?",
                (run_row["source_id"],),
            ).fetchone()
            if src_row is None:
                raise ValueError("原文版本不存在")
            metadata = json.loads(src_row["metadata"])
            raw_sha = hashlib.sha256(src_row["raw"]).hexdigest()
            text_sha = hashlib.sha256(src_row["text_bytes"]).hexdigest()
            if raw_sha != metadata["raw_sha256"] or text_sha != metadata["text_sha256"]:
                raise ValueError("保存的原文指纹不匹配，停止导出")
            rec_stats_row = self.db.execute(
                "SELECT COALESCE(MAX(rowid), 0) AS mr, COUNT(*) AS cnt "
                "FROM records WHERE run_id=? AND kind != 'invocation'",
                (run_id,),
            ).fetchone()
            corr_rows = self.db.execute(
                "SELECT id, start, data FROM records WHERE run_id=? AND kind='correction' ORDER BY start, id",
                (run_id,),
            ).fetchall()
            roster_rows = self.db.execute(
                "SELECT rowid, id FROM records WHERE run_id=? AND kind='roster_output' ORDER BY start, id",
                (run_id,),
            ).fetchall()
        state_obj = json.loads(run_row["state"])
        state_for_fp = {
            k: state_obj.get(k)
            for k in ("chapters", "discovery", "identity")
        }
        return digest([
            "query-snapshot-v1",
            raw_sha,
            text_sha,
            state_for_fp,
            int(rec_stats_row["mr"]) if rec_stats_row else 0,
            int(rec_stats_row["cnt"]) if rec_stats_row else 0,
            [(r["id"], r["start"], r["data"]) for r in corr_rows],
            [(int(r["rowid"]), r["id"]) for r in roster_rows],
        ])

    def save_query_snapshot(
        self,
        run_id: str,
        fingerprint: str,
        meta_obj: dict,
        entities_obj: list[dict],
        reviews_obj: list[dict],
        entity_mentions_rows: list[tuple[str, str, str]],
    ) -> None:
        now = utc_now()
        meta_s = json.dumps(meta_obj, ensure_ascii=False, separators=(",", ":"))
        entities_s = json.dumps(entities_obj, ensure_ascii=False, separators=(",", ":"))
        reviews_s = json.dumps(reviews_obj, ensure_ascii=False, separators=(",", ":"))
        try:
            with self.lock, self.db:
                self.db.execute("DELETE FROM query_entity_mentions WHERE run_id=?", (run_id,))
                self.db.execute(
                    "INSERT OR REPLACE INTO query_snapshots(run_id,fingerprint,meta_json,entities_json,reviews_json,updated) "
                    "VALUES(?,?,?,?,?,?)",
                    (run_id, fingerprint, meta_s, entities_s, reviews_s, now),
                )
                if entity_mentions_rows:
                    self.db.executemany(
                        "INSERT INTO query_entity_mentions(run_id,entity_id,entity_json,mentions_json) VALUES(?,?,?,?)",
                        [(run_id, eid, e_json, m_json) for eid, e_json, m_json in entity_mentions_rows],
                    )
        except sqlite3.OperationalError:
            pass

    def state(self, run_id: str, **updates: Any) -> None:
        with self.lock, self.db:
            state = self.run(run_id)["state"]
            state.update(updates)
            self.db.execute("UPDATE runs SET state=?,updated=? WHERE id=?", (dumps(state), utc_now(), run_id))
            if any(k in updates for k in ("chapters", "discovery", "identity")):
                self._invalidate_query_snapshot(run_id)

    def put(self, run_id: str, kind: str, record: dict) -> None:
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO records VALUES(?,?,?,?,?)",
                (run_id, kind, record["id"], record.get("start"), dumps(record)),
            )
            if kind != "invocation":
                self._invalidate_query_snapshot(run_id)

    def records(self, run_id: str, kind: str) -> list[dict]:
        return [json.loads(r["data"]) for r in self.rows(
            "SELECT data FROM records WHERE run_id=? AND kind=? ORDER BY start,id", (run_id, kind))]

    def corrections(self, run_id: str) -> list[dict]:
        return self.records(run_id, "correction")

    def append_correction(self, run_id: str, build_record) -> dict:
        with self.lock, self.db:
            row = self.db.execute(
                "SELECT COALESCE(MAX(start), 0) AS max_seq FROM records WHERE run_id=? AND kind='correction'",
                (run_id,),
            ).fetchone()
            seq = int(row["max_seq"]) + 1
            record = build_record(seq)
            self.db.execute(
                "INSERT INTO records VALUES(?,?,?,?,?)",
                (run_id, "correction", record["id"], seq, dumps(record)),
            )
            self._invalidate_query_snapshot(run_id)
            return record

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
            if records:
                self._invalidate_query_snapshot(run_id)

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
