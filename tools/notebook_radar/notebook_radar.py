#!/usr/bin/env python
"""Kaggle Notebook Radar - Web-based Public Notebook Collector and Score Ranker.

Collects public Kaggle notebooks by category/type, fetches live public LB scores,
and presents an interactive dark-mode dashboard ordered by score.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = ROOT / "state" / "notebook_radar"
DEFAULT_COMPETITION = "biohub-cell-tracking-during-development"
KAGGLE = Path(os.environ.get(
    "KAGGLE_EXE",
    r"C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/Scripts/kaggle.exe",
))
DEFAULT_SETTINGS = {
    "competition": DEFAULT_COMPETITION,
    "interval_hours": 1.0,
    "workers": 8,
    "auto_collect_enabled": False,
    "limit": 150,
    "port": 8792,
}


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def kstnow() -> str:
    now_kst = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    return now_kst.strftime("%Y. %m. %d. %H:%M:%S")


def sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return value[:100] or "unknown"


CATEGORY_META = {
    "baseline": {"name": "baseline", "label": "🏆 Baseline", "color": "#56d489", "desc": "0.945+ 검증 엔드투엔드 파이프라인 및 제출 노트북"},
    "tracking": {"name": "tracking", "label": "🔬 Tracking & Matching", "color": "#76c8ff", "desc": "DeepCenter, ILP, Hungarian matching, Lineage association"},
    "training": {"name": "training", "label": "🧠 Model Training", "color": "#efc464", "desc": "UNet3D, Node Transformer, GNN 등 모델 학습 파이프라인"},
    "postprocess": {"name": "postprocess", "label": "⚙️ Post-Process & TTA", "color": "#b78cfc", "desc": "Harmonic fusion, TTA, 앙상블, 검증자 후처리"},
    "eda": {"name": "eda", "label": "📊 EDA & Visualization", "color": "#48cae4", "desc": "데이터 탐색, 3D 비디오 뷰어, 세포 궤적 시각화"},
    "utility": {"name": "utility", "label": "🛠️ Utility & Tools", "color": "#9db0a4", "desc": "데이터셋 전처리, Wheel 패키징, 헬퍼 스크립트"},
    "general": {"name": "general", "label": "📁 General Notebook", "color": "#cfd8dc", "desc": "일반 분석 또는 미분류 공개 노트북"},
}

STATUS_META = {
    "frontier": {"name": "frontier", "label": "FRONTIER (0.947+)", "color": "#56d489", "desc": "검증된 0.947+ 최상위 프론티어 기준선"},
    "competitive": {"name": "competitive", "label": "COMPETITIVE (0.940+)", "color": "#76c8ff", "desc": "0.940 ~ 0.946 경쟁력 있는 상위권"},
    "experimental": {"name": "experimental", "label": "EXPERIMENTAL", "color": "#efc464", "desc": "0.900 ~ 0.939 또는 개발/실험 중"},
    "stale": {"name": "stale", "label": "STALE / PRE-PATCH", "color": "#ef7b74", "desc": "메트릭 패치 이전 0.950+ 잔재 또는 검증 필요"},
    "eda": {"name": "eda", "label": "EDA / TOOL", "color": "#48cae4", "desc": "점수 목적이 아닌 시각화 및 탐색 도구"},
    "no_score": {"name": "no_score", "label": "NO SCORE", "color": "#9db0a4", "desc": "제출 미생성 또는 점수 미확인"},
}


def classify_notebook(row: dict) -> dict:
    """Classify notebook into categories, status tiers, and semantic tags."""
    title = (row.get("title") or "").lower()
    slug = (row.get("slug") or "").lower()
    combined = f"{title} {slug}"
    score = row.get("best_public_score") or row.get("public_score")

    tags = []
    # Tags
    if "0.947" in combined: tags.append("0.947")
    elif "0.946" in combined: tags.append("0.946")
    elif "0.945" in combined: tags.append("0.945")
    if "deepcenter" in combined: tags.append("DeepCenter")
    if "ilp" in combined: tags.append("ILP")
    if "hungarian" in combined or "lsa" in combined: tags.append("Hungarian")
    if "unet" in combined: tags.append("UNet3D")
    if "transformer" in combined or "trans" in combined: tags.append("Transformer")
    if "tta" in combined: tags.append("TTA")
    if "harmonic" in combined or "fusion" in combined: tags.append("HarmonicFusion")
    if "lineage" in combined: tags.append("LineageForge")
    if "graph" in combined or "gnn" in combined: tags.append("Graph")
    if "detector" in combined or "point" in combined: tags.append("PointDetector")
    if "eda" in combined or "visual" in combined or "viewer" in combined: tags.append("EDA")
    if "ablation" in combined: tags.append("Ablation")
    if "ensemble" in combined or "blend" in combined: tags.append("Ensemble")
    if "dctta" in combined: tags.append("DCTTA")

    # Category determination
    if any(x in combined for x in ["eda", "visual", "viewer", "plot", "explore", "overview"]):
        category = "eda"
    elif any(x in combined for x in ["train", "fit", "epoch", "finetune", "backbone", "checkpoint"]):
        category = "training"
    elif any(x in combined for x in ["postprocess", "post-process", "tta", "harmonic", "fusion", "repair", "blend", "ensemble", "safe-division"]):
        category = "postprocess"
    elif any(x in combined for x in ["deepcenter", "ilp", "tracker", "tracking", "hungarian", "association", "lineage"]):
        category = "tracking"
    elif (score is not None and score >= 0.945) or any(x in combined for x in ["baseline", "sub", "submission", "pipeline", "0.94"]):
        category = "baseline"
    elif any(x in combined for x in ["prep", "pack", "wheel", "convert", "clean", "extract", "dataset", "generator"]):
        category = "utility"
    else:
        category = "general"

    # Status determination
    if score is not None:
        if score > 0.9485:
            # Per HANDOFF.md: Old notebooks titled >0.948 used stale pre-patch metrics
            status = "stale"
        elif score >= 0.947:
            status = "frontier"
        elif score >= 0.940:
            status = "competitive"
        else:
            status = "experimental"
    else:
        if category == "eda":
            status = "eda"
        else:
            status = "no_score"

    return {
        "category": category,
        "status": status,
        "tags": tags,
    }


_SCHEMA_INITIALIZED = False
COLLECT_LOCK = threading.Lock()


class Store:
    def __init__(self, state: Path | str = DEFAULT_STATE):
        global _SCHEMA_INITIALIZED
        self.state = Path(state).resolve()
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "listings").mkdir(exist_ok=True)
        (self.state / "pulled").mkdir(exist_ok=True)
        self.db_path = self.state / "radar.sqlite3"
        self.db = sqlite3.connect(self.db_path, timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=60000")
        if not _SCHEMA_INITIALIZED:
            self.migrate()
            _SCHEMA_INITIALIZED = True

    def close(self):
        self.db.close()

    def migrate(self):
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS meta(
          key TEXT PRIMARY KEY, value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS notebooks(
          id INTEGER PRIMARY KEY,
          ref TEXT NOT NULL UNIQUE,
          title TEXT NOT NULL,
          author TEXT NOT NULL,
          slug TEXT NOT NULL,
          url TEXT NOT NULL,
          public_score REAL,
          best_public_score REAL,
          public_votes INTEGER,
          score_checked_at TEXT,
          last_run TEXT,
          category TEXT NOT NULL DEFAULT 'general',
          status TEXT NOT NULL DEFAULT 'no_score',
          tags_json TEXT NOT NULL DEFAULT '[]',
          first_seen TEXT NOT NULL,
          last_seen TEXT NOT NULL,
          version_key TEXT,
          raw_json TEXT
        );
        CREATE TABLE IF NOT EXISTS events(
          id INTEGER PRIMARY KEY,
          created_at TEXT NOT NULL,
          level TEXT NOT NULL,
          kind TEXT NOT NULL,
          message TEXT NOT NULL,
          detail_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_notebooks_score ON notebooks(best_public_score DESC, public_score DESC);
        CREATE INDEX IF NOT EXISTS idx_notebooks_cat ON notebooks(category);
        CREATE INDEX IF NOT EXISTS idx_notebooks_status ON notebooks(status);
        """)
        self.db.commit()

    def event(self, kind: str, message: str, detail=None, level="info"):
        self.db.execute(
            "INSERT INTO events(created_at,level,kind,message,detail_json) VALUES(?,?,?,?,?)",
            (utcnow(), level, kind, message,
             json.dumps(detail, ensure_ascii=False, sort_keys=True) if detail is not None else None),
        )
        self.db.commit()

    def set_meta(self, key: str, value):
        self.db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
                        (key, json.dumps(value, ensure_ascii=False)))
        self.db.commit()

    def get_meta(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row[0])
        except Exception:
            return row[0]


def runtime_settings(store: Store) -> dict:
    saved = store.get_meta("runtime_settings", {}) or {}
    return {**DEFAULT_SETTINGS, **{k: saved[k] for k in DEFAULT_SETTINGS if k in saved}}


def run_kaggle(args: list[str], timeout=180) -> str:
    exe = KAGGLE if KAGGLE.exists() else Path("kaggle")
    env = dict(os.environ)
    if "KAGGLE_API_TOKEN" not in env:
        token_path = ROOT.parent / "api-keys" / "KAGGLE_MCP_TOKEN.txt"
        if token_path.exists():
            env["KAGGLE_API_TOKEN"] = token_path.read_text(encoding="utf-8").strip()
        # otherwise the Kaggle CLI falls back to ~/.kaggle/access_token or kaggle.json
    proc = subprocess.run([str(exe), *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout, env=env)
    if proc.returncode:
        msg = (proc.stderr or proc.stdout).strip()[-1000:]
        raise RuntimeError(f"kaggle {' '.join(args[:3])} failed: {msg}")
    return proc.stdout


def normalize_listing_item(raw: dict) -> dict:
    ref = raw.get("ref") or raw.get("id") or raw.get("kernelRef")
    if not ref or "/" not in ref:
        raise ValueError("missing notebook ref")
    author, slug = ref.split("/", 1)
    title = raw.get("title") or raw.get("kernelTitle") or slug.replace("-", " ")
    last_run = (raw.get("lastRunTime") or raw.get("last_run_time") or
                raw.get("lastUpdated") or raw.get("lastUpdatedTime") or "")
    version = (raw.get("scriptVersionId") or raw.get("versionNumber") or
               raw.get("currentVersionNumber") or raw.get("id_no"))
    stable = str(version) if version is not None else sha256(
        json.dumps({"last_run": str(last_run), "title": title}, sort_keys=True))[:16]
    score = raw.get("score") or raw.get("kernelScore")
    return {
        "ref": ref,
        "author": author,
        "slug": slug,
        "title": title,
        "url": f"https://www.kaggle.com/code/{ref}",
        "last_run": str(last_run),
        "version_key": stable,
        "public_score": float(score) if score is not None else None,
        "public_votes": int(raw["totalVotes"]) if raw.get("totalVotes") is not None else 0,
        "best_public_score": None,
        "score_checked_at": None,
        "raw": raw,
    }


def parse_listing(text: str) -> list[dict]:
    data = json.loads(text.lstrip("\ufeff"))
    if isinstance(data, dict):
        data = data.get("kernels") or data.get("items") or data.get("results") or []
    if not isinstance(data, list):
        raise ValueError("Kaggle listing is not an array")
    rows = []
    for item in data:
        try:
            rows.append(normalize_listing_item(item))
        except (TypeError, ValueError):
            continue
    return rows


def fetch_public_score(row: dict, timeout=15) -> dict:
    """Fetch live public score using Kaggle's public view model API."""
    payload = json.dumps({"authorUserName": row["author"], "kernelSlug": row["slug"],
                          "kernelVersionId": 0}).encode()
    request = urllib.request.Request(
        "https://www.kaggle.com/api/i/kernels.LegacyKernelsService/GetKernelViewModel",
        data=payload,
        headers={"accept": "application/json", "content-type": "application/json",
                 "user-agent": "Kaggle-Notebook-Radar/1.0"},
        method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        model = json.loads(response.read())
    kernel = model.get("kernel") or {}
    submission = model.get("submission") or {}
    best = model.get("bestSubmissionScore") or {}
    current = submission.get("scoreFormatted")
    best_score = best.get("scoreFormatted")
    if best_score is None:
        best_score = kernel.get("bestPublicScore")
    return {
        "public_score": float(current) if current not in (None, "") else None,
        "best_public_score": float(best_score) if best_score not in (None, "") else None,
        "public_votes": int(kernel["upvoteCount"]) if kernel.get("upvoteCount") is not None
                        else row.get("public_votes", 0),
        "score_checked_at": utcnow(),
    }


def enrich_public_scores(rows: list[dict], workers=8) -> dict:
    """Fetch scores concurrently with thread pool."""
    checked = failed = 0
    if not rows:
        return {"checked": 0, "failed": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(rows))) as pool:
        futures = {pool.submit(fetch_public_score, row): row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                row.update(future.result())
                checked += 1
            except Exception as exc:
                row["score_error"] = str(exc)[:200]
                failed += 1
    return {"checked": checked, "failed": failed}


def collect_cycle(store: Store, competition: str = None, limit: int = 150, workers: int = 8) -> dict:
    """Executes a full crawl cycle for the given competition."""
    settings = runtime_settings(store)
    comp = competition or settings.get("competition") or DEFAULT_COMPETITION
    limit = limit or settings.get("limit") or 150
    workers = workers or settings.get("workers") or 8
    now = utcnow()

    store.event("crawl_start", f"노트북 수집 시작: 대회 '{comp}', 최대 {limit}개", {"competition": comp})
    gathered = {}
    raw_dir = store.state / "listings"
    raw_dir.mkdir(exist_ok=True)

    try:
        sort_orders = ("scoreDescending", "dateRun", "voteCount")
        for sort in sort_orders:
            try:
                text = run_kaggle(["kernels", "list", "--competition", comp,
                                   "--page-size", str(limit), "--sort-by", sort, "--format", "json"])
                (raw_dir / f"{dt.datetime.now():%Y%m%d-%H%M%S}-{sort}.json").write_text(
                    text, encoding="utf-8")
                for row in parse_listing(text):
                    gathered[row["ref"]] = row
            except Exception as exc:
                store.event("crawl_warn", f"정렬 '{sort}' 수집 중 경고", {"error": str(exc)}, "warning")

        rows = list(gathered.values())
        score_res = enrich_public_scores(rows, workers=workers)

        new_count = updated_count = score_improved = 0

        # Pre-classify in memory to minimize SQLite write transaction lock time
        classified_rows = []
        for r in rows:
            cls = classify_notebook(r)
            r.update(cls)
            classified_rows.append(r)

        with store.db:
            for r in classified_rows:
                existing = store.db.execute(
                    "SELECT id, best_public_score, public_score, first_seen FROM notebooks WHERE ref=?",
                    (r["ref"],)
                ).fetchone()

                if not existing:
                    store.db.execute("""
                    INSERT INTO notebooks(ref,title,author,slug,url,public_score,best_public_score,
                      public_votes,score_checked_at,last_run,category,status,tags_json,first_seen,
                      last_seen,version_key,raw_json)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (r["ref"], r["title"], r["author"], r["slug"], r["url"],
                          r["public_score"], r["best_public_score"], r["public_votes"],
                          r["score_checked_at"], r["last_run"], r["category"], r["status"],
                          json.dumps(r["tags"], ensure_ascii=False), now, now,
                          r["version_key"], json.dumps(r["raw"], ensure_ascii=False)))
                    new_count += 1
                else:
                    old_best = existing["best_public_score"]
                    new_best = r["best_public_score"]
                    if new_best is not None and (old_best is None or new_best > old_best):
                        score_improved += 1
                        store.event("score_up", f"[{r['title']}] 스코어 상승: {old_best} -> {new_best}",
                                    {"ref": r["ref"], "old": old_best, "new": new_best})

                    store.db.execute("""
                    UPDATE notebooks SET
                      title=?, author=?, slug=?, url=?,
                      public_score=COALESCE(?, public_score),
                      best_public_score=COALESCE(?, best_public_score),
                      public_votes=COALESCE(?, public_votes),
                      score_checked_at=COALESCE(?, score_checked_at),
                      last_run=?, category=?, status=?, tags_json=?,
                      last_seen=?, version_key=?, raw_json=?
                    WHERE ref=?
                    """, (r["title"], r["author"], r["slug"], r["url"],
                          r["public_score"], r["best_public_score"], r["public_votes"],
                          r["score_checked_at"], r["last_run"], r["category"], r["status"],
                          json.dumps(r["tags"], ensure_ascii=False), now, r["version_key"],
                          json.dumps(r["raw"], ensure_ascii=False), r["ref"]))
                    updated_count += 1

        summary = {
            "competition": comp,
            "total_gathered": len(rows),
            "scores_checked": score_res["checked"],
            "scores_failed": score_res["failed"],
            "new_notebooks": new_count,
            "updated_notebooks": updated_count,
            "score_improvements": score_improved,
            "at": now,
        }
        store.set_meta("last_crawl", summary)
        store.event("crawl_done", f"수집 완료: 총 {len(rows)}개 (신규 {new_count}, 갱신 {updated_count}, 점수상승 {score_improved})", summary)
        return summary
    except Exception as exc:
        err_summary = {
            "competition": comp,
            "error": str(exc),
            "at": now,
        }
        try:
            store.set_meta("last_crawl", err_summary)
            store.event("crawl_error", f"수집 중 오류 발생: {exc}", {"error": str(exc)}, "error")
        except Exception:
            pass
        return err_summary


def dashboard_snapshot(store: Store) -> dict:
    """Prepares structured snapshot data for the web UI."""
    now_dt = dt.datetime.now(dt.timezone.utc)
    cutoff_24h = (now_dt - dt.timedelta(hours=24)).isoformat(timespec="seconds")

    cursor = store.db.execute("""
    SELECT * FROM notebooks
    ORDER BY
      CASE WHEN best_public_score IS NOT NULL THEN 0 ELSE 1 END,
      best_public_score DESC,
      public_score DESC,
      public_votes DESC,
      last_run DESC
    """)
    notebooks = []
    category_counts = {k: 0 for k in CATEGORY_META}
    category_counts["all"] = 0
    scores_valid = 0
    top_score = None
    competitive_count = 0
    new_count = 0

    for row in cursor:
        d = dict(row)
        d["tags"] = json.loads(d["tags_json"] or "[]")
        # Determine NEW badge
        is_new = bool((d["first_seen"] and d["first_seen"] >= cutoff_24h) or
                      (d["last_run"] and str(d["last_run"]) >= cutoff_24h))
        d["is_new"] = is_new
        if is_new:
            new_count += 1

        cat = d.get("category", "general")
        category_counts[cat] = category_counts.get(cat, 0) + 1
        category_counts["all"] += 1

        score = d["best_public_score"] or d["public_score"]
        if score is not None:
            scores_valid += 1
            if top_score is None or score > top_score:
                if score <= 0.9485: # filter out stale pre-patch metric
                    top_score = score
            if score >= 0.940 and score <= 0.9485:
                competitive_count += 1

        notebooks.append(d)

    events = [dict(x) for x in store.db.execute("SELECT * FROM events ORDER BY id DESC LIMIT 50")]
    settings = runtime_settings(store)

    return {
        "generated_at": utcnow(),
        "kst_time": kstnow(),
        "summary": {
            "total": len(notebooks),
            "scores_valid": scores_valid,
            "top_score": top_score if top_score is not None else 0.947,
            "competitive": competitive_count,
            "new_24h": new_count,
            "categories": category_counts,
        },
        "settings": settings,
        "category_meta": CATEGORY_META,
        "status_meta": STATUS_META,
        "notebooks": notebooks,
        "events": events,
    }


class CollectorThread(threading.Thread):
    def __init__(self, state_path: Path):
        super().__init__(name="notebook-radar-scheduler", daemon=True)
        self.state_path = Path(state_path)
        self.stop_event = threading.Event()
        self.last_run_at = None
        self.next_run_at = None
        self.last_error = None

    def snapshot(self) -> dict:
        return {
            "alive": self.is_alive(),
            "next_run_at": self.next_run_at,
            "last_run_at": self.last_run_at,
            "last_error": self.last_error,
        }

    def run(self):
        # Give the server 3 seconds to bind port and start cleanly
        time.sleep(3)
        while not self.stop_event.is_set():
            try:
                store = Store(self.state_path)
                try:
                    settings = runtime_settings(store)
                    if settings.get("auto_collect_enabled"):
                        interval_hours = float(settings.get("interval_hours", 1.0))
                        interval_sec = max(60.0, interval_hours * 3600)  # at least 1 min
                        last = store.get_meta("last_crawl")
                        now_dt = dt.datetime.now(dt.timezone.utc)
                        should_run = False

                        if not last or not last.get("at"):
                            should_run = True
                            self.next_run_at = utcnow()
                        else:
                            try:
                                last_dt = dt.datetime.fromisoformat(last["at"])
                                elapsed = (now_dt - last_dt).total_seconds()
                                if elapsed >= interval_sec:
                                    should_run = True
                                    self.next_run_at = utcnow()
                                else:
                                    remaining = interval_sec - elapsed
                                    self.next_run_at = (now_dt + dt.timedelta(seconds=remaining)).isoformat(timespec="seconds")
                            except Exception:
                                should_run = True
                                self.next_run_at = utcnow()

                        if should_run:
                            if COLLECT_LOCK.acquire(blocking=False):
                                try:
                                    self.last_run_at = utcnow()
                                    self.last_error = None
                                    collect_cycle(store)
                                    now_after = dt.datetime.now(dt.timezone.utc)
                                    self.next_run_at = (now_after + dt.timedelta(seconds=interval_sec)).isoformat(timespec="seconds")
                                except Exception as crawl_err:
                                    self.last_error = str(crawl_err)
                                    print(f"[CollectorThread Crawl Error] {crawl_err}", flush=True)
                                finally:
                                    COLLECT_LOCK.release()
                    else:
                        self.next_run_at = None
                finally:
                    store.close()
            except Exception as loop_err:
                self.last_error = str(loop_err)
                print(f"[CollectorThread Loop Error] {loop_err}", flush=True)

            # Check every 10 seconds for responsive setting changes
            self.stop_event.wait(10)

    def stop(self):
        self.stop_event.set()


class HeartbeatManager:
    """Tracks active client browser tabs and triggers auto-shutdown when all tabs are closed."""
    def __init__(self, shutdown_grace_sec: float = 12.0, initial_grace_sec: float = 40.0):
        self.lock = threading.Lock()
        self.shutdown_grace_sec = shutdown_grace_sec
        self.initial_grace_sec = initial_grace_sec
        self.start_time = time.time()
        self.clients: dict[str, float] = {}
        self.last_heartbeat_time: float | None = None
        self.has_had_client = False
        self.shutdown_reason = ""

    def ping(self, client_id: str):
        with self.lock:
            now = time.time()
            self.clients[client_id] = now
            self.last_heartbeat_time = now
            self.has_had_client = True

    def leave(self, client_id: str):
        with self.lock:
            self.clients.pop(client_id, None)

    def should_shutdown(self) -> bool:
        with self.lock:
            now = time.time()
            # Stale timeout: 8s without heartbeat removes client
            stale_threshold = now - 8.0
            self.clients = {cid: t for cid, t in self.clients.items() if t > stale_threshold}

            if not self.has_had_client:
                if (now - self.start_time) > self.initial_grace_sec:
                    self.shutdown_reason = f"초기 대기 시간({int(self.initial_grace_sec)}초) 동안 브라우저 연결이 없어 자동 종료합니다."
                    return True
                return False

            if len(self.clients) == 0:
                last_active = self.last_heartbeat_time or self.start_time
                if (now - last_active) > self.shutdown_grace_sec:
                    self.shutdown_reason = f"모든 웹 대시보드 창이 닫혀({int(self.shutdown_grace_sec)}초 대기 후) 서버를 자동 종료합니다."
                    return True

            return False

    def active_count(self) -> int:
        with self.lock:
            now = time.time()
            stale_threshold = now - 8.0
            return len([t for t in self.clients.values() if t > stale_threshold])


DASHBOARD_HTML = r'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kaggle Notebook Radar - Biohub Cell Tracking</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #09110d;
      --card: #111d17;
      --card-alt: #16251e;
      --line: #284034;
      --text: #e9f3ec;
      --muted: #9db0a4;
      --green: #56d489;
      --gold: #efc464;
      --blue: #76c8ff;
      --purple: #b78cfc;
      --cyan: #48cae4;
      --red: #ef7b74;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 14px system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    main {
      max-width: 1560px;
      margin: auto;
      padding: 24px;
    }
    header {
      margin-bottom: 16px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 28px;
      font-weight: 700;
      color: #ffffff;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .badge-live {
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(86, 212, 137, 0.2);
      color: var(--green);
      border: 1px solid var(--green);
      text-transform: uppercase;
      font-weight: 600;
    }
    .muted { color: var(--muted); }
    .subdesc {
      font-size: 13px;
      margin-top: 4px;
      line-height: 1.6;
    }

    /* Action bar */
    .bar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 18px 0;
      align-items: center;
      background: var(--card);
      padding: 12px 16px;
      border-radius: 12px;
      border: 1px solid var(--line);
    }
    button {
      background: #1f6c42;
      color: white;
      border: 0;
      border-radius: 7px;
      padding: 9px 15px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s ease;
    }
    button:hover { background: #288854; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    button.btn-secondary {
      background: #25392f;
      color: var(--text);
      border: 1px solid var(--line);
    }
    button.btn-secondary:hover { background: #324c3f; }
    button.btn-toggle-on { background: #1f6c42; }
    button.btn-toggle-off { background: #5c2c2c; }
    button.btn-toggle-off:hover { background: #733737; }

    input, select {
      background: #09110d;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 10px;
      font-size: 13px;
    }
    input[type="number"] { width: 70px; }
    input[type="text"] { width: 220px; }

    /* Summary metric cards */
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      margin: 16px 0;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px 18px;
      position: relative;
    }
    .card .metric {
      font-size: 26px;
      font-weight: 800;
      color: var(--green);
      line-height: 1.2;
    }
    .card .label {
      font-size: 13px;
      color: var(--muted);
      margin-top: 4px;
    }

    /* Category explanation legend cards */
    .legend {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 10px;
      margin: 16px 0 22px;
    }
    .legend-card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 14px;
      font-size: 12px;
    }
    .legend-card b {
      display: block;
      margin-bottom: 3px;
      font-size: 13px;
    }

    /* Category filter pill buttons */
    .filters {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 14px 0 10px;
      align-items: center;
    }
    .filter-pill {
      background: var(--card);
      border: 1px solid var(--line);
      color: var(--text);
      border-radius: 20px;
      padding: 6px 14px;
      font-size: 12px;
      cursor: pointer;
      font-weight: 500;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .filter-pill.active {
      background: #1f6c42;
      border-color: var(--green);
      color: #ffffff;
      font-weight: 700;
    }
    .pill-count {
      background: rgba(255, 255, 255, 0.15);
      border-radius: 10px;
      padding: 1px 6px;
      font-size: 11px;
    }

    /* Tabs */
    .tabs {
      display: flex;
      gap: 6px;
      margin-top: 20px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 1px;
    }
    .tabs button {
      background: transparent;
      border: 1px solid transparent;
      border-bottom: 0;
      border-radius: 8px 8px 0 0;
      color: var(--muted);
      padding: 10px 18px;
      font-size: 14px;
    }
    .tabs button.on {
      background: var(--card);
      border-color: var(--line);
      color: var(--green);
      font-weight: 700;
    }

    /* Panels */
    .panel { display: none; margin-top: 14px; }
    .panel.on { display: block; }

    /* Table */
    .table-container {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .scroll {
      max-height: 72vh;
      overflow: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      text-align: left;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }
    th {
      position: sticky;
      top: 0;
      background: #14231b;
      font-weight: 600;
      color: #ffffff;
      z-index: 10;
      font-size: 13px;
    }
    tr:hover td {
      background: rgba(86, 212, 137, 0.04);
    }
    a {
      color: var(--blue);
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }

    /* Badges */
    .badge {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
    }
    .badge-frontier { background: rgba(86, 212, 137, 0.2); color: var(--green); border: 1px solid var(--green); }
    .badge-competitive { background: rgba(118, 200, 255, 0.2); color: var(--blue); border: 1px solid var(--blue); }
    .badge-experimental { background: rgba(239, 196, 100, 0.2); color: var(--gold); border: 1px solid var(--gold); }
    .badge-stale { background: rgba(239, 123, 116, 0.2); color: var(--red); border: 1px solid var(--red); }
    .badge-eda { background: rgba(72, 202, 228, 0.2); color: var(--cyan); border: 1px solid var(--cyan); }
    .badge-no_score { background: rgba(157, 176, 164, 0.15); color: var(--muted); }

    .tag-pill {
      display: inline-block;
      padding: 1px 5px;
      margin: 1px 2px;
      border-radius: 4px;
      background: #1e3328;
      color: #9eccb0;
      font-size: 11px;
    }

    .new {
      display: inline-block;
      margin-left: 6px;
      padding: 2px 6px;
      border-radius: 999px;
      background: #efc464;
      color: #182017;
      font-size: 10px;
      font-weight: 800;
      vertical-align: middle;
    }

    .score-hl {
      font-size: 14px;
      font-weight: 700;
      color: var(--green);
    }
    .score-best {
      color: var(--gold);
      font-weight: 600;
    }

    /* Category view grid */
    .cat-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 16px;
    }
    .cat-column {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
    }
    .cat-column h3 {
      margin: 0 0 12px;
      font-size: 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--line);
      padding-bottom: 8px;
    }
    .cat-item {
      padding: 8px 0;
      border-bottom: 1px solid rgba(40, 64, 52, 0.5);
    }
    .cat-item:last-child { border-bottom: 0; }
    .cat-item .item-title {
      font-weight: 600;
      font-size: 13px;
      margin-bottom: 2px;
    }
    .cat-item .item-sub {
      font-size: 11px;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
    }

    /* Toast */
    #actionStatus {
      font-size: 13px;
      margin-left: 10px;
      color: var(--green);
      font-weight: 500;
    }
  </style>
</head>
<body>
<main>
  <header>
    <h1>
      <span>🔬 Kaggle Notebook Radar</span>
      <span class="badge-live">Live Tracker</span>
    </h1>
    <div class="muted" id="timestamp">불러오는 중…</div>
    <div class="muted subdesc">
      대회: <code>biohub-cell-tracking-during-development</code> · Kaggle 공개 노트북 유형별 실시간 수집 및 Score 정렬 대시보드입니다.<br>
      최근 24시간 내 신규 등록 또는 코드 갱신된 노트북에는 <span class="new">NEW</span> 배지가 표시됩니다.
    </div>
  </header>

  <!-- Controls Bar -->
  <div class="bar">
    <button id="collectBtn" onclick="triggerCollect()">
      <span>🔄 지금 수집</span>
    </button>
    <button id="autoCollectBtn" class="btn-toggle-off" onclick="toggleAutoCollect()">
      자동 수집 OFF · 켜기
    </button>
    <label class="muted">수집 주기(시간):
      <input id="intervalHours" type="number" min="0.25" max="72" step="0.25" value="1.0">
    </label>
    <button class="btn-secondary" onclick="saveSettings()">설정 저장</button>
    <button class="btn-secondary" onclick="refreshDashboard()">화면 새로고침</button>
    <button class="btn-secondary btn-toggle-off" onclick="shutdownServer()" title="백엔드 서버를 즉시 종료합니다">🛑 서버 종료</button>
    <span id="nextRunInfo" class="muted" style="font-size:12px; margin-left:6px; font-weight:500;"></span>
    <span id="connStatus" class="muted" style="font-size:12px; margin-left:auto; display:flex; align-items:center; gap:5px;">
      <span style="color:var(--green)">●</span> 창 닫을 시 자동 종료
    </span>
    <span id="actionStatus"></span>
  </div>

  <!-- Metric Summary Cards -->
  <div class="cards" id="summaryCards">
    <div class="card"><div class="metric" id="mTotal">—</div><div class="label">수집된 노트북</div></div>
    <div class="card"><div class="metric" id="mValid">—</div><div class="label">LB 제출 유효 점수</div></div>
    <div class="card"><div class="metric" id="mTop" style="color:var(--gold);">—</div><div class="label">최고 Frontier Score</div></div>
    <div class="card"><div class="metric" id="mCompetitive" style="color:var(--blue);">—</div><div class="label">0.940+ 상위권</div></div>
    <div class="card"><div class="metric" id="mNew" style="color:var(--gold);">—</div><div class="label">24h 신규/갱신</div></div>
  </div>

  <!-- Category Explanation Legend -->
  <div class="legend" id="legendCards">
    <div class="legend-card"><b style="color:var(--green)">🏆 Baseline</b>0.945+ 검증 엔드투엔드 파이프라인 및 핵심 제출 기준선.</div>
    <div class="legend-card"><b style="color:var(--blue)">🔬 Tracking & Matching</b>DeepCenter, ILP, Hungarian matching, Lineage repair.</div>
    <div class="legend-card"><b style="color:var(--gold)">🧠 Model Training</b>UNet3D, Node Transformer, GNN 세포 추적 모델 학습.</div>
    <div class="legend-card"><b style="color:var(--purple)">⚙️ Post-Process & TTA</b>Harmonic fusion, TTA, Test-time 앙상블 및 후처리.</div>
    <div class="legend-card"><b style="color:var(--cyan)">📊 EDA & Visualization</b>데이터셋 탐색, 3D 비디오 뷰어, 궤적 시각화 도구.</div>
  </div>

  <!-- Category Filter Bar -->
  <div class="filters">
    <span class="muted" style="font-size:13px; font-weight:600; margin-right:4px;">유형 필터:</span>
    <button class="filter-pill active" onclick="setCategoryFilter('all', this)">전체 <span class="pill-count" id="count-all">0</span></button>
    <button class="filter-pill" onclick="setCategoryFilter('baseline', this)">🏆 Baseline <span class="pill-count" id="count-baseline">0</span></button>
    <button class="filter-pill" onclick="setCategoryFilter('tracking', this)">🔬 Tracking <span class="pill-count" id="count-tracking">0</span></button>
    <button class="filter-pill" onclick="setCategoryFilter('training', this)">🧠 Training <span class="pill-count" id="count-training">0</span></button>
    <button class="filter-pill" onclick="setCategoryFilter('postprocess', this)">⚙️ Post-Process <span class="pill-count" id="count-postprocess">0</span></button>
    <button class="filter-pill" onclick="setCategoryFilter('eda', this)">📊 EDA & Tools <span class="pill-count" id="count-eda">0</span></button>
    <button class="filter-pill" onclick="setCategoryFilter('utility', this)">🛠️ Utility <span class="pill-count" id="count-utility">0</span></button>
    <div style="margin-left:auto; display:flex; gap:8px; align-items:center;">
      <input id="searchInput" type="text" placeholder="노트북 제목 또는 작성자 검색…" oninput="renderTable()">
    </div>
  </div>

  <!-- Tabs Navigation -->
  <div class="tabs">
    <button class="on" onclick="switchTab('rank', this)">📊 랭킹 (Score 순)</button>
    <button onclick="switchTab('categories', this)">📑 유형별 모아보기</button>
    <button onclick="switchTab('events', this)">📝 수집 기록 (Events)</button>
  </div>

  <!-- TAB 1: Ranked Table Panel -->
  <section id="rank" class="panel on">
    <div class="table-container scroll">
      <table>
        <thead>
          <tr>
            <th style="width: 50px;">순위</th>
            <th>공유 노트북</th>
            <th>유형 (Category)</th>
            <th>작성자</th>
            <th>게시/갱신</th>
            <th>상태</th>
            <th>현재 Score</th>
            <th>최고 Score</th>
            <th>추천</th>
            <th>주요 태그</th>
          </tr>
        </thead>
        <tbody id="notebookRows">
          <tr><td colspan="10" style="text-align:center; padding:30px;" class="muted">데이터를 불러오는 중입니다…</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- TAB 2: Categories Grid Panel -->
  <section id="categories" class="panel">
    <div class="cat-grid" id="categoryColumns">
      <!-- Injected by JavaScript -->
    </div>
  </section>

  <!-- TAB 3: Event Log Panel -->
  <section id="events" class="panel">
    <div class="table-container scroll">
      <table>
        <thead>
          <tr>
            <th style="width: 170px;">시각 (KST)</th>
            <th style="width: 110px;">구분</th>
            <th>내용</th>
          </tr>
        </thead>
        <tbody id="eventRows">
          <!-- Injected by JavaScript -->
        </tbody>
      </table>
    </div>
  </section>
</main>

<script>
let globalData = null;
let currentFilter = 'all';

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function fmtTime(s) {
  if (!s) return '—';
  let d = new Date(s);
  if (Number.isNaN(d.getTime())) return String(s).slice(0, 19).replace('T', ' ');
  return new Intl.DateTimeFormat('ko-KR', {
    timeZone: 'Asia/Seoul',
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: false
  }).format(d);
}

function switchTab(id, btn) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('on'));
  document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('on'));
  document.getElementById(id).classList.add('on');
  btn.classList.add('on');
}

function setCategoryFilter(cat, btn) {
  currentFilter = cat;
  document.querySelectorAll('.filter-pill').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderTable();
}

async function refreshDashboard() {
  try {
    let r = await fetch('/api/status', { cache: 'no-store' });
    if (!r.ok) throw new Error('API request failed');
    globalData = await r.json();
    renderAll();
  } catch (err) {
    document.getElementById('actionStatus').textContent = '데이터 로드 실패: ' + err.message;
  }
}

function renderAll() {
  if (!globalData) return;
  const d = globalData;
  const s = d.summary;

  // Header & Timestamp
  document.getElementById('timestamp').textContent = '갱신 ' + d.kst_time + ' · 대기 중';

  // Metrics
  document.getElementById('mTotal').textContent = s.total;
  document.getElementById('mValid').textContent = s.scores_valid;
  document.getElementById('mTop').textContent = s.top_score != null ? s.top_score.toFixed(3) : '—';
  document.getElementById('mCompetitive').textContent = s.competitive;
  document.getElementById('mNew').textContent = s.new_24h;

  // Filter pill counts
  const cats = s.categories || {};
  document.getElementById('count-all').textContent = s.total;
  document.getElementById('count-baseline').textContent = cats.baseline || 0;
  document.getElementById('count-tracking').textContent = cats.tracking || 0;
  document.getElementById('count-training').textContent = cats.training || 0;
  document.getElementById('count-postprocess').textContent = cats.postprocess || 0;
  document.getElementById('count-eda').textContent = cats.eda || 0;
  document.getElementById('count-utility').textContent = cats.utility || 0;

  // Settings
  const autoBtn = document.getElementById('autoCollectBtn');
  const autoOn = !!d.settings.auto_collect_enabled;
  autoBtn.textContent = autoOn ? '자동 수집 ON · 끄기' : '자동 수집 OFF · 켜기';
  autoBtn.className = autoOn ? 'btn-toggle-on' : 'btn-toggle-off';
  if (document.activeElement.id !== 'intervalHours') {
    document.getElementById('intervalHours').value = d.settings.interval_hours || 1.0;
  }

  // Render sub-components
  renderTable();
  renderCategories();
  renderEvents();
}

function renderTable() {
  if (!globalData) return;
  const q = (document.getElementById('searchInput').value || '').trim().toLowerCase();
  const list = globalData.notebooks.filter(item => {
    if (currentFilter !== 'all' && item.category !== currentFilter) return false;
    if (q) {
      const matchTitle = (item.title || '').toLowerCase().includes(q);
      const matchAuthor = (item.author || '').toLowerCase().includes(q);
      const matchTags = (item.tags || []).some(t => t.toLowerCase().includes(q));
      if (!matchTitle && !matchAuthor && !matchTags) return false;
    }
    return true;
  });

  const tbody = document.getElementById('notebookRows');
  if (list.length === 0) {
    tbody.innerHTML = '<tr><td colspan="10" style="text-align:center; padding:30px;" class="muted">조건에 일치하는 노트북이 없습니다.</td></tr>';
    return;
  }

  tbody.innerHTML = list.map((n, i) => {
    const curScore = n.public_score != null ? `<span class="score-hl">${n.public_score.toFixed(3)}</span>` : '<span class="muted">—</span>';
    const bestScore = n.best_public_score != null ? `<span class="score-best">${n.best_public_score.toFixed(3)}</span>` : '<span class="muted">—</span>';
    const dateStr = n.last_run ? String(n.last_run).slice(0, 10) : '—';
    const newBadge = n.is_new ? '<span class="new">NEW</span>' : '';
    const statusClass = 'badge-' + (n.status || 'no_score');
    const catLabel = (globalData.category_meta[n.category] || {}).label || n.category;
    const catColor = (globalData.category_meta[n.category] || {}).color || '#cfd8dc';
    const tagsHtml = (n.tags || []).map(t => `<span class="tag-pill">${esc(t)}</span>`).join('');

    return `<tr>
      <td><b>${i + 1}</b></td>
      <td>
        <a href="${esc(n.url)}" target="_blank" style="font-weight:600;">${esc(n.title)}</a>
        ${newBadge}
      </td>
      <td><span style="color:${catColor}; font-weight:600; font-size:12px;">${catLabel}</span></td>
      <td class="muted">${esc(n.author)}</td>
      <td class="muted">${esc(dateStr)}</td>
      <td><span class="badge ${statusClass}">${esc(n.status)}</span></td>
      <td>${curScore}</td>
      <td>${bestScore}</td>
      <td>👍 ${n.public_votes || 0}</td>
      <td>${tagsHtml}</td>
    </tr>`;
  }).join('');
}

function renderCategories() {
  if (!globalData) return;
  const container = document.getElementById('categoryColumns');
  const cats = ['baseline', 'tracking', 'training', 'postprocess', 'eda', 'utility'];

  container.innerHTML = cats.map(catKey => {
    const meta = globalData.category_meta[catKey] || { label: catKey, color: '#fff', desc: '' };
    const items = globalData.notebooks.filter(n => n.category === catKey).slice(0, 8);
    const count = globalData.summary.categories[catKey] || 0;

    const itemsHtml = items.length === 0
      ? '<div class="muted" style="padding:10px 0; font-size:12px;">해당 카테고리의 노트북이 아직 없습니다.</div>'
      : items.map(it => {
          const score = it.best_public_score != null ? it.best_public_score.toFixed(3) : (it.public_score != null ? it.public_score.toFixed(3) : '—');
          return `<div class="cat-item">
            <div class="item-title">
              <a href="${esc(it.url)}" target="_blank">${esc(it.title)}</a>
              ${it.is_new ? '<span class="new">NEW</span>' : ''}
            </div>
            <div class="item-sub">
              <span>작성자: ${esc(it.author)}</span>
              <span>Score: <b style="color:var(--gold);">${score}</b> (👍 ${it.public_votes || 0})</span>
            </div>
          </div>`;
        }).join('');

    return `<div class="cat-column">
      <h3 style="color:${meta.color};">
        <span>${meta.label}</span>
        <span style="font-size:12px; color:var(--muted); font-weight:normal;">${count}개</span>
      </h3>
      <div class="muted" style="font-size:11px; margin-bottom:12px;">${meta.desc}</div>
      ${itemsHtml}
    </div>`;
  }).join('');
}

function renderEvents() {
  if (!globalData) return;
  const tbody = document.getElementById('eventRows');
  if (!globalData.events || globalData.events.length === 0) {
    tbody.innerHTML = '<tr><td colspan="3" style="text-align:center; padding:20px;" class="muted">기록된 이벤트가 없습니다.</td></tr>';
    return;
  }
  tbody.innerHTML = globalData.events.map(e => {
    return `<tr>
      <td class="muted">${esc(fmtTime(e.created_at))}</td>
      <td><code>${esc(e.kind)}</code></td>
      <td>${esc(e.message)}</td>
    </tr>`;
  }).join('');
}

async function triggerCollect() {
  const btn = document.getElementById('collectBtn');
  const status = document.getElementById('actionStatus');
  btn.disabled = true;
  status.textContent = '수집 요청 중…';
  try {
    const r = await fetch('/api/collect', { method: 'POST' });
    const res = await r.json();
    status.textContent = res.message || '백그라운드 수집이 시작되었습니다.';
    pollProgress();
  } catch (err) {
    status.textContent = '수집 요청 실패: ' + err.message;
    btn.disabled = false;
  }
}

async function toggleAutoCollect() {
  const btn = document.getElementById('autoCollectBtn');
  const status = document.getElementById('actionStatus');
  btn.disabled = true;
  status.textContent = '자동 수집 설정 변경 중…';
  try {
    const r = await fetch('/api/auto-collect/toggle', { method: 'POST' });
    const res = await r.json();
    const autoOn = !!res.settings.auto_collect_enabled;
    btn.textContent = autoOn ? '자동 수집 ON · 끄기' : '자동 수집 OFF · 켜기';
    btn.className = autoOn ? 'btn-toggle-on' : 'btn-toggle-off';
    status.textContent = autoOn ? '자동 수집을 켰습니다.' : '자동 수집을 껐습니다.';
    await refreshDashboard();
  } catch (err) {
    status.textContent = '설정 변경 실패: ' + err.message;
  } finally {
    btn.disabled = false;
  }
}

async function saveSettings() {
  const btn = document.querySelector('button[onclick="saveSettings()"]');
  const status = document.getElementById('actionStatus');
  const interval = Number(document.getElementById('intervalHours').value);
  const autoBtn = document.getElementById('autoCollectBtn');
  const autoOn = autoBtn.classList.contains('btn-toggle-on');

  if (btn) btn.disabled = true;
  status.textContent = '설정 저장 중…';
  try {
    const r = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ interval_hours: interval, auto_collect_enabled: autoOn })
    });
    const res = await r.json();
    status.textContent = '저장되었습니다 (' + res.settings.interval_hours + '시간 간격)';
    await refreshDashboard();
  } catch (err) {
    status.textContent = '설정 저장 실패: ' + err.message;
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function pollProgress() {
  try {
    const r = await fetch('/api/progress');
    const res = await r.json();
    const collectBtn = document.getElementById('collectBtn');
    const autoBtn = document.getElementById('autoCollectBtn');
    const nextRun = document.getElementById('nextRunInfo');

    if (res.collecting) {
      collectBtn.disabled = true;
      collectBtn.innerHTML = '<span>⏳ 수집 진행 중…</span>';
    } else {
      collectBtn.disabled = false;
      collectBtn.innerHTML = '<span>🔄 지금 수집</span>';
    }

    if (typeof res.auto_collect_enabled === 'boolean') {
      const autoOn = res.auto_collect_enabled;
      autoBtn.textContent = autoOn ? '자동 수집 ON · 끄기' : '자동 수집 OFF · 켜기';
      autoBtn.className = autoOn ? 'btn-toggle-on' : 'btn-toggle-off';
    }

    if (res.auto_collect_enabled && res.scheduler && res.scheduler.next_run_at) {
      nextRun.textContent = '다음 자동 수집: ' + fmtTime(res.scheduler.next_run_at);
    } else if (res.auto_collect_enabled) {
      nextRun.textContent = '자동 수집 대기 중';
    } else {
      nextRun.textContent = '';
    }
  } catch (_) {}
}

// Heartbeat & Auto-shutdown synchronization
const RADAR_TAB_ID = 'tab_' + Math.random().toString(36).slice(2, 9);
async function sendHeartbeat(action = 'ping') {
  const url = `/api/heartbeat?action=${action}&client=${RADAR_TAB_ID}`;
  try {
    if (action === 'leave' && navigator.sendBeacon) {
      navigator.sendBeacon(url);
    } else {
      await fetch(url, { method: 'POST', keepalive: true });
    }
  } catch (_) {}
}

sendHeartbeat('ping');
setInterval(() => sendHeartbeat('ping'), 3500);
window.addEventListener('beforeunload', () => sendHeartbeat('leave'));
window.addEventListener('pagehide', () => sendHeartbeat('leave'));

async function shutdownServer() {
  if (!confirm('백엔드 서버를 종료하시겠습니까?\n종료 후에는 대시보드를 다시 보려면 public-radar.html을 새로 열어야 합니다.')) return;
  const status = document.getElementById('actionStatus');
  if (status) status.textContent = '서버 종료 요청 중…';
  try {
    await fetch('/api/shutdown', { method: 'POST' });
    document.body.innerHTML = `
      <div style="min-height:100vh; display:grid; place-items:center; background:#09110d; color:#e9f3ec; font:16px system-ui, -apple-system, sans-serif;">
        <div style="text-align:center; padding:36px; border:1px solid #284034; border-radius:16px; background:#111d17; box-shadow:0 18px 60px rgba(0,0,0,0.6); max-width:500px;">
          <h2 style="color:#56d489; margin:0 0 12px; font-size:24px;">🛑 서버가 안전하게 종료되었습니다</h2>
          <p style="color:#9db0a4; margin:0 0 20px; line-height:1.6;">브라우저 창이나 탭을 닫으셔도 됩니다.<br>다시 열고 싶을 땐 <code>public-radar.html</code>을 열면 바로 실행됩니다.</p>
        </div>
      </div>
    `;
  } catch (e) {
    alert('서버 종료 요청이 전송되었습니다. 창을 닫으셔도 됩니다.');
  }
}

// Initial fetch and regular polling
refreshDashboard();
setInterval(refreshDashboard, 30000);
pollProgress();
setInterval(pollProgress, 4000);
</script>
</body>
</html>'''


def serve(state=DEFAULT_STATE, host="127.0.0.1", port=8792, open_browser=False, auto_shutdown=True):
    store = Store(state)
    comp_name = runtime_settings(store).get("competition")
    store.close()

    scheduler = CollectorThread(Path(state))
    scheduler.start()

    heartbeat_mgr = HeartbeatManager(shutdown_grace_sec=12.0, initial_grace_sec=40.0)
    stop_monitor = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def send_data(self, code, data, content_type="application/json; charset=utf-8"):
            body = data if isinstance(data, bytes) else data.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            nonlocal scheduler
            if not scheduler.is_alive() and not scheduler.stop_event.is_set():
                scheduler = CollectorThread(Path(state))
                scheduler.start()

            if self.path in ("/", "/index.html"):
                self.send_data(200, DASHBOARD_HTML, "text/html; charset=utf-8")
            elif self.path == "/api/status":
                local = Store(state)
                try:
                    payload = dashboard_snapshot(local)
                    payload["scheduler"] = scheduler.snapshot()
                finally:
                    local.close()
                self.send_data(200, json.dumps(payload, ensure_ascii=False))
            elif self.path == "/api/progress":
                local = Store(state)
                try:
                    cur_settings = runtime_settings(local)
                finally:
                    local.close()
                payload = {
                    "collecting": COLLECT_LOCK.locked(),
                    "auto_collect_enabled": bool(cur_settings.get("auto_collect_enabled")),
                    "scheduler": scheduler.snapshot(),
                }
                self.send_data(200, json.dumps(payload, ensure_ascii=False))
            elif self.path.startswith("/api/heartbeat"):
                query = urllib.parse.urlparse(self.path).query
                params = urllib.parse.parse_qs(query)
                action = params.get("action", ["ping"])[0]
                client = params.get("client", ["default"])[0]
                if action == "leave":
                    heartbeat_mgr.leave(client)
                else:
                    heartbeat_mgr.ping(client)
                self.send_data(200, json.dumps({
                    "status": "ok",
                    "action": action,
                    "active_clients": heartbeat_mgr.active_count()
                }))
            else:
                self.send_data(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            if self.path == "/api/collect":
                if not COLLECT_LOCK.acquire(blocking=False):
                    return self.send_data(409, json.dumps({"message": "이미 수집 중입니다."}, ensure_ascii=False))

                def bg_task():
                    try:
                        local = Store(state)
                        try:
                            collect_cycle(local)
                        finally:
                            local.close()
                    finally:
                        COLLECT_LOCK.release()

                threading.Thread(target=bg_task, daemon=True).start()
                return self.send_data(202, json.dumps({"message": "백그라운드 수집을 시작했습니다."}, ensure_ascii=False))

            elif self.path == "/api/settings":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    local = Store(state)
                    try:
                        settings = runtime_settings(local)
                        if "interval_hours" in payload:
                            settings["interval_hours"] = max(0.25, float(payload["interval_hours"]))
                        if "competition" in payload:
                            settings["competition"] = str(payload["competition"]).strip()
                        if "auto_collect_enabled" in payload:
                            settings["auto_collect_enabled"] = bool(payload["auto_collect_enabled"])
                        local.set_meta("runtime_settings", settings)
                    finally:
                        local.close()
                    return self.send_data(200, json.dumps({"settings": settings}, ensure_ascii=False))
                except Exception as exc:
                    return self.send_data(400, json.dumps({"error": str(exc)}, ensure_ascii=False))

            elif self.path == "/api/auto-collect/toggle":
                local = Store(state)
                try:
                    settings = runtime_settings(local)
                    settings["auto_collect_enabled"] = not bool(settings.get("auto_collect_enabled"))
                    local.set_meta("runtime_settings", settings)
                finally:
                    local.close()
                return self.send_data(200, json.dumps({"settings": settings}, ensure_ascii=False))

            elif self.path.startswith("/api/heartbeat"):
                query = urllib.parse.urlparse(self.path).query
                params = urllib.parse.parse_qs(query)
                action = params.get("action", ["ping"])[0]
                client = params.get("client", ["default"])[0]
                if action == "leave":
                    heartbeat_mgr.leave(client)
                else:
                    heartbeat_mgr.ping(client)
                self.send_data(200, json.dumps({
                    "status": "ok",
                    "action": action,
                    "active_clients": heartbeat_mgr.active_count()
                }))

            elif self.path == "/api/shutdown":
                def do_shutdown():
                    time.sleep(0.3)
                    server.shutdown()
                threading.Thread(target=do_shutdown, daemon=True).start()
                return self.send_data(200, json.dumps({"status": "shutting_down", "message": "서버를 종료합니다."}, ensure_ascii=False))

            else:
                return self.send_data(404, json.dumps({"error": "not found"}))

        def log_message(self, fmt, *args):
            pass

    print(f"\n==================================================================")
    print(f"  Kaggle Notebook Radar Web Dashboard")
    print(f"  URL: http://{host}:{port}")
    print(f"  Competition: {comp_name}")
    print(f"  State Directory: {state}")
    if auto_shutdown:
        print(f"  Auto-Shutdown: 활성화 (브라우저 창 닫을 시 자동 종료)")
    print(f"==================================================================\n", flush=True)

    if open_browser:
        def launch():
            time.sleep(1.2)
            import webbrowser
            webbrowser.open(f"http://{host}:{port}")
        threading.Thread(target=launch, daemon=True).start()

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        if getattr(exc, 'winerror', None) == 10048 or "already in use" in str(exc).lower() or getattr(exc, 'errno', None) == 98:
            print(f"[알림] 서버가 이미 http://{host}:{port} 에서 실행 중입니다!")
            if open_browser:
                import webbrowser
                webbrowser.open(f"http://{host}:{port}")
            scheduler.stop()
            return
        scheduler.stop()
        raise

    if auto_shutdown:
        def monitor_loop():
            while not stop_monitor.is_set():
                time.sleep(1.5)
                if heartbeat_mgr.should_shutdown():
                    print(f"\n[자동 종료] {heartbeat_mgr.shutdown_reason}", flush=True)
                    threading.Thread(target=server.shutdown, daemon=True).start()
                    break

        monitor_t = threading.Thread(target=monitor_loop, daemon=True)
        monitor_t.start()

    try:
        server.serve_forever()
    finally:
        stop_monitor.set()
        scheduler.stop()
        server.server_close()
        print("[종료 완료] 백엔드 서버가 안전하게 종료되었습니다.", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    sub = ap.add_subparsers(dest="command", required=True)

    crawl = sub.add_parser("crawl")
    crawl.add_argument("--competition", default=None)
    crawl.add_argument("--limit", type=int, default=100)
    crawl.add_argument("--workers", type=int, default=8)

    web = sub.add_parser("serve")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8792)
    web.add_argument("--no-browser", action="store_true")
    web.add_argument("--no-auto-shutdown", action="store_true", help="브라우저 창을 닫아도 서버를 자동 종료하지 않고 상시 유지")

    sub.add_parser("status")

    args = ap.parse_args(argv)

    if args.command == "crawl":
        store = Store(args.state)
        res = collect_cycle(store, competition=args.competition, limit=args.limit, workers=args.workers)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        store.close()

    elif args.command == "serve":
        serve(args.state, args.host, args.port, open_browser=not args.no_browser, auto_shutdown=not args.no_auto_shutdown)

    elif args.command == "status":
        store = Store(args.state)
        snap = dashboard_snapshot(store)
        print(f"총 수집: {snap['summary']['total']}개 (유효점수 {snap['summary']['scores_valid']}개, 최고점수 {snap['summary']['top_score']})")
        store.close()


if __name__ == "__main__":
    main()
