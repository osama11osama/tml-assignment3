#!/usr/bin/env python3
"""
Desktop GUI for Assignment 3 — Adversarial Robustness submissions.

Adapted from Assignment 2 Stolen Model Detection GUI (same server, shared API key).

  - Lists .pt checkpoints from results/checkpoints (configurable scan folder)
  - Validate state_dict + architecture, then upload with model_name field
  - Leaderboard tab: Robustness (03_robustness)
  - Shows clean / robust / unified % (from progress.json, local eval, or server response)
  - Distinguishes HTTP 200 upload vs evaluation passed/failed; smart cooldown (60m scored / 2m rejected)
  - "Reset cooldown (local)" clears GUI timer only
  - History + queue + top-3 track under %APPDATA%\\tml_submit_gui\\task3\\

Run from Assignment3 root:

    python _private/tools/tml_submit_gui.py

Or:

    powershell -File _private/tools/launch_submit_gui.ps1
"""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.model import ALLOWED_ARCHITECTURES, assert_model_io, make_model

BASE_URL = "http://34.63.153.158"
LEADERBOARD_PAGE = f"{BASE_URL}/leaderboard_page"
TASK_ID = "03-robustness"
LEADERBOARD_TAB_ID = "03_robustness"
DEFAULT_COOLDOWN_S = 3600
DEFAULT_TEAM_NAME = "team_XLVII"
DEFAULT_SCAN_SUBDIR = "results/checkpoints"
DEFAULT_MODEL_NAME = "resnet18"
CONFIG_SUBDIR = "task3"
LEADERBOARD_REFETCH_DELAY_S = 2.0
LB_SCORE_EPS = 1e-8
_WAIT_SECONDS_RE = re.compile(r"wait\s+(\d+)\s+seconds?", re.I)
MIN_CLEAN_ACC = 0.50
FAILED_EVAL_COOLDOWN_S = 120
SUCCESS_EVAL_COOLDOWN_S = DEFAULT_COOLDOWN_S
TASK_INFO_LINES = (
    "Task 03 — Adversarial Robustness: upload a .pt state_dict (resnet18 / resnet34 / resnet50).",
    "Server metric: Score = 0.5 × clean_accuracy + 0.5 × robustness_accuracy (PGD attack on hidden test set).",
    "Gate: clean accuracy must be > 50% or the submission is rejected (HTTP 200 ≠ passed evaluation).",
    "Cooldown: 60 min after a scored submission; ~2 min after a rejected one. Local timer can be reset below.",
    "Train ERM / FGSM-AT / PGD-AT first — do not submit template_sanity.pt (random weights).",
)


def _config_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home())
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    p = Path(base) / "tml_submit_gui" / CONFIG_SUBDIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def _legacy_settings_paths() -> list[Path]:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home())
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    root = Path(base) / "tml_submit_gui"
    return [root / "settings.json", root / "task2" / "settings.json"]


def _settings_path() -> Path:
    return _config_dir() / "settings.json"


def _history_path() -> Path:
    return _config_dir() / "submit_history.jsonl"


def _queue_path() -> Path:
    return _config_dir() / "submit_queue.json"


def _queue_events_path() -> Path:
    return _config_dir() / "submit_queue_events.jsonl"


def _top3_history_path() -> Path:
    return _config_dir() / "leaderboard_top3_history.jsonl"


def _local_metrics_cache_path() -> Path:
    return _config_dir() / "local_metrics_cache.json"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_settings() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "api_key": "",
        "cooldown_seconds": DEFAULT_COOLDOWN_S,
        "scan_subdir": DEFAULT_SCAN_SUBDIR,
        "team_name": DEFAULT_TEAM_NAME,
        "default_model_name": DEFAULT_MODEL_NAME,
        "auto_submit_queue": False,
    }
    p = _settings_path()
    if p.is_file():
        try:
            disk = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(disk, dict):
                defaults.update(disk)
        except json.JSONDecodeError:
            pass
    if not defaults.get("api_key"):
        for legacy in _legacy_settings_paths():
            if not legacy.is_file():
                continue
            try:
                old = json.loads(legacy.read_text(encoding="utf-8"))
                if isinstance(old, dict) and old.get("api_key"):
                    defaults["api_key"] = str(old["api_key"]).strip()
                    break
            except json.JSONDecodeError:
                continue
    return defaults


def save_settings(d: dict[str, Any]) -> None:
    _settings_path().write_text(json.dumps(d, indent=2), encoding="utf-8")


def append_history(entry: dict[str, Any]) -> None:
    entry.setdefault("entry_uuid", str(uuid.uuid4()))
    with _history_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_history_rows(max_rows: int = 500) -> list[dict[str, Any]]:
    p = _history_path()
    if not p.is_file():
        return []
    out: list[dict[str, Any]] = []
    for ln in p.read_text(encoding="utf-8").strip().splitlines()[-max_rows:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return list(reversed(out))


def append_queue_event(entry: dict[str, Any]) -> None:
    entry.setdefault("event_uuid", str(uuid.uuid4()))
    entry.setdefault("ts_utc", utc_now())
    with _queue_events_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_queue_events(max_rows: int = 500) -> list[dict[str, Any]]:
    p = _queue_events_path()
    if not p.is_file():
        return []
    out: list[dict[str, Any]] = []
    for ln in p.read_text(encoding="utf-8").strip().splitlines()[-max_rows:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return list(reversed(out))


def load_submit_queue() -> list[dict[str, Any]]:
    p = _queue_path()
    if not p.is_file():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        model_path = str(item.get("model_path", "")).strip()
        if not model_path:
            continue
        out.append(
            {
                "queue_id": str(item.get("queue_id") or uuid.uuid4()),
                "model_path": model_path,
                "model_basename": str(item.get("model_basename") or Path(model_path).name),
                "model_name": str(item.get("model_name") or DEFAULT_MODEL_NAME),
                "added_ts_utc": str(item.get("added_ts_utc") or utc_now()),
                "attempts": max(0, int(item.get("attempts", 0) or 0)),
                "last_event": str(item.get("last_event") or "Queued"),
                "last_event_ts_utc": str(item.get("last_event_ts_utc") or utc_now()),
            }
        )
    return out


def save_submit_queue(items: list[dict[str, Any]]) -> None:
    _queue_path().write_text(json.dumps(items, indent=2), encoding="utf-8")


def _normalize_path_str(path_like: str | Path) -> str:
    try:
        return os.path.normcase(os.path.normpath(str(Path(path_like).resolve())))
    except Exception:
        return os.path.normcase(os.path.normpath(str(path_like)))


def _empty_submit_state() -> dict[str, Any]:
    return {
        "attempts": 0,
        "accepted": 0,
        "scored": 0,
        "rejected": 0,
        "last_http_status": None,
        "last_ts_utc": "",
        "last_submission_id": "",
        "last_no_change": False,
        "last_eval_status": "",
        "last_clean_acc": None,
        "last_robust_acc": None,
        "last_unified_score": None,
        "last_lb_score": None,
        "last_message": "",
    }


def fmt_pct(v: Any, *, na: str = "—") -> str:
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "unknown", "none", "—")):
        return na
    try:
        x = float(v)
        if 0.0 <= x <= 1.0:
            return f"{x * 100:.2f}%"
        if 0.0 < x <= 100.0:
            return f"{x:.2f}%"
        return f"{x:.4f}"
    except (TypeError, ValueError):
        return na


def _extract_metric(body: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k not in body or body[k] is None:
            continue
        try:
            return float(body[k])
        except (TypeError, ValueError):
            continue
    return None


def parse_server_eval(body: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "eval_status": "unknown",
        "clean_acc": None,
        "robust_acc": None,
        "unified_score": None,
        "message": "",
        "submission_id": "",
    }
    if not isinstance(body, dict):
        return out
    out["submission_id"] = str(body.get("submission_id", body.get("id", "")) or "")
    out["message"] = str(body.get("message", "") or "")
    if not out["message"] and isinstance(body.get("detail"), str):
        out["message"] = str(body["detail"])
    pre_eval = str(body.get("eval_status", "")).strip().lower()
    if pre_eval in ("passed", "failed"):
        out["eval_status"] = pre_eval
    status = str(body.get("status", "")).strip().lower()
    if status in ("failed", "rejected", "error"):
        out["eval_status"] = "failed"
    elif status in ("success", "ok", "completed", "passed", "accepted"):
        out["eval_status"] = "passed"
    elif status == "local_reset":
        out["eval_status"] = "unknown"
    msg_l = out["message"].lower()
    if out["eval_status"] == "unknown":
        if any(k in msg_l for k in ("reject", "must be greater", "submission failed")):
            out["eval_status"] = "failed"
        elif any(k in msg_l for k in ("successfully", "check the leaderboard", "evaluated successfully")):
            out["eval_status"] = "passed"
    out["clean_acc"] = _extract_metric(
        body,
        ("clean_accuracy", "clean_acc", "clean", "accuracy_clean"),
    )
    out["robust_acc"] = _extract_metric(
        body,
        ("robustness_accuracy", "robust_accuracy", "robust_acc", "robust", "accuracy_robust"),
    )
    out["unified_score"] = _extract_metric(
        body,
        ("unified_score", "leaderboard_score", "score", "metric"),
    )
    if out["unified_score"] is None and out["clean_acc"] is not None and out["robust_acc"] is not None:
        out["unified_score"] = 0.5 * float(out["clean_acc"]) + 0.5 * float(out["robust_acc"])
    for nest in ("result", "data", "submission", "metrics", "evaluation"):
        nested = body.get(nest)
        if not isinstance(nested, dict):
            continue
        inner = parse_server_eval(nested)
        for key in ("eval_status", "clean_acc", "robust_acc", "unified_score", "message", "submission_id"):
            if out.get(key) in (None, "", "unknown") and inner.get(key) not in (None, "", "unknown"):
                out[key] = inner[key]
    return out


def eval_cooldown_for_response(http_status: int, body: dict[str, Any], default: int) -> tuple[int, str]:
    if http_status == 429:
        return extract_cooldown_seconds(None, body, default), "rate_limit"
    if http_status != 200:
        return 0, "http_error"
    ev = parse_server_eval(body)
    if ev["eval_status"] == "passed":
        return SUCCESS_EVAL_COOLDOWN_S, "eval_passed"
    if ev["eval_status"] == "failed":
        return FAILED_EVAL_COOLDOWN_S, "eval_failed"
    if ev.get("message"):
        return FAILED_EVAL_COOLDOWN_S, "eval_failed_message"
    return SUCCESS_EVAL_COOLDOWN_S, "http_200_unknown"


def load_local_metrics_cache() -> dict[str, dict[str, Any]]:
    p = _local_metrics_cache_path()
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def save_local_metrics_cache(cache: dict[str, dict[str, Any]]) -> None:
    _local_metrics_cache_path().write_text(json.dumps(cache, indent=2), encoding="utf-8")


def discover_progress_metrics() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    runs_dir = REPO_ROOT / "results" / "runs"
    if not runs_dir.is_dir():
        return out
    for prog in runs_dir.glob("*/progress.json"):
        try:
            data = json.loads(prog.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        metrics = {
            "clean_acc": data.get("final_clean_acc", data.get("best_clean_acc")),
            "robust_acc": data.get("final_robust_acc"),
            "unified_score": data.get("final_unified_score"),
            "epochs": data.get("epochs_completed"),
            "source": f"progress:{prog.parent.name}",
        }
        for key in ("submit_copy", "best_checkpoint", "resume_checkpoint"):
            raw_path = str(data.get(key, "")).strip()
            if raw_path:
                out[_normalize_path_str(raw_path)] = metrics
    return out


def lookup_checkpoint_metrics(
    model_path: str | Path,
    progress_map: dict[str, dict[str, Any]] | None = None,
    cache_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    key = _normalize_path_str(model_path)
    progress_map = progress_map if progress_map is not None else discover_progress_metrics()
    cache_map = cache_map if cache_map is not None else load_local_metrics_cache()
    if key in cache_map:
        merged = dict(cache_map[key])
        merged.setdefault("source", "local_eval")
        return merged
    if key in progress_map:
        return dict(progress_map[key])
    return {}


def metrics_summary_text(metrics: dict[str, Any], submit_state: dict[str, Any] | None = None) -> str:
    if not metrics and not submit_state:
        return "Metrics: unknown — run Local eval or check training progress.json"
    parts: list[str] = []
    if metrics:
        src = str(metrics.get("source", "unknown"))
        clean = fmt_pct(metrics.get("clean_acc"))
        robust = fmt_pct(metrics.get("robust_acc"))
        local_uni = fmt_pct(metrics.get("unified_score"))
        quick = " (quick robust)" if metrics.get("quick_robust") else ""
        parts.append(f"Local val ({src}{quick}): clean {clean} | robust {robust} | est. {local_uni}")
        clean_v = metrics.get("clean_acc")
        if isinstance(clean_v, (int, float)):
            parts.append("GATE OK" if float(clean_v) > MIN_CLEAN_ACC else "GATE FAIL (<50% clean)")
    if submit_state:
        last = str(submit_state.get("last_eval_status", "")).strip().lower()
        lb = submit_state.get("last_lb_score", submit_state.get("last_unified_score"))
        if last == "passed" and lb is not None:
            parts.append(f"Server LB score: {_fmt_lb(lb)} (rank check: Update leaderboard)")
        elif last == "failed":
            parts.append(f"Last submit: rejected — {str(submit_state.get('last_message', ''))[:120]}")
    return " | ".join(parts) if parts else "Metrics: —"


def run_local_eval_checkpoint(path: Path, model_name: str, *, quick: bool = True) -> dict[str, Any]:
    from src.data import make_loaders
    from src.eval_utils import accuracy, robust_accuracy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    model = make_model(model_name)
    model.load_state_dict(state, strict=True)
    model.to(device)
    _, val_loader = make_loaders(batch_size=256, num_workers=0)
    clean = accuracy(model, val_loader, device)
    robust = robust_accuracy(
        model,
        val_loader,
        device,
        max_batches=12 if quick else None,
    )
    unified = 0.5 * clean + 0.5 * robust
    return {
        "clean_acc": clean,
        "robust_acc": robust,
        "unified_score": unified,
        "quick_robust": quick,
        "source": "local_eval",
        "ts_utc": utc_now(),
    }


def build_submit_history_index(max_rows: int = 4000) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_path: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for row in reversed(load_history_rows(max_rows)):
        model_path = str(row.get("model_path", "")).strip()
        if not model_path or str(row.get("submission_trigger", "")) == "local_reset":
            continue
        model_name = str(row.get("model_basename", "")).strip() or Path(model_path).name
        key_path = _normalize_path_str(model_path) if model_path else ""
        key_name = model_name.lower()
        state = by_path.get(key_path) or by_name.get(key_name) or _empty_submit_state()
        state["attempts"] = int(state.get("attempts", 0)) + 1
        try:
            http_status = int(row.get("http_status", 0))
        except (TypeError, ValueError):
            http_status = 0
        if http_status == 200:
            state["accepted"] = int(state.get("accepted", 0)) + 1
        state["last_http_status"] = http_status or state.get("last_http_status")
        state["last_ts_utc"] = str(row.get("ts_utc", "")) or str(state.get("last_ts_utc", ""))
        state["last_no_change"] = bool(row.get("leaderboard_score_unchanged", False))
        row_eval = str(row.get("eval_status", "")).strip().lower()
        ev = parse_server_eval(row.get("response") or row.get("server_eval"))
        if row_eval in ("passed", "failed"):
            ev["eval_status"] = row_eval
        lb_after = row.get("leaderboard_score_after")
        if ev["eval_status"] == "passed" and lb_after not in (None, "", "unknown"):
            try:
                ev["unified_score"] = float(lb_after)
                state["last_lb_score"] = float(lb_after)
            except (TypeError, ValueError):
                pass
        if ev["eval_status"] == "passed":
            state["scored"] = int(state.get("scored", 0)) + 1
        elif ev["eval_status"] == "failed":
            state["rejected"] = int(state.get("rejected", 0)) + 1
        state["last_eval_status"] = ev["eval_status"] or state.get("last_eval_status", "")
        for k in ("clean_acc", "robust_acc", "unified_score"):
            if ev.get(k) is not None:
                state[f"last_{k}"] = ev[k]
        if ev.get("message"):
            state["last_message"] = ev["message"]
        body = row.get("response")
        if isinstance(body, dict):
            sid = str(body.get("submission_id", body.get("id", row.get("server_submission_id", ""))))
            if sid:
                state["last_submission_id"] = sid
        if key_path:
            by_path[key_path] = state
        if key_name:
            by_name[key_name] = state
    return by_path, by_name


def summarize_submit_state(
    model_path: str | Path,
    path_index: dict[str, dict[str, Any]],
    name_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    key_path = _normalize_path_str(model_path)
    key_name = Path(str(model_path)).name.lower()
    state = path_index.get(key_path) or name_index.get(key_name)
    return state if state else _empty_submit_state()


def submit_state_label(state: dict[str, Any]) -> str:
    last = str(state.get("last_eval_status", "")).strip().lower()
    if last == "passed" or int(state.get("scored", 0)) > 0:
        return "Scored"
    if last == "failed" or int(state.get("rejected", 0)) > 0:
        return "Rejected"
    if int(state.get("accepted", 0)) > 0:
        return "Uploaded"
    if int(state.get("attempts", 0)) > 0:
        return "Tried only"
    return "Not submitted"


def submit_state_detail_text(state: dict[str, Any]) -> str:
    attempts = int(state.get("attempts", 0))
    last_http = state.get("last_http_status")
    last_ts = str(state.get("last_ts_utc", "")).strip() or "—"
    eval_st = str(state.get("last_eval_status", "")).strip() or "—"
    clean = fmt_pct(state.get("last_clean_acc"))
    robust = fmt_pct(state.get("last_robust_acc"))
    unified = fmt_pct(state.get("last_unified_score"))
    msg = str(state.get("last_message", "")).strip()
    if attempts == 0:
        return "History: not submitted yet."
    base = (
        f"History: {attempts} attempt(s); last HTTP {last_http or '—'}; eval {eval_st}; "
        f"server clean {clean} | robust {robust} | unified {unified} @ {last_ts}."
    )
    if msg:
        base += f" Message: {msg[:180]}"
    return base


def parse_ts_utc_to_epoch(ts: str) -> float | None:
    s = (ts or "").strip().replace("Z", "")[:19]
    try:
        return float(calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%S")))
    except ValueError:
        return None


def parse_wait_seconds_from_text(text: str) -> int | None:
    m = _WAIT_SECONDS_RE.search(text or "")
    return int(m.group(1)) if m else None


def cooldown_until_from_history(http_status: int) -> float:
    for row in load_history_rows(500):
        if row.get("http_status") != http_status:
            continue
        ep = parse_ts_utc_to_epoch(str(row.get("ts_utc", "")))
        if ep is None:
            continue
        wait: int | None = None
        raw = row.get("cooldown_applied_seconds")
        if isinstance(raw, (int, float)) and raw > 0:
            wait = int(raw)
        if wait is None and isinstance(row.get("response"), dict):
            wait, _ = eval_cooldown_for_response(
                int(row.get("http_status") or 0),
                row["response"],
                DEFAULT_COOLDOWN_S,
            )
        if wait is None:
            wait = DEFAULT_COOLDOWN_S if http_status == 200 else 0
        if wait > 0:
            until = ep + float(wait)
            if until > time.time():
                return until
    return 0.0


def validate_checkpoint_file(path: Path, model_name: str) -> None:
    if model_name not in ALLOWED_ARCHITECTURES:
        raise ValueError(f"model_name must be one of {ALLOWED_ARCHITECTURES}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("File must be a state_dict (torch.save(model.state_dict(), path))")
    model = make_model(model_name)
    model.load_state_dict(state, strict=True)
    assert_model_io(model, device="cpu")


def discover_checkpoints(scan_rel: str) -> list[str]:
    root = (REPO_ROOT / scan_rel).resolve()
    if not root.is_dir():
        return []
    return sorted(str(p) for p in root.glob("*.pt"))


def _fmt_lb(v: Any) -> str:
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "unknown", "none")):
        return "unknown"
    try:
        return f"{float(v):.6f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(v)


def fetch_leaderboard_html(url: str = LEADERBOARD_PAGE, timeout: float = 25.0) -> tuple[str | None, str | None]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text, None
    except Exception as e:
        return None, str(e)


def parse_leaderboard_rank_table(html: str) -> tuple[list[tuple[int, str, float]], str | None]:
    tab_idx = html.find(f'id="{LEADERBOARD_TAB_ID}"')
    search_from = tab_idx if tab_idx >= 0 else 0
    idx = html.find("<th>Rank</th>", search_from)
    if idx < 0:
        return [], "Could not find Rank column (layout changed?)"
    start = html.rfind("<table", search_from, idx)
    end = html.find("</table>", idx)
    if start < 0 or end < 0:
        return [], "Could not parse leaderboard table."
    table_html = html[start : end + len("</table>")]
    tbody_m = re.search(r"<tbody>(.*?)</tbody>", table_html, re.I | re.S)
    if not tbody_m:
        return [], "Table has no tbody."
    out: list[tuple[int, str, float]] = []
    for tr in re.finditer(r"<tr>(.*?)</tr>", tbody_m.group(1), re.I | re.S):
        block = tr.group(1)
        chip = re.search(r'class="chip">(\d+)</span>', block)
        team_m = re.search(r"\b(team_[A-Z]+)\b", block)
        score_m = re.search(r'class="score-cell"[^>]*>\s*([-+0-9.eE]+)\s*</td>', block)
        if chip and team_m and score_m:
            try:
                out.append((int(chip.group(1)), team_m.group(1), float(score_m.group(1))))
            except ValueError:
                continue
    out.sort(key=lambda x: x[0])
    return out, None if out else "No rows parsed."


def lookup_team_row(rows: list[tuple[int, str, float]], team: str) -> tuple[int | None, float | None]:
    for rank, name, score in rows:
        if name == team.strip():
            return rank, score
    return None, None


def extract_leaderboard_score(body: dict[str, Any]) -> float | None:
    for k in ("leaderboard_score", "score", "unified_score", "metric", "value", "result"):
        if k in body and body[k] is not None:
            try:
                return float(body[k])
            except (TypeError, ValueError):
                pass
    for nest in ("leaderboard", "result", "data", "submission"):
        nested = body.get(nest)
        if isinstance(nested, dict):
            inner = extract_leaderboard_score(nested)
            if inner is not None:
                return inner
    return None


def extract_cooldown_seconds(resp: requests.Response | None, body: dict[str, Any], default: int) -> int:
    if resp is not None:
        ra = resp.headers.get("Retry-After")
        if ra:
            try:
                return max(0, int(float(ra.strip())))
            except (TypeError, ValueError):
                pass
    if isinstance(body, dict):
        for key in ("cooldown_seconds", "wait_seconds", "retry_after_seconds", "next_submit_in_seconds"):
            if key in body and body[key] is not None:
                try:
                    return max(0, int(float(body[key])))
                except (TypeError, ValueError):
                    pass
        detail = body.get("detail")
        if isinstance(detail, str):
            w = parse_wait_seconds_from_text(detail)
            if w is not None:
                return w
    return max(0, int(default))


def _http_status_meaning(code: int, body: dict[str, Any] | None = None) -> str:
    if code == 200 and isinstance(body, dict):
        ev = parse_server_eval(body)
        if ev["eval_status"] == "passed":
            return "HTTP 200 — upload OK, evaluation passed"
        if ev["eval_status"] == "failed":
            return "HTTP 200 — upload OK, evaluation REJECTED"
        return "HTTP 200 — upload OK (check eval status in response)"
    return {
        200: "HTTP 200 — upload OK",
        400: "Bad request",
        401: "Unauthorized — check API key",
        413: "File too large",
        422: "Validation error",
        429: "Rate limit — wait for cooldown",
        500: "Server error",
    }.get(code, "See server response")


def format_server_detail(body: dict[str, Any]) -> tuple[list[str], str]:
    if not isinstance(body, dict) or "detail" not in body:
        return [], ""
    d = body["detail"]
    lines = ["  —— server detail ——"]
    summary = ""
    if isinstance(d, str):
        lines.append(f"  {d}")
        summary = d[:160]
    elif isinstance(d, list):
        for item in d:
            if isinstance(item, dict):
                msg = item.get("msg") or item.get("message") or ""
                lines.append(f"  • {msg}")
                summary = summary or str(msg)[:160]
    else:
        lines.append(f"  {d!r}")
        summary = str(d)[:160]
    lines.append("  —— end detail ——")
    return lines, summary


def append_top3_snapshot(rows: list[tuple[int, str, float]], source: str) -> None:
    top = [{"rank": r, "team": n, "score": sc} for r, n, sc in rows[:3]]
    if not top:
        return
    entry = {"ts_utc": utc_now(), "source": source, "top3": top}
    with _top3_history_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_top3_snapshots(max_lines: int = 600) -> list[dict[str, Any]]:
    p = _top3_history_path()
    if not p.is_file():
        return []
    out: list[dict[str, Any]] = []
    for ln in p.read_text(encoding="utf-8").strip().splitlines()[-max_lines:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return list(reversed(out))


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("TML Assignment 3 — Adversarial Robustness")
        self.geometry("1220x940")
        self.minsize(1000, 760)
        self._settings = load_settings()
        self._cooldown_200_until = float(cooldown_until_from_history(200))
        self._rate_limit_until = float(cooldown_until_from_history(429))
        self._submit_in_flight = False
        self._queue_launch_pending = False
        self._queue_pause_reason = ""
        self._queue_inflight_id: str | None = None
        self._queue_items = load_submit_queue()
        self._selected_abs = ""
        self._file_tree_paths: dict[str, str] = {}
        self._file_tree_meta: dict[str, dict[str, Any]] = {}
        self._hist_row_meta: dict[str, dict[str, Any]] = {}
        self._extra_paths: set[str] = set()
        self._local_eval_in_flight = False
        self._progress_metrics = discover_progress_metrics()
        self._local_metrics_cache = load_local_metrics_cache()
        self.auto_queue_var = tk.BooleanVar(value=bool(self._settings.get("auto_submit_queue", False)))
        self.model_name_var = tk.StringVar(value=str(self._settings.get("default_model_name", DEFAULT_MODEL_NAME)))

        hdr = ttk.Frame(self, padding=(12, 10, 12, 4))
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text="Trustworthy ML — Assignment 3: Adversarial Robustness", font=("Segoe UI", 12, "bold")).pack(
            anchor=tk.W
        )
        task_fr = ttk.LabelFrame(hdr, text="Task rules (read before submit)", padding=6)
        task_fr.pack(fill=tk.X, pady=(6, 0))
        for line in TASK_INFO_LINES:
            ttk.Label(task_fr, text=f"• {line}", wraplength=1120, justify=tk.LEFT).pack(anchor=tk.W)

        pick_fr = ttk.LabelFrame(self, text="Checkpoints (.pt)", padding=8)
        pick_fr.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        cols = ("file", "clean", "robust", "server_lb", "status", "scored_n", "size_mb", "modified")
        self.file_tree = ttk.Treeview(pick_fr, columns=cols, show="headings", height=9)
        for c, t, w in [
            ("file", "File", 210),
            ("clean", "Local clean", 82),
            ("robust", "Local robust", 86),
            ("server_lb", "Server LB", 78),
            ("status", "Status", 88),
            ("scored_n", "Scored #", 68),
            ("size_mb", "MB", 52),
            ("modified", "Modified", 128),
        ]:
            self.file_tree.heading(c, text=t)
            self.file_tree.column(c, width=w)
        self.file_tree.tag_configure("f_submitted", background="#e8f5e9")
        self.file_tree.tag_configure("f_rejected", background="#ffebee")
        self.file_tree.tag_configure("f_tried", background="#fff8e1")
        self.file_tree.tag_configure("f_low_clean", background="#fff3e0")
        sb = ttk.Scrollbar(pick_fr, orient="vertical", command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=sb.set)
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_tree.bind("<<TreeviewSelect>>", self._on_file_pick)

        btn_fr = ttk.Frame(pick_fr)
        btn_fr.pack(fill=tk.X, pady=6)
        ttk.Button(btn_fr, text="Browse .pt…", command=self._browse).pack(side=tk.LEFT)
        ttk.Button(btn_fr, text="Refresh", command=self._refresh_list).pack(side=tk.LEFT, padx=6)
        ttk.Label(btn_fr, text="Architecture:").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Combobox(
            btn_fr,
            textvariable=self.model_name_var,
            values=list(ALLOWED_ARCHITECTURES),
            width=12,
            state="readonly",
        ).pack(side=tk.LEFT)

        path_fr = ttk.Frame(self, padding=(12, 0, 12, 4))
        path_fr.pack(fill=tk.X)
        self.full_path_var = tk.StringVar()
        self.selected_history_var = tk.StringVar(value="History: —")
        self.selected_metrics_var = tk.StringVar(value="Metrics: —")
        ttk.Label(path_fr, text="Selected path").pack(anchor=tk.W)
        ttk.Entry(path_fr, textvariable=self.full_path_var, state="readonly").pack(fill=tk.X)
        ttk.Label(path_fr, textvariable=self.selected_metrics_var, font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(path_fr, textvariable=self.selected_history_var, wraplength=1120, justify=tk.LEFT).pack(anchor=tk.W, pady=2)

        lb_fr = ttk.LabelFrame(self, text="Live leaderboard", padding=8)
        lb_fr.pack(fill=tk.X, padx=12, pady=4)
        self.lb_top3_var = tk.StringVar(value="Top 3: —")
        self.lb_mine_var = tk.StringVar(value="Your team: —")
        self.lb_status_var = tk.StringVar()
        ttk.Label(lb_fr, textvariable=self.lb_top3_var).pack(anchor=tk.W)
        ttk.Label(lb_fr, textvariable=self.lb_mine_var).pack(anchor=tk.W, pady=2)
        ttk.Label(lb_fr, textvariable=self.lb_status_var, foreground="#555").pack(anchor=tk.W)
        ttk.Button(lb_fr, text="Update leaderboard", command=self._update_leaderboard).pack(anchor=tk.W, pady=4)
        self.lb_last_var = tk.StringVar(value="Last submit: before — | after —")
        ttk.Label(lb_fr, textvariable=self.lb_last_var, font=("Consolas", 9)).pack(anchor=tk.W)

        ctl = ttk.Frame(self, padding=(12, 6))
        ctl.pack(fill=tk.X)
        self.cooldown_var = tk.StringVar(value="Cooldown: —")
        ttk.Label(ctl, textvariable=self.cooldown_var, font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.queue_status_var = tk.StringVar(value="Queue: empty")
        ttk.Label(ctl, textvariable=self.queue_status_var, foreground="#1565c0").pack(anchor=tk.W)

        act = ttk.Frame(self, padding=(12, 4, 12, 8))
        act.pack(fill=tk.X)
        ttk.Button(act, text="Validate checkpoint", command=self._validate).pack(side=tk.LEFT)
        ttk.Button(act, text="Local eval (quick)", command=self._local_eval).pack(side=tk.LEFT, padx=6)
        self.submit_btn = ttk.Button(act, text="Submit to leaderboard", command=self._submit)
        self.submit_btn.pack(side=tk.LEFT, padx=8)
        ttk.Button(act, text="Reset cooldown (local)", command=self._reset_local_cooldown).pack(side=tk.LEFT, padx=4)
        ttk.Button(act, text="Add to queue", command=self._queue_add_selected).pack(side=tk.LEFT)
        ttk.Checkbutton(act, text="Auto-run queue", variable=self.auto_queue_var, command=self._save_auto_queue).pack(
            side=tk.LEFT, padx=8
        )
        ttk.Button(act, text="Settings…", command=self._settings_dialog).pack(side=tk.LEFT, padx=8)

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        log_fr = ttk.Frame(nb, padding=4)
        nb.add(log_fr, text="Log")
        self.log = tk.Text(log_fr, height=12, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 10))
        self.log.pack(fill=tk.BOTH, expand=True)

        hist_fr = ttk.Frame(nb, padding=4)
        nb.add(hist_fr, text="History")
        hcols = ("time", "file", "arch", "http", "eval", "clean", "robust", "unified", "lb_after", "message")
        self.hist_tree = ttk.Treeview(hist_fr, columns=hcols, show="headings", height=14)
        for c, t, w in [
            ("time", "Time UTC", 128),
            ("file", "Checkpoint", 150),
            ("arch", "Arch", 64),
            ("http", "HTTP", 44),
            ("eval", "Eval", 64),
            ("clean", "Clean %", 72),
            ("robust", "Robust %", 72),
            ("unified", "Unified", 72),
            ("lb_after", "LB score", 72),
            ("message", "Server message", 300),
        ]:
            self.hist_tree.heading(c, text=t)
            self.hist_tree.column(c, width=w)
        hsb = ttk.Scrollbar(hist_fr, orient="vertical", command=self.hist_tree.yview)
        self.hist_tree.configure(yscrollcommand=hsb.set)
        self.hist_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        hsb.pack(side=tk.RIGHT, fill=tk.Y)

        queue_fr = ttk.Frame(nb, padding=4)
        nb.add(queue_fr, text="Queue")
        qcols = ("pos", "file", "arch", "added", "attempts", "last")
        self.queue_tree = ttk.Treeview(queue_fr, columns=qcols, show="headings", height=8)
        for c, t, w in [
            ("pos", "#", 36),
            ("file", "Checkpoint", 240),
            ("arch", "Arch", 72),
            ("added", "Added UTC", 140),
            ("attempts", "Try", 44),
            ("last", "Last event", 320),
        ]:
            self.queue_tree.heading(c, text=t)
            self.queue_tree.column(c, width=w)
        qsb = ttk.Scrollbar(queue_fr, orient="vertical", command=self.queue_tree.yview)
        self.queue_tree.configure(yscrollcommand=qsb.set)
        self.queue_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        qsb.pack(side=tk.RIGHT, fill=tk.Y)
        qbtn = ttk.Frame(queue_fr)
        qbtn.pack(fill=tk.X, pady=4)
        ttk.Button(qbtn, text="Remove selected", command=self._queue_remove_selected).pack(side=tk.LEFT)
        ttk.Button(qbtn, text="Clear queue", command=self._queue_clear).pack(side=tk.LEFT, padx=6)

        top3_fr = ttk.Frame(nb, padding=4)
        nb.add(top3_fr, text="Top 3 track")
        t3cols = ("time", "src", "r1", "s1", "r2", "s2", "r3", "s3")
        self.top3_tree = ttk.Treeview(top3_fr, columns=t3cols, show="headings", height=12)
        for c, t, w in [
            ("time", "Time", 130),
            ("src", "Source", 90),
            ("r1", "#1 team", 88),
            ("s1", "#1 score", 72),
            ("r2", "#2 team", 88),
            ("s2", "#2 score", 72),
            ("r3", "#3 team", 88),
            ("s3", "#3 score", 72),
        ]:
            self.top3_tree.heading(c, text=t)
            self.top3_tree.column(c, width=w)
        self.top3_tree.pack(fill=tk.BOTH, expand=True)

        self._log(f"Repo: {REPO_ROOT}")
        self._log(f"Config: {_config_dir()}")
        self._log(f"History: {_history_path()}")
        self._refresh_list()
        self._reload_history_table()
        self._reload_queue_table()
        self._reload_top3_table()
        self.after(500, self._tick_timer)

    def _log(self, msg: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _on_file_pick(self, _evt: object | None = None) -> None:
        sel = self.file_tree.selection()
        if not sel:
            return
        self._selected_abs = self._file_tree_paths.get(sel[0], "")
        self.full_path_var.set(self._selected_abs)
        meta = self._file_tree_meta.get(sel[0], {})
        state = meta.get("history", {})
        metrics = meta.get("metrics", {})
        self.selected_metrics_var.set(
            metrics_summary_text(metrics if isinstance(metrics, dict) else {}, state if isinstance(state, dict) else None)
        )
        self.selected_history_var.set(submit_state_detail_text(state) if isinstance(state, dict) else "History: —")

    def _browse(self) -> None:
        p = filedialog.askopenfilename(
            title="Choose checkpoint .pt",
            filetypes=[("PyTorch", "*.pt"), ("All", "*.*")],
            initialdir=str(REPO_ROOT / DEFAULT_SCAN_SUBDIR),
        )
        if p:
            self._extra_paths.add(str(Path(p).resolve()))
            self._refresh_list()

    def _refresh_list(self) -> None:
        scan = str(self._settings.get("scan_subdir") or DEFAULT_SCAN_SUBDIR)
        paths = discover_checkpoints(scan)
        for bp in self._extra_paths:
            if Path(bp).is_file() and bp not in paths:
                paths.append(bp)
        self._progress_metrics = discover_progress_metrics()
        self._local_metrics_cache = load_local_metrics_cache()
        path_index, name_index = build_submit_history_index()
        selected_before = _normalize_path_str(self._selected_abs) if self._selected_abs else ""

        for i in self.file_tree.get_children():
            self.file_tree.delete(i)
        self._file_tree_paths.clear()
        self._file_tree_meta.clear()

        rows: list[tuple[str, tuple[Any, ...], str, dict[str, Any], dict[str, Any], str]] = []
        for full in paths:
            p = Path(full).resolve()
            state = summarize_submit_state(str(p), path_index, name_index)
            metrics = lookup_checkpoint_metrics(str(p), self._progress_metrics, self._local_metrics_cache)
            if state.get("last_clean_acc") is not None and not metrics.get("clean_acc"):
                metrics["clean_acc"] = state.get("last_clean_acc")
            if state.get("last_robust_acc") is not None and not metrics.get("robust_acc"):
                metrics["robust_acc"] = state.get("last_robust_acc")
            if state.get("last_unified_score") is not None and not metrics.get("unified_score"):
                metrics["unified_score"] = state.get("last_unified_score")
            if int(state.get("scored", 0)) > 0 or str(state.get("last_eval_status", "")).lower() == "passed":
                tag = "f_submitted"
            elif int(state.get("rejected", 0)) > 0 or str(state.get("last_eval_status", "")).lower() == "failed":
                tag = "f_rejected"
            elif int(state.get("attempts", 0)) > 0:
                tag = "f_tried"
            else:
                tag = "f_new"
            clean_v = metrics.get("clean_acc")
            if isinstance(clean_v, (int, float)) and float(clean_v) <= MIN_CLEAN_ACC and tag == "f_new":
                tag = "f_low_clean"
            server_lb = state.get("last_lb_score", state.get("last_unified_score"))
            server_lb_s = _fmt_lb(server_lb) if server_lb is not None else "—"
            try:
                st = p.stat()
                size_mb = f"{st.st_size / 1e6:.1f}"
                mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
            except OSError:
                size_mb, mtime = "—", "—"
            rows.append(
                (
                    p.name,
                    (
                        p.name,
                        fmt_pct(metrics.get("clean_acc")),
                        fmt_pct(metrics.get("robust_acc")),
                        server_lb_s,
                        submit_state_label(state),
                        int(state.get("scored", 0)),
                        size_mb,
                        mtime,
                    ),
                    str(p),
                    state,
                    metrics,
                    tag,
                )
            )
        rows.sort(key=lambda x: x[0])
        selected_iid = ""
        for idx, (_n, vals, abspath, state, metrics, tag) in enumerate(rows):
            iid = f"r{idx}"
            self._file_tree_paths[iid] = abspath
            self._file_tree_meta[iid] = {"path": abspath, "history": state, "metrics": metrics}
            self.file_tree.insert("", tk.END, iid=iid, values=vals, tags=(tag,))
            if selected_before and _normalize_path_str(abspath) == selected_before:
                selected_iid = iid
        if rows:
            iid = selected_iid or self.file_tree.get_children()[0]
            self.file_tree.selection_set(iid)
            self._on_file_pick()
        self._log(f"Listed {len(rows)} checkpoint(s) from {scan}")

    def _selected_path(self) -> Path | None:
        if not self._selected_abs:
            return None
        p = Path(self._selected_abs)
        return p if p.is_file() else None

    def _current_model_name(self) -> str:
        name = self.model_name_var.get().strip()
        return name if name in ALLOWED_ARCHITECTURES else DEFAULT_MODEL_NAME

    def _validate(self) -> None:
        p = self._selected_path()
        arch = self._current_model_name()
        if p is None:
            messagebox.showerror("Validate", "Select a checkpoint first.")
            return
        try:
            validate_checkpoint_file(p, arch)
        except Exception as e:
            messagebox.showerror("Validate", str(e))
            self._log(f"VALIDATE FAIL: {e}")
            return
        messagebox.showinfo("Validate", f"OK — {p.name} loads as {arch}, output shape (1, 9).")
        self._log(f"VALIDATE OK: {p.name} ({arch})")
        self._settings["default_model_name"] = arch
        save_settings(self._settings)

    def _update_leaderboard(self) -> None:
        team = str(self._settings.get("team_name") or DEFAULT_TEAM_NAME).strip()

        def worker() -> None:
            html, err = fetch_leaderboard_html()
            rows: list[tuple[int, str, float]] = []
            perr: str | None = None
            if html:
                rows, perr = parse_leaderboard_rank_table(html)
            self.after(0, lambda: self._leaderboard_done(err, perr, rows, team))

        threading.Thread(target=worker, daemon=True).start()

    def _leaderboard_done(
        self,
        net_err: str | None,
        parse_err: str | None,
        rows: list[tuple[int, str, float]],
        team: str,
    ) -> None:
        if net_err:
            messagebox.showerror("Leaderboard", net_err)
            return
        if parse_err:
            messagebox.showwarning("Leaderboard", parse_err)
            return
        if rows[:3]:
            self.lb_top3_var.set(
                "Top 3: " + " | ".join(f"#{r} {n} {sc:.6f}" for r, n, sc in rows[:3])
            )
        rank, sc = lookup_team_row(rows, team)
        if rank is not None and sc is not None:
            self.lb_mine_var.set(f"Your team ({team}): rank {rank}/{len(rows)} — score {sc:.6f}")
        else:
            self.lb_mine_var.set(f"Your team ({team}): not found — check Settings.")
        self.lb_status_var.set(f"Updated {utc_now()} · {len(rows)} teams")
        append_top3_snapshot(rows, "manual_update")
        self._reload_top3_table()
        self._log(f"LEADERBOARD OK: {len(rows)} teams")

    def _start_submit(
        self,
        p: Path,
        model_name: str,
        *,
        trigger: str = "manual",
        queue_id: str | None = None,
        queue_attempt: int = 0,
    ) -> bool:
        key = str(self._settings.get("api_key", "")).strip()
        if not key:
            messagebox.showerror("Submit", "Set API key in Settings.")
            return False
        block = max(self._cooldown_200_until, self._rate_limit_until)
        if time.time() < block:
            if trigger == "manual":
                messagebox.showinfo("Submit", "Wait for cooldown.")
            return False
        if self._submit_in_flight:
            return False
        try:
            validate_checkpoint_file(p, model_name)
        except Exception as e:
            if queue_id:
                self._pause_queue(queue_id, p.name, str(e))
            messagebox.showerror("Submit", str(e))
            return False

        self._submit_in_flight = True
        self._queue_inflight_id = queue_id
        self.submit_btn.configure(state=tk.DISABLED)
        team = str(self._settings.get("team_name") or DEFAULT_TEAM_NAME).strip()
        cd_default = int(self._settings.get("cooldown_seconds") or DEFAULT_COOLDOWN_S)

        def worker() -> None:
            err: str | None = None
            status = 0
            body: dict[str, Any] = {}
            resp: requests.Response | None = None
            lb_before_str = "unknown"
            lb_after_str = "unknown"
            html_b, _ = fetch_leaderboard_html()
            rows_before: list[tuple[int, str, float]] = []
            if html_b:
                rows_before, _ = parse_leaderboard_rank_table(html_b)
            if rows_before:
                _, sc_b = lookup_team_row(rows_before, team)
                if sc_b is not None:
                    lb_before_str = _fmt_lb(sc_b)
            try:
                with open(p, "rb") as f:
                    resp = requests.post(
                        f"{BASE_URL}/submit/{TASK_ID}",
                        headers={"X-API-Key": key},
                        files={"file": (p.name, f, "application/x-pytorch")},
                        data={"model_name": model_name},
                        timeout=120,
                    )
                status = resp.status_code
                try:
                    body = resp.json()
                except ValueError:
                    body = {"_raw": resp.text[:2000]}
            except Exception as e:
                err = str(e)

            rows_after: list[tuple[int, str, float]] = []
            if not err and status == 200:
                time.sleep(LEADERBOARD_REFETCH_DELAY_S)
            html2, _ = fetch_leaderboard_html()
            if html2:
                rows_after, _ = parse_leaderboard_rank_table(html2)
            if rows_after:
                append_top3_snapshot(rows_after, "after_submit")
                _, sc_a = lookup_team_row(rows_after, team)
                if sc_a is not None:
                    lb_after_str = _fmt_lb(sc_a)
            if lb_after_str == "unknown":
                parsed = extract_leaderboard_score(body)
                if parsed is not None:
                    lb_after_str = _fmt_lb(parsed)

            no_change = False
            try:
                b = float(lb_before_str)
                a = float(lb_after_str)
                no_change = abs(b - a) <= LB_SCORE_EPS
            except ValueError:
                pass

            def done() -> None:
                self._submit_in_flight = False
                self._queue_inflight_id = None
                self.submit_btn.configure(state=tk.NORMAL)
                now = time.time()
                wait_logged = 0
                cooldown_reason = ""
                server_eval = parse_server_eval(body)
                if not err:
                    if status == 429:
                        wait_logged = extract_cooldown_seconds(resp, body, cd_default)
                        self._rate_limit_until = now + wait_logged
                        cooldown_reason = "rate_limit"
                    elif status == 200:
                        self._rate_limit_until = 0.0
                        wait_logged, cooldown_reason = eval_cooldown_for_response(status, body, cd_default)
                        self._cooldown_200_until = now + wait_logged

                detail_lines, detail_summary = format_server_detail(body)
                eval_label = server_eval.get("eval_status", "unknown")
                clean_s = fmt_pct(server_eval.get("clean_acc"))
                robust_s = fmt_pct(server_eval.get("robust_acc"))
                unified_s = fmt_pct(server_eval.get("unified_score"))
                msg = str(server_eval.get("message", "") or detail_summary or "")
                summary = f"{'QUEUE' if trigger == 'queue_auto' else 'MANUAL'} HTTP {status} | eval {eval_label}"
                summary += f" | clean {clean_s} | robust {robust_s} | unified {unified_s}"
                if msg:
                    summary += f" | {msg[:100]}"
                summary += f" | LB {lb_before_str}->{lb_after_str}"
                if wait_logged > 0:
                    summary += f" | cooldown {wait_logged}s ({cooldown_reason})"

                entry = {
                    "ts_utc": utc_now(),
                    "model_path": str(p),
                    "model_basename": p.name,
                    "model_name": model_name,
                    "http_status": status,
                    "error": err,
                    "response": body,
                    "server_eval": server_eval,
                    "eval_status": eval_label,
                    "clean_accuracy": server_eval.get("clean_acc"),
                    "robustness_accuracy": server_eval.get("robust_acc"),
                    "unified_score": server_eval.get("unified_score"),
                    "server_message": msg,
                    "leaderboard_score_before": lb_before_str,
                    "leaderboard_score_after": lb_after_str,
                    "leaderboard_score_unchanged": no_change,
                    "cooldown_applied_seconds": wait_logged,
                    "cooldown_reason": cooldown_reason,
                    "log_summary": summary,
                    "submission_trigger": trigger,
                }
                if queue_id:
                    entry["queue_item_id"] = queue_id
                    entry["queue_attempt"] = queue_attempt
                append_history(entry)
                self.lb_last_var.set(
                    f"Last submit: eval {eval_label} | clean {clean_s} | robust {robust_s} | LB {lb_before_str}->{lb_after_str}"
                )
                for ln in detail_lines:
                    self._log(ln)
                self._log(summary if not err else f"ERROR: {err}")
                if queue_id:
                    if err:
                        self._pause_queue(queue_id, p.name, err)
                    elif status == 200 and server_eval.get("eval_status") == "passed":
                        self._queue_items = [q for q in self._queue_items if q.get("queue_id") != queue_id]
                        save_submit_queue(self._queue_items)
                        append_queue_event({"action": "submit_ok", "model_basename": p.name, "queue_id": queue_id})
                    elif status == 200 and server_eval.get("eval_status") == "failed":
                        self._pause_queue(queue_id, p.name, msg or "evaluation rejected")
                    elif status == 429:
                        append_queue_event({"action": "submit_wait", "model_basename": p.name, "details": f"{wait_logged}s"})
                    elif status == 200:
                        self._queue_items = [q for q in self._queue_items if q.get("queue_id") != queue_id]
                        save_submit_queue(self._queue_items)
                    else:
                        self._pause_queue(queue_id, p.name, f"HTTP {status}")
                self._refresh_list()
                self._reload_history_table()
                self._reload_queue_table()
                self._reload_top3_table()
                if err:
                    messagebox.showerror("Submit", err)
                elif status == 200:
                    title = "Submit — scored" if server_eval.get("eval_status") == "passed" else "Submit — rejected"
                    icon = messagebox.showinfo if server_eval.get("eval_status") == "passed" else messagebox.showwarning
                    icon(
                        title,
                        f"{_http_status_meaning(status, body)}\n\n"
                        f"Clean: {clean_s}\nRobust: {robust_s}\nUnified: {unified_s}\n\n"
                        f"{msg or 'No extra message.'}\n\n"
                        f"Cooldown: {wait_logged}s ({cooldown_reason or 'none'})",
                    )
                else:
                    messagebox.showwarning(
                        "Submit",
                        f"HTTP {status}\n{_http_status_meaning(status, body)}\n{detail_summary}",
                    )

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()
        return True

    def _confirm_submit_allowed(self, p: Path) -> bool:
        name_l = p.name.lower()
        if "template" in name_l or "sanity" in name_l:
            if not messagebox.askyesno(
                "Template checkpoint?",
                f"{p.name} looks like the course template (random weights).\n"
                "It will be rejected (clean accuracy << 50%).\n\nSubmit anyway?",
            ):
                return False
        metrics = lookup_checkpoint_metrics(str(p), self._progress_metrics, self._local_metrics_cache)
        clean_v = metrics.get("clean_acc")
        if isinstance(clean_v, (int, float)) and float(clean_v) <= MIN_CLEAN_ACC:
            if not messagebox.askyesno(
                "Clean accuracy gate",
                f"Known clean accuracy is {fmt_pct(clean_v)} (must be > 50%).\n"
                "Server will reject this submission.\n\nSubmit anyway?",
            ):
                return False
        return True

    def _submit(self) -> None:
        p = self._selected_path()
        if p is None:
            messagebox.showerror("Submit", "Select a checkpoint.")
            return
        if not self._confirm_submit_allowed(p):
            return
        self._start_submit(p, self._current_model_name(), trigger="manual")

    def _reset_local_cooldown(self) -> None:
        if not messagebox.askyesno(
            "Reset local cooldown",
            "Clears only the GUI timer (_cooldown_200_until / rate limit).\n\n"
            "The server may still enforce its own 60-minute window after a scored submit.\n"
            "Use this if the timer is stuck or you had a rejected evaluation (2 min rule).\n\n"
            "Reset now?",
        ):
            return
        self._cooldown_200_until = 0.0
        self._rate_limit_until = 0.0
        append_history(
            {
                "ts_utc": utc_now(),
                "model_path": "",
                "model_basename": "(local cooldown reset)",
                "model_name": "",
                "http_status": 0,
                "error": None,
                "response": {"status": "local_reset", "message": "User reset local cooldown timer"},
                "log_summary": "LOCAL: cooldown timer reset by user",
                "submission_trigger": "local_reset",
                "cooldown_applied_seconds": 0,
            }
        )
        self.cooldown_var.set("Submit cooldown: ready (reset locally)")
        self._log("LOCAL COOLDOWN RESET — GUI timer cleared")
        self._reload_history_table()

    def _local_eval(self) -> None:
        p = self._selected_path()
        arch = self._current_model_name()
        if p is None:
            messagebox.showerror("Local eval", "Select a checkpoint first.")
            return
        if self._local_eval_in_flight:
            return
        try:
            validate_checkpoint_file(p, arch)
        except Exception as e:
            messagebox.showerror("Local eval", str(e))
            return
        self._local_eval_in_flight = True
        self.selected_metrics_var.set("Metrics: running local eval (quick robust on val split)…")
        self._log(f"LOCAL EVAL start: {p.name} ({arch})")

        def worker() -> None:
            err: str | None = None
            metrics: dict[str, Any] = {}
            try:
                metrics = run_local_eval_checkpoint(p, arch, quick=True)
            except Exception as e:
                err = str(e)

            def done() -> None:
                self._local_eval_in_flight = False
                if err:
                    messagebox.showerror("Local eval", err)
                    self._log(f"LOCAL EVAL FAIL: {err}")
                    return
                key = _normalize_path_str(p)
                self._local_metrics_cache[key] = metrics
                save_local_metrics_cache(self._local_metrics_cache)
                self._refresh_list()
                self._log(
                    f"LOCAL EVAL OK: clean {fmt_pct(metrics.get('clean_acc'))} | "
                    f"robust {fmt_pct(metrics.get('robust_acc'))} (quick) | "
                    f"unified {fmt_pct(metrics.get('unified_score'))}"
                )
                messagebox.showinfo("Local eval", metrics_summary_text(metrics, None))

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _save_auto_queue(self) -> None:
        self._settings["auto_submit_queue"] = bool(self.auto_queue_var.get())
        save_settings(self._settings)

    def _queue_add_selected(self) -> None:
        p = self._selected_path()
        arch = self._current_model_name()
        if p is None:
            messagebox.showerror("Queue", "Select a checkpoint.")
            return
        try:
            validate_checkpoint_file(p, arch)
        except Exception as e:
            messagebox.showerror("Queue", str(e))
            return
        resolved = str(p.resolve())
        if any(str(q.get("model_path")) == resolved for q in self._queue_items):
            messagebox.showinfo("Queue", "Already queued.")
            return
        item = {
            "queue_id": str(uuid.uuid4()),
            "model_path": resolved,
            "model_basename": p.name,
            "model_name": arch,
            "added_ts_utc": utc_now(),
            "attempts": 0,
            "last_event": "Queued",
            "last_event_ts_utc": utc_now(),
        }
        self._queue_items.append(item)
        save_submit_queue(self._queue_items)
        append_queue_event({"action": "queued", "model_basename": p.name, "model_name": arch})
        self._reload_queue_table()
        self._log(f"QUEUE ADD: {p.name}")

    def _queue_remove_selected(self) -> None:
        sel = self.queue_tree.selection()
        if not sel:
            return
        qid = sel[0]
        self._queue_items = [q for q in self._queue_items if str(q.get("queue_id")) != qid]
        save_submit_queue(self._queue_items)
        self._reload_queue_table()

    def _queue_clear(self) -> None:
        if self._queue_items and messagebox.askyesno("Queue", "Clear queue?"):
            self._queue_items = []
            save_submit_queue(self._queue_items)
            self._reload_queue_table()

    def _pause_queue(self, queue_id: str, name: str, reason: str) -> None:
        self._queue_pause_reason = reason
        self.auto_queue_var.set(False)
        self._settings["auto_submit_queue"] = False
        save_settings(self._settings)
        for q in self._queue_items:
            if str(q.get("queue_id")) == queue_id:
                q["last_event"] = f"Paused: {reason}"
        save_submit_queue(self._queue_items)

    def _reload_queue_table(self) -> None:
        for i in self.queue_tree.get_children():
            self.queue_tree.delete(i)
        for pos, item in enumerate(self._queue_items, 1):
            qid = str(item.get("queue_id"))
            self.queue_tree.insert(
                "",
                tk.END,
                iid=qid,
                values=(
                    pos,
                    item.get("model_basename"),
                    item.get("model_name"),
                    item.get("added_ts_utc"),
                    item.get("attempts", 0),
                    item.get("last_event"),
                ),
            )

    def _reload_history_table(self) -> None:
        for i in self.hist_tree.get_children():
            self.hist_tree.delete(i)
        self._hist_row_meta.clear()
        for row in load_history_rows(200):
            eu = row.get("entry_uuid")
            iid = str(eu) if eu else hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20]
            self._hist_row_meta[iid] = row
            ev = parse_server_eval(row.get("server_eval") or row.get("response"))
            clean = row.get("clean_accuracy", ev.get("clean_acc"))
            robust = row.get("robustness_accuracy", ev.get("robust_acc"))
            unified = row.get("unified_score", ev.get("unified_score"))
            msg = str(row.get("server_message", ev.get("message", "")) or "")
            self.hist_tree.insert(
                "",
                tk.END,
                iid=iid,
                values=(
                    row.get("ts_utc", ""),
                    row.get("model_basename", ""),
                    row.get("model_name", ""),
                    row.get("http_status", ""),
                    row.get("eval_status", ev.get("eval_status", "")),
                    fmt_pct(clean),
                    fmt_pct(robust),
                    fmt_pct(unified),
                    row.get("leaderboard_score_after", ""),
                    msg[:320],
                ),
            )

    def _reload_top3_table(self) -> None:
        for i in self.top3_tree.get_children():
            self.top3_tree.delete(i)
        for snap in load_top3_snapshots(200):
            t3 = snap.get("top3") if isinstance(snap.get("top3"), list) else []

            def slot(i: int) -> tuple[str, str]:
                if len(t3) <= i:
                    return "—", "—"
                return str(t3[i].get("team", "—")), _fmt_lb(t3[i].get("score"))

            t1, s1 = slot(0)
            t2, s2 = slot(1)
            t3n, s3 = slot(2)
            self.top3_tree.insert(
                "",
                tk.END,
                values=(snap.get("ts_utc"), snap.get("source"), t1, s1, t2, s2, t3n, s3),
            )

    def _submit_queue_head_if_ready(self) -> None:
        self._queue_launch_pending = False
        if not self.auto_queue_var.get() or self._submit_in_flight or not self._queue_items:
            return
        if time.time() < max(self._cooldown_200_until, self._rate_limit_until):
            return
        head = self._queue_items[0]
        qid = str(head.get("queue_id"))
        path = Path(str(head.get("model_path")))
        arch = str(head.get("model_name") or DEFAULT_MODEL_NAME)
        head["attempts"] = int(head.get("attempts", 0)) + 1
        save_submit_queue(self._queue_items)
        self._start_submit(path, arch, trigger="queue_auto", queue_id=qid, queue_attempt=int(head["attempts"]))

    def _tick_timer(self) -> None:
        t = time.time()

        def fmt(sec: int) -> str:
            m, s = divmod(max(0, sec), 60)
            h, m = divmod(m, 60)
            return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

        block_until = max(self._cooldown_200_until, self._rate_limit_until)
        if block_until > t:
            which = "rate limit" if self._rate_limit_until >= self._cooldown_200_until else "post-submit"
            self.cooldown_var.set(f"Submit cooldown ({which}): {fmt(int(block_until - t))} remaining")
        else:
            self._cooldown_200_until = 0.0
            self._rate_limit_until = 0.0
            self.cooldown_var.set("Submit cooldown: ready")

        can = t >= max(self._cooldown_200_until, self._rate_limit_until) and not self._submit_in_flight
        self.submit_btn.configure(state=tk.NORMAL if can else tk.DISABLED)

        if self._queue_items:
            head = self._queue_items[0].get("model_basename", "")
            if self.auto_queue_var.get() and can:
                self.queue_status_var.set(f"Queue: {len(self._queue_items)} — auto-ready → {head}")
                if not self._queue_launch_pending:
                    self._queue_launch_pending = True
                    self.after(50, self._submit_queue_head_if_ready)
            else:
                self.queue_status_var.set(f"Queue: {len(self._queue_items)} pending — next {head}")
        else:
            self.queue_status_var.set("Queue: empty")

        self.after(500, self._tick_timer)

    def _settings_dialog(self) -> None:
        win = tk.Toplevel(self)
        win.title("Settings")
        win.geometry("560x260")
        fr = ttk.Frame(win, padding=10)
        fr.pack(fill=tk.BOTH, expand=True)
        ttk.Label(fr, text="TML_API_KEY").grid(row=0, column=0, sticky=tk.W)
        key_e = ttk.Entry(fr, width=50, show="*")
        key_e.insert(0, str(self._settings.get("api_key", "")))
        key_e.grid(row=0, column=1, sticky=tk.EW)
        ttk.Label(fr, text="Team name").grid(row=1, column=0, sticky=tk.W, pady=6)
        team_e = ttk.Entry(fr, width=30)
        team_e.insert(0, str(self._settings.get("team_name") or DEFAULT_TEAM_NAME))
        team_e.grid(row=1, column=1, sticky=tk.W, pady=6)
        ttk.Label(fr, text="Cooldown fallback (s)").grid(row=2, column=0, sticky=tk.W)
        cd_e = ttk.Entry(fr, width=10)
        cd_e.insert(0, str(int(self._settings.get("cooldown_seconds", DEFAULT_COOLDOWN_S))))
        cd_e.grid(row=2, column=1, sticky=tk.W)
        ttk.Label(fr, text="Scan folder (relative)").grid(row=3, column=0, sticky=tk.W, pady=6)
        sc_e = ttk.Entry(fr, width=50)
        sc_e.insert(0, str(self._settings.get("scan_subdir") or DEFAULT_SCAN_SUBDIR))
        sc_e.grid(row=3, column=1, sticky=tk.EW, pady=6)
        fr.columnconfigure(1, weight=1)

        def save() -> None:
            self._settings["api_key"] = key_e.get().strip()
            self._settings["team_name"] = team_e.get().strip() or DEFAULT_TEAM_NAME
            self._settings["scan_subdir"] = sc_e.get().strip() or DEFAULT_SCAN_SUBDIR
            try:
                self._settings["cooldown_seconds"] = max(0, int(cd_e.get().strip()))
            except ValueError:
                messagebox.showerror("Settings", "Cooldown must be integer.")
                return
            save_settings(self._settings)
            self._refresh_list()
            win.destroy()

        ttk.Button(fr, text="Save", command=save).grid(row=4, column=1, sticky=tk.E, pady=12)


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
