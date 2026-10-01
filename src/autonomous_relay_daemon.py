#!/usr/bin/env python3
"""Autonomous Kernel Relay & Submission Daemon for Biohub Cell Tracking.

Monitors running Kaggle GPU kernels (C003, C004), automatically submits
their outputs upon completion, tracks live leaderboard scoring,
updates submission logs and HANDOFF.md, and automatically pushes
staged candidate kernels (C005, C006) as GPU slots become available.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from kaggle.api.kaggle_api_extended import KaggleApi

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "biohub-cell-tracking-during-development"
SUBMISSION_LOG_PATH = REPO_ROOT / "experiments" / "submission_log.csv"
HANDOFF_PATH = REPO_ROOT / "HANDOFF.md"
DAEMON_STATE_PATH = REPO_ROOT / "state" / "relay_daemon_state.json"

# Candidate Kernel Definitions
KERNELS_MONITORED = {
    "C003": {
        "kernel": "taeyangg4/biohub-c003-medal-frontier",
        "version": 1,
        "message": "C003 medal frontier (diverge 1.0, sym 0.8, tight 5.5, expanded PP)",
        "target": ">= 0.948 (Bronze safe)",
        "dir": REPO_ROOT / "experiments" / "candidates" / "c003_medal_frontier",
        "verify_script": None,
    },
    "C004": {
        "kernel": "taeyangg4/biohub-c004-adaptive-lineage",
        "version": 1,
        "message": "C004 adaptive lineage enveloping (sister 16.0, exist 12.0, diverge 0.5, sym 0.85, margin 0.001)",
        "target": ">= 0.949 (Silver safe)",
        "dir": REPO_ROOT / "experiments" / "candidates" / "c004_adaptive_lineage",
        "verify_script": REPO_ROOT / "src" / "verify_c004.py",
    },
    "C005": {
        "kernel": "taeyangg4/biohub-c005-gold-fusion",
        "version": 3,
        "message": "C005 gold fusion pipeline (kinematic anti-swap + cytokinesis 0.5 + gap2 + short rescue)",
        "target": ">= 0.959 (Gold Medal safe)",
        "dir": REPO_ROOT / "experiments" / "candidates" / "c005_gold_fusion",
        "verify_script": REPO_ROOT / "src" / "verify_c005.py",
    },
    "C006": {
        "kernel": "taeyangg4/biohub-c006-consensus-fusion",
        "version": 2,
        "message": "C006 consensus fusion (localized gap anchor 6.0 um, dual-seed consensus, cytokinesis 0.5)",
        "target": ">= 0.965 (Gold High Frontier)",
        "dir": REPO_ROOT / "experiments" / "candidates" / "c006_consensus_fusion",
        "verify_script": REPO_ROOT / "src" / "verify_c006.py",
    },
    "C007": {
        "kernel": "taeyangg4/biohub-c007-champion-frontier",
        "version": 1,
        "message": "C007 champion frontier (kinematic anti-swap 0.80 + bidirectional consensus + cytokinesis 0.5 + gap2 11.5)",
        "target": ">= 0.974 (1st Place Champion Tier)",
        "dir": REPO_ROOT / "experiments" / "candidates" / "c007_champion_frontier",
        "verify_script": REPO_ROOT / "src" / "verify_c007.py",
    },
    "C008": {
        "kernel": "taeyangg4/biohub-c008-local-transformer-fusion",
        "version": 1,
        "message": "C008 local transformer fusion (anchored on proven C004 + 4070Ti local UNet transformer weights)",
        "target": ">= 0.955 (Silver/Gold Frontier)",
        "dir": REPO_ROOT / "experiments" / "candidates" / "c008_local_transformer_fusion",
        "verify_script": REPO_ROOT / "src" / "verify_c008.py",
    },
    "C009": {
        "kernel": "taeyangg4/biohub-c009-boost-geometric-fusion",
        "version": 1,
        "message": "C009 boost geometric fusion (C004 anchor + local 4070Ti weights + weak-leaf prune + prefix-guard)",
        "target": ">= 0.955 (Silver/Gold Frontier)",
        "dir": REPO_ROOT / "experiments" / "candidates" / "c009_boost_geometric_fusion",
        "verify_script": REPO_ROOT / "src" / "verify_c009.py",
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("autonomous_relay_daemon")


class AutonomousRelayDaemon:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.api = KaggleApi()
        self.api.authenticate()
        self.state: Dict[str, Any] = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        DAEMON_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if DAEMON_STATE_PATH.exists():
            try:
                st = json.loads(DAEMON_STATE_PATH.read_text(encoding="utf-8"))
                if not isinstance(st.get("submissions"), dict):
                    st["submissions"] = {}
                if not isinstance(st.get("pushed_kernels"), list):
                    st["pushed_kernels"] = ["C003", "C004"]
                if not isinstance(st.get("kernel_versions"), dict):
                    st["kernel_versions"] = {}
                return st
            except Exception as e:
                logger.warning(f"Could not load state from {DAEMON_STATE_PATH}: {e}")
        return {
            "submissions": {},
            "pushed_kernels": ["C003", "C004"],
            "kernel_versions": {},
            "last_check_ts": None,
            "kernel_statuses": {},
        }

    def _save_state(self) -> None:
        try:
            DAEMON_STATE_PATH.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to persist state: {e}")

    def query_kernel_status(self, kernel_slug: str, retries: int = 3) -> Optional[Dict[str, Any]]:
        """Queries Kaggle API for kernel execution status with rate-limit backoff."""
        for attempt in range(retries):
            try:
                status_obj = self.api.kernels_status(kernel_slug)
                status_val = getattr(status_obj, "status", None)
                failure_msg = getattr(status_obj, "failure_message", None) or getattr(status_obj, "failureMessage", None)
                return {"status": str(status_val), "failure_message": failure_msg}
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "Too Many Requests" in msg:
                    wait_time = (attempt + 1) * 30
                    logger.warning(f"Rate limited (429) checking {kernel_slug}. Backing off for {wait_time}s...")
                    time.sleep(wait_time)
                elif "403" in msg or "Cannot access kernel" in msg:
                    logger.debug(f"Kernel {kernel_slug} not pushed / access denied: {msg}")
                    return {"status": "NOT_PUSHED", "failure_message": None}
                else:
                    logger.warning(f"Error checking {kernel_slug} (attempt {attempt+1}/{retries}): {exc}")
                    if attempt < retries - 1:
                        time.sleep(5)
        return None

    def query_recent_submissions(self) -> List[Any]:
        try:
            return self.api.competition_submissions(COMPETITION)
        except Exception as exc:
            logger.error(f"Error querying competition submissions: {exc}")
            return []

    def submit_kernel(self, cid: str) -> Optional[str]:
        """Submits kernel output to Kaggle competition and logs ref safely."""
        cfg = KERNELS_MONITORED[cid]
        kernel_slug = cfg["kernel"]
        version = self.state.get("kernel_versions", {}).get(cid) or cfg.get("version")
        msg = cfg["message"]

        logger.info(f"Submitting {cid} ({kernel_slug} v{version}) to {COMPETITION}...")
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would submit {kernel_slug} v{version} with message '{msg}'")
            return "dry_run_ref"

        sub_ref: Optional[str] = None
        try:
            res = self.api.competition_submit_code(
                file_name="submission.csv",
                message=msg,
                competition=COMPETITION,
                kernel=kernel_slug,
                kernel_version=version,
                quiet=False,
            )
            logger.info(f"Kaggle submit response for {cid}: {res}")
            if res is not None and getattr(res, "ref", None):
                sub_ref = str(res.ref)
        except Exception as exc:
            logger.error(f"Kaggle submit exception for {cid} (version {version}): {exc}")
            try:
                logger.info(f"Retrying submit for {cid} with kernel_version=None (latest version)...")
                res = self.api.competition_submit_code(
                    file_name="submission.csv",
                    message=msg,
                    competition=COMPETITION,
                    kernel=kernel_slug,
                    quiet=False,
                )
                logger.info(f"Kaggle submit fallback response for {cid}: {res}")
                if res is not None and getattr(res, "ref", None):
                    sub_ref = str(res.ref)
            except Exception as exc2:
                logger.error(f"Kaggle submit fallback exception for {cid}: {exc2}")

        # If submit response did not supply ref directly, check recent submissions matching this candidate
        if not sub_ref:
            time.sleep(5)
            subs = self.query_recent_submissions()
            for s in subs[:5]:
                desc = getattr(s, "description", "") or ""
                ref = str(getattr(s, "ref", ""))
                # Strict match on message or candidate ID
                if (msg[:25] in desc or cid in desc) and ref:
                    sub_ref = ref
                    logger.info(f"Matched submission ref for {cid} from competition submissions: {sub_ref}")
                    break

        if not sub_ref:
            logger.error(f"Failed to obtain valid submission ref for {cid}. Submission will be retried.")
            return None

        self.state["submissions"][cid] = {
            "ref": sub_ref,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kernel": kernel_slug,
            "message": msg,
            "status": "pending_evaluation",
            "public_score": None,
        }
        self._save_state()
        self._update_submission_log(cid, sub_ref, "submitted", "Submitted automatically by autonomous relay daemon")
        self._update_handoff(cid, sub_ref)
        logger.info(f"Successfully recorded submission ref {sub_ref} for {cid}")
        return sub_ref

    def check_pending_submissions(self) -> None:
        """Polls Kaggle competition submissions to record newly scored results."""
        pending_cids = [
            cid for cid, sdata in self.state.get("submissions", {}).items()
            if sdata.get("status") != "scored" and sdata.get("public_score") is None
        ]
        if not pending_cids:
            return

        subs = self.query_recent_submissions()
        if not subs:
            return

        sub_by_ref: Dict[str, Any] = {str(getattr(s, "ref", "")): s for s in subs}

        for cid in pending_cids:
            sdata = self.state["submissions"][cid]
            ref = str(sdata.get("ref", ""))
            sub_obj = sub_by_ref.get(ref)
            if not sub_obj:
                continue

            sub_status = getattr(sub_obj, "status", None)
            status_str = str(sub_status).upper() if sub_status else ""
            public_score = getattr(sub_obj, "public_score", None) or getattr(sub_obj, "publicScore", None)

            if "COMPLETE" in status_str and public_score is not None and str(public_score).strip() not in ("", "None"):
                score_clean = str(public_score).strip()
                sdata["status"] = "scored"
                sdata["public_score"] = score_clean
                sdata["scored_at"] = datetime.now(timezone.utc).isoformat()
                self._save_state()

                self._update_submission_log(
                    cid, ref, "scored", f"Scored on Kaggle LB: {score_clean}"
                )
                self._update_handoff_score(cid, ref, score_clean)
                logger.info(f"🎉 SUCCESS! Candidate {cid} (submission {ref}) SCORED on Kaggle LB: {score_clean}!")
            elif "ERROR" in status_str:
                err_msg = getattr(sub_obj, "error_description", "") or "Failed evaluation on Kaggle"
                sdata["status"] = "error"
                sdata["error"] = str(err_msg)
                self._save_state()
                self._update_submission_log(cid, ref, "error", f"LB evaluation error: {err_msg}")
                self._update_handoff_score(cid, ref, "ERROR")
                logger.error(f"❌ Candidate {cid} (submission {ref}) evaluation ERROR: {err_msg}")

    def push_kernel(self, cid: str) -> bool:
        """Pushes candidate kernel to Kaggle after local verification."""
        cfg = KERNELS_MONITORED[cid]
        kernel_dir = cfg["dir"]
        verify_script = cfg["verify_script"]

        if not kernel_dir.exists():
            logger.error(f"Cannot push {cid}: directory {kernel_dir} does not exist!")
            return False

        # Run verification suite if available
        if verify_script and verify_script.exists():
            logger.info(f"Running verification for {cid} ({verify_script})...")
            try:
                subprocess.run([sys.executable, str(verify_script)], check=True, capture_output=True, text=True)
                logger.info(f"Verification for {cid} PASSED.")
            except subprocess.CalledProcessError as err:
                logger.error(f"Verification for {cid} FAILED: {err.stdout} {err.stderr}")
                return False

        logger.info(f"Pushing kernel {cid} ({cfg['kernel']}) to Kaggle from {kernel_dir}...")
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would push {cfg['kernel']}")
            return True

        pushed_version = None
        # Try programmatic push first via KaggleApi
        try:
            push_res = self.api.kernels_push(str(kernel_dir))
            if push_res is not None:
                err = getattr(push_res, "error", None)
                if err:
                    logger.warning(f"KaggleApi.kernels_push returned error for {cid}: {err}")
                else:
                    pushed_version = getattr(push_res, "version_number", None)
                    logger.info(f"KaggleApi.kernels_push succeeded for {cid}: version={pushed_version}")
        except Exception as api_err:
            logger.debug(f"Direct API push encountered: {api_err}. Falling back to CLI...")

        if pushed_version is None:
            res = subprocess.run(
                "kaggle kernels push -p .",
                shell=True,
                cwd=str(kernel_dir),
                capture_output=True,
                text=True,
            )
            combined_output = (res.stdout or "") + "\n" + (res.stderr or "")
            logger.info(f"CLI Push output for {cid}:\n{combined_output.strip()}")

            if res.returncode != 0 or "error" in combined_output.lower() or "maximum batch gpu session count" in combined_output.lower():
                logger.warning(f"Kernel push for {cid} was rejected (GPU sessions likely full): {combined_output.strip()}")
                return False

            m = re.search(r"Kernel version (\d+) successfully pushed", combined_output, re.IGNORECASE)
            if m:
                pushed_version = int(m.group(1))

        if pushed_version:
            cfg["version"] = pushed_version
            self.state.setdefault("kernel_versions", {})[cid] = pushed_version

        if cid not in self.state["pushed_kernels"]:
            self.state["pushed_kernels"].append(cid)
        self._save_state()
        self._update_submission_log(cid, "pending", "kernel_running", "Automatically pushed to Kaggle GPU slot by relay daemon")
        self._update_handoff_push(cid)
        return True

    def _update_submission_log(self, cid: str, sub_id: str, status: str, notes: str) -> None:
        """Updates experiments/submission_log.csv idempotently."""
        if not SUBMISSION_LOG_PATH.exists():
            return
        today = datetime.now().strftime("%Y-%m-%d")
        cfg = KERNELS_MONITORED[cid]
        exp_id = cid.lower() + "_" + cfg["kernel"].split("-", 2)[-1].replace("-", "_")

        rows: List[Dict[str, str]] = []
        fieldnames: List[str] = []
        with SUBMISSION_LOG_PATH.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = [fn for fn in (reader.fieldnames or []) if fn]
            for r in reader:
                clean_r = {k: v for k, v in r.items() if k in fieldnames}
                rows.append(clean_r)

        # Look for existing row for this candidate
        updated = False
        for r in rows:
            if cid.lower() in r.get("experiment_id", "").lower() or cfg["kernel"] in r.get("kaggle_ref", ""):
                if sub_id != "pending" or not r.get("submission_id") or r.get("submission_id") == "pending":
                    r["submission_id"] = sub_id
                r["status"] = status
                r["notes"] = notes
                r["datetime"] = today
                if status == "scored":
                    sub_info = self.state.get("submissions", {}).get(cid, {})
                    if sub_info.get("public_score"):
                        r["public_lb"] = str(sub_info["public_score"])
                updated = True
                break

        if not updated:
            new_row = {
                "submission_id": sub_id,
                "datetime": today,
                "experiment_id": exp_id,
                "kaggle_ref": f"{cfg['kernel']} v{cfg.get('version', 1)}",
                "message": cfg["message"],
                "public_lb": "",
                "private_lb": "",
                "status": status,
                "notes": notes,
            }
            rows.append(new_row)

        with SUBMISSION_LOG_PATH.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Updated {SUBMISSION_LOG_PATH} for {cid} (status={status})")

    def _update_handoff(self, cid: str, sub_ref: str) -> None:
        """Updates HANDOFF.md submission ledger with newly submitted candidate."""
        if not HANDOFF_PATH.exists():
            return
        text = HANDOFF_PATH.read_text(encoding="utf-8")
        cfg = KERNELS_MONITORED[cid]
        line = f"| {sub_ref} | {cfg['kernel']} v{cfg.get('version', 1)} | PENDING | submitted automatically by autonomous relay daemon |\n"
        if sub_ref not in text:
            target_marker = "| 56209758 | clean public-0.947 settings anchor | 0.946 | reproduced public settings, but missed 0.947 due to disabled validator |\n"
            if target_marker in text:
                text = text.replace(target_marker, target_marker + line)
                HANDOFF_PATH.write_text(text, encoding="utf-8")
                logger.info(f"Added submission {sub_ref} ({cid}) to HANDOFF.md ledger.")

    def _update_handoff_score(self, cid: str, sub_ref: str, score: str) -> None:
        """Updates HANDOFF.md when a submission finishes scoring on Kaggle."""
        if not HANDOFF_PATH.exists():
            return
        text = HANDOFF_PATH.read_text(encoding="utf-8")
        cfg = KERNELS_MONITORED[cid]
        old_pattern = rf"\| {sub_ref} \| ([^\|]+) \| PENDING \|"
        replacement = f"| {sub_ref} | \\1 | {score} |"
        if re.search(old_pattern, text):
            text = re.sub(old_pattern, replacement, text)
            HANDOFF_PATH.write_text(text, encoding="utf-8")
            logger.info(f"Updated {cid} score in HANDOFF.md to {score}.")

    def _update_handoff_push(self, cid: str) -> None:
        """Updates HANDOFF.md with kernel push status."""
        if not HANDOFF_PATH.exists():
            return
        text = HANDOFF_PATH.read_text(encoding="utf-8")
        pat = rf"(- ID: `{cid}`[\s\S]*?submission status: )([^\n]+)"
        m = re.search(pat, text)
        if m:
            text = text[:m.start(2)] + "PUSHED & RUNNING (automated via autonomous_relay_daemon)" + text[m.end(2):]
            HANDOFF_PATH.write_text(text, encoding="utf-8")
            logger.info(f"Updated {cid} push status in HANDOFF.md.")

    def cycle(self) -> Dict[str, Any]:
        """Runs one full poll and action cycle."""
        logger.info("=== Running Autonomous Kernel Relay Cycle ===")
        active_running_gpu = 0
        statuses: Dict[str, Dict[str, Any]] = {}

        # 1. Query status of all known kernels
        for cid in ["C003", "C004", "C005", "C006", "C007", "C008", "C009"]:
            cfg = KERNELS_MONITORED[cid]
            if cid not in self.state["pushed_kernels"]:
                logger.info(f"[{cid}] Not yet pushed to Kaggle.")
                continue

            # If kernel already has a recorded submission ref or is completed/submitted, skip to prevent duplicate submissions
            if cid in self.state.get("submissions", {}):
                sub_info = self.state["submissions"][cid]
                if sub_info.get("ref"):
                    logger.info(f"[{cid}] Kernel already submitted (ref: {sub_info.get('ref')}, status: {sub_info.get('status')})")
                    continue

            info = self.query_kernel_status(cfg["kernel"])
            if info:
                statuses[cid] = info
                status_raw = str(info["status"])
                status_upper = status_raw.upper()
                logger.info(f"[{cid}] {cfg['kernel']}: status = {status_raw}")

                is_running = "RUNNING" in status_upper or "QUEUED" in status_upper
                is_complete = "COMPLETE" in status_upper
                is_error = "ERROR" in status_upper or "FAIL" in status_upper

                if is_running:
                    active_running_gpu += 1

                # If COMPLETE and not submitted, submit now!
                if is_complete:
                    sub_info = self.state.get("submissions", {}).get(cid, {})
                    if not sub_info.get("ref"):
                        logger.info(f">>> KERNEL {cid} HAS COMPLETED! Triggering auto-submit...")
                        self.submit_kernel(cid)
                elif is_error:
                    logger.error(f"[{cid}] Kernel encountered ERROR: {info.get('failure_message')}")
            time.sleep(2)  # Rate limit courtesy

        self.state["kernel_statuses"] = statuses
        self.state["last_check_ts"] = datetime.now(timezone.utc).isoformat()
        self._save_state()

        # 2. Check pending submissions for final LB scores
        self.check_pending_submissions()

        logger.info(f"Active GPU sessions: {active_running_gpu} / 2 maximum concurrent.")

        # 3. Check if a GPU slot is available to push staged candidates
        if active_running_gpu < 2:
            slots_available = 2 - active_running_gpu
            logger.info(f"{slots_available} GPU slot(s) available for deployment!")

            # Priority 1: C005 (Gold Fusion)
            if "C005" not in self.state["pushed_kernels"] and KERNELS_MONITORED["C005"]["dir"].exists():
                logger.info(">>> Triggering automated push of staged Candidate C005 (Gold Fusion)...")
                if self.push_kernel("C005"):
                    active_running_gpu += 1
                    slots_available -= 1

            # Priority 2: C006 (Consensus Fusion)
            if slots_available > 0 and "C006" not in self.state["pushed_kernels"] and KERNELS_MONITORED["C006"]["dir"].exists():
                logger.info(">>> Triggering automated push of staged Candidate C006...")
                if self.push_kernel("C006"):
                    active_running_gpu += 1
                    slots_available -= 1

            # Priority 3: C007 (Champion Frontier)
            if slots_available > 0 and "C007" not in self.state["pushed_kernels"] and KERNELS_MONITORED["C007"]["dir"].exists():
                logger.info(">>> Triggering automated push of staged Candidate C007 (Champion Frontier)...")
                if self.push_kernel("C007"):
                    active_running_gpu += 1
                    slots_available -= 1

            # Priority 4: C008 (Local Transformer Fusion)
            if slots_available > 0 and "C008" not in self.state["pushed_kernels"] and KERNELS_MONITORED["C008"]["dir"].exists():
                logger.info(">>> Triggering automated push of staged Candidate C008 (Local Transformer Fusion)...")
                if self.push_kernel("C008"):
                    active_running_gpu += 1
                    slots_available -= 1

            # Priority 5: C009 (Boost Geometric Fusion)
            if slots_available > 0 and "C009" not in self.state["pushed_kernels"] and KERNELS_MONITORED["C009"]["dir"].exists():
                logger.info(">>> Triggering automated push of staged Candidate C009 (Boost Geometric Fusion)...")
                if self.push_kernel("C009"):
                    active_running_gpu += 1
                    slots_available -= 1
        else:
            logger.info("GPU slots are fully occupied (2/2 running). Staged kernels remain ready.")

        return {
            "active_running_gpu": active_running_gpu,
            "statuses": statuses,
            "pushed": self.state["pushed_kernels"],
            "submissions": self.state["submissions"],
        }

    def run_daemon(self, poll_interval: int = 60) -> None:
        logger.info(f"Starting autonomous relay daemon loop (interval={poll_interval}s)...")
        try:
            while True:
                try:
                    self.cycle()
                except Exception as cycle_err:
                    logger.error(f"Cycle encountered unexpected error: {cycle_err}", exc_info=True)
                logger.info(f"Sleeping for {poll_interval}s before next check...")
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info("Daemon stopped by user.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Biohub Autonomous Kernel Relay & Submission Daemon")
    parser.add_argument("--once", action="store_true", help="Run a single poll-and-action cycle and exit")
    parser.add_argument("--daemon", action="store_true", help="Run daemon continuously in background loop")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds (default: 60)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate actions without submitting or pushing")
    args = parser.parse_args()

    daemon = AutonomousRelayDaemon(dry_run=args.dry_run)
    if args.once or not args.daemon:
        daemon.cycle()
    else:
        daemon.run_daemon(poll_interval=args.interval)


if __name__ == "__main__":
    main()
