#!/usr/bin/env python3
"""Automated submit script for Candidate C005 (Gold Fusion).

Submits kernel taeyangg4/biohub-c005-gold-fusion v3 targeting score >= 0.959 (Gold Medal safe zone).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from autonomous_relay_daemon import AutonomousRelayDaemon, KERNELS_MONITORED


def main() -> None:
    daemon = AutonomousRelayDaemon()
    cid = "C005"
    cfg = KERNELS_MONITORED[cid]
    status_info = daemon.query_kernel_status(cfg["kernel"])
    status = status_info.get("status") if status_info else "UNKNOWN"
    print(f"Kernel {cfg['kernel']} status: {status}")

    if "COMPLETE" not in str(status).upper():
        print(f"Kernel {cfg['kernel']} is not yet COMPLETE (status: {status}). Skipping submit.")
        return

    ref = daemon.submit_kernel(cid)
    if ref:
        print(f"Successfully submitted {cid}! Ref: {ref}")
    else:
        print(f"Failed to submit {cid}.")


if __name__ == "__main__":
    main()
