from __future__ import annotations
"""Simple alerting utilities.

- format_run_summary: produce a compact human-readable summary string
- send_alert: log and optionally POST to a webhook URL (from arg or env OPENQUANT_ALERT_WEBHOOK)

No third-party deps; uses urllib from stdlib. If webhook not provided, only logs.
"""
from typing import Dict, Any, Optional
import os
import json
import urllib.request

from .logging import get_logger

LOGGER = get_logger(__name__)


def format_run_summary(summary: Dict[str, Any]) -> str:
    """Format a human-readable multi-line summary.
    Keys expected (optional): run_id, total_rows, ok_before_caps, ok_after_caps,
    cap_demotions, guardrail_violations, new_best_count
    """
    parts = []
    rid = summary.get("run_id")
    if rid:
        parts.append(f"Run {rid}")
    parts.append(f"rows={summary.get('total_rows', 0)}")
    if "ok_before_caps" in summary and "ok_after_caps" in summary:
        parts.append(f"ok_before_caps={summary.get('ok_before_caps')} -> ok_after_caps={summary.get('ok_after_caps')}")
        parts.append(f"cap_demotions={summary.get('cap_demotions', 0)}")
    if "guardrail_violations" in summary:
        parts.append(f"guardrail_violations={summary.get('guardrail_violations', 0)}")
    if "new_best_count" in summary:
        parts.append(f"new_or_changed_best={summary.get('new_best_count', 0)}")
    return " | ".join(parts)


def send_alert(subject: str, body: str, *, severity: str = "INFO", webhook_url: Optional[str] = None) -> None:
    """Send an alert by logging and optional webhook POST.
    If webhook_url is None, tries env OPENQUANT_ALERT_WEBHOOK.
    Payload: {"subject": ..., "body": ..., "severity": ...}
    """
    url = webhook_url or os.getenv("OPENQUANT_ALERT_WEBHOOK")
    log_msg = f"[{severity}] {subject}: {body}"
    if severity.upper() in ("ERROR", "CRITICAL"):
        LOGGER.error(log_msg)
    elif severity.upper() == "WARNING":
        LOGGER.warning(log_msg)
    else:
        LOGGER.info(log_msg)

    if not url:
        return
    try:
        data = json.dumps({"subject": subject, "body": body, "severity": severity}).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310 (webhook opt-in)
            _ = resp.read()
    except Exception as e:
        LOGGER.warning(f"Alert webhook failed: {e}")

