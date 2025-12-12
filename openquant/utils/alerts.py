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
import smtplib
from email.mime.text import MIMEText

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
        # Try email if configured
        _send_email(subject, body, severity)
        # Try SMS/webhook if configured via OPENQUANT_SMS_WEBHOOK
        sms_url = os.getenv("OPENQUANT_SMS_WEBHOOK")
        if sms_url:
            try:
                data = json.dumps({"text": f"[{severity}] {subject}: {body}"}).encode("utf-8")
                req = urllib.request.Request(sms_url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    _ = resp.read()
            except Exception as e:
                LOGGER.warning(f"SMS webhook failed: {e}")
        return
    try:
        data = json.dumps({"subject": subject, "body": body, "severity": severity}).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310 (webhook opt-in)
            _ = resp.read()
    except Exception as e:
        LOGGER.warning(f"Alert webhook failed: {e}")
        _send_email(subject, body, severity)


def _send_email(subject: str, body: str, severity: str) -> None:
    host = os.getenv("OPENQUANT_SMTP_HOST")
    port = int(os.getenv("OPENQUANT_SMTP_PORT", "0") or 0)
    user = os.getenv("OPENQUANT_SMTP_USER")
    pwd = os.getenv("OPENQUANT_SMTP_PASS")
    to_addr = os.getenv("OPENQUANT_SMTP_TO")
    if not (host and port and user and pwd and to_addr):
        return
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = f"[{severity}] {subject}"
        msg["From"] = user
        msg["To"] = to_addr
        with smtplib.SMTP(host, port, timeout=5) as s:
            s.starttls()
            s.login(user, pwd)
            s.sendmail(user, [to_addr], msg.as_string())
    except Exception as e:
        LOGGER.warning(f"SMTP email failed: {e}")

