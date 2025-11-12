# Security Policy

This repository enforces strict handling of credentials and sensitive data.

## 1) Secrets and credentials
- Never hardcode credentials, API keys, or passwords in source code.
- Use environment variables for all secrets.
- Provide examples via `.env.example`; do not commit real `.env` files.
- The MT5 credentials must be set via these variables before runs:
  - `OQ_MT5_TERMINAL` — full path to `terminal64.exe`
  - `OQ_MT5_SERVER` — server name (e.g., `MetaQuotes-Demo`)
  - `OQ_MT5_LOGIN` — account login (integer)
  - `OQ_MT5_PASSWORD` — account password

## 2) Logging and PII
- Do not log secrets. Error messages should not echo credentials.
- MT5 `last_error()` may be logged for diagnostics, but never include passwords.

## 3) Data handling
- Portfolio ledgers are stored locally in DuckDB under `data/`. Treat as local-only.
- Do not commit personal or sensitive result data.

## 4) Dependency and supply chain
- Install packages only via the package manager (pip/poetry). No manual edits of lockfiles.
- Keep dependencies up-to-date and pinned where appropriate.

## 5) Execution safety (MT5)
- Default assumption is DEMO usage. Confirm and review guardrails before any live trading.
- Guardrails (max drawdown, CVaR, daily loss cap) should be enabled when running unattended.
- Concentration limits and exposure caps must be set conservatively for live.

## 6) Vulnerability reporting
- If you discover a security issue, open a private issue or contact the maintainer. Avoid public disclosure until a fix is available.
