# Project Policies

1. Single-responsibility modules and clean architecture.
2. Reproducibility: deterministic configs (YAML), seeds, run manifests.
3. Security: never commit secrets; use .env (gitignored) + environment variables.
4. Testing-first: unit tests for every module; integration smoke tests.
5. Data hygiene: timezone-aware, no look-ahead bias, purged CV for overlapping samples.
6. Risk-first: selection under constraints (MaxDD, CVaR, exposure).
7. Documentation: code comments, README usage, ROADMAP tracked.
8. No unnecessary files; keep repo lean; remove dead code promptly.
9. Dependency management via package managers; pin if needed; avoid heavy deps unless justified.
10. Minimal privileges: connectors only load keys from env and fail closed.

