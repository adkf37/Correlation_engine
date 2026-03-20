# Phase 11: Reporting

**Status:** Not Started
**Depends on:** Phase 09 (Finding schema), Phase 10 (FindingsDatabase)
**Estimated scope:** M

## Objective

Convert `Finding` records into human-readable summaries using rule-based templates (always available) and optionally enhanced by a local Ollama LLM. Generate self-contained HTML reports and email digests that can be triggered after each scan.

## Tasks

- [ ] Create `src/correlation_engine/reporting/__init__.py` (empty init)
- [ ] Create `src/correlation_engine/reporting/templates.py`:
  - `generate_template_summary(finding: Finding) -> str`
  - Rule-based natural language generation using `finding.trigger_types` as branching logic
  - Core template patterns:
    - High correlation: *"{A} and {B} show a {strong/moderate} {positive/inverse} correlation (r = {r:.2f})."*
    - With lag: *"...with {A} leading {B} by {|lag|} month(s) (r = {lag_r:.2f} at lag)."*
    - Regime change: *"This relationship has {strengthened/weakened} significantly over the trailing 90 days (z-score = {z:.1f}), suggesting a potential regime shift."*
    - Granger: *"Statistical tests suggest {A} Granger-causes {B} (p = {p:.3f}), indicating predictive information flows from {A} to {B}."*
    - Newly emerging: *"This correlation was not present in recent scans — it appears to be newly forming."*
  - Combines applicable clauses based on which triggers fired
  - Returns multi-sentence paragraph, ~50–150 words
- [ ] Create `src/correlation_engine/reporting/llm_summary.py`:
  - `enhance_with_llm(finding: Finding, model: str = "llama3.2", host: str = "http://localhost:11434") -> str | None`
  - Uses `ollama` Python package: `ollama.chat(model=model, messages=[...])`
  - Prompt: structured context block (series names, r, lag, trigger types, template summary) + instruction to write 2–3 sentence analyst insight
  - Graceful fallback: returns `None` if:
    - `ollama` package not importable
    - Ollama server not reachable (catch `ConnectionError` / `requests.exceptions.ConnectionError`)
    - Model not found on local instance
  - Never raises — always returns `str | None`
  - `is_ollama_available(host, model) -> bool` helper for pre-flight check
- [ ] Create `src/correlation_engine/reporting/html_report.py`:
  - `generate_html_report(findings: list[Finding], scan_metadata: dict, output_path: str | Path) -> Path`
  - Self-contained HTML (no external CDN dependencies — inline CSS only)
  - Structure: header (scan timestamp, N findings, top score), summary stats section, findings cards (ranked by score)
  - Each finding card shows: series names, score badge, trigger type pills, key stats (r, lag, z-score, Granger p), full summary text
  - Color coding: green = positive correlation, red = negative, yellow = regime change flag
  - Writes to `output_path` (e.g. `data/reports/report_{YYYYMMDD_HHMMSS}.html`)
  - Returns the resolved `Path` of the written file
  - Use Jinja2 template string (inline, no separate .html template file needed)
- [ ] Create `src/correlation_engine/reporting/email_digest.py`:
  - `send_email_digest(findings: list[Finding], config: EmailConfig) -> bool`
  - `EmailConfig` dataclass: `smtp_host`, `smtp_port`, `smtp_user`, `smtp_password`, `from_addr`, `to_addrs: list[str]`, `use_tls: bool = True`
  - Loaded from environment variables: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `ALERT_EMAIL` (comma-separated for multiple)
  - `build_email_config_from_env() -> EmailConfig | None` — returns None if any required var missing
  - Email format: HTML body (top 10 findings as a table + summary paragraph), plain-text fallback
  - Uses `smtplib` + `email.mime` (stdlib only — no new dependency)
  - Returns `True` on success, `False` on failure (logs error, never raises)
- [ ] Create `tests/test_reporting.py`:
  - Test `generate_template_summary` for each trigger type combination produces non-empty, grammatically correct strings
  - Test `generate_template_summary` with all 6 triggers fires → includes all clause types
  - Test `enhance_with_llm` returns None when ollama unavailable (mock the import/connection)
  - Test `generate_html_report` produces valid HTML file containing all finding IDs
  - Test `build_email_config_from_env` returns None if SMTP_HOST missing, EmailConfig if all set
  - Test `send_email_digest` with mocked SMTP connection

## Key Files

- `src/correlation_engine/reporting/__init__.py` — create: empty init
- `src/correlation_engine/reporting/templates.py` — create: rule-based summary generator
- `src/correlation_engine/reporting/llm_summary.py` — create: Ollama integration (optional)
- `src/correlation_engine/reporting/html_report.py` — create: HTML report generator using Jinja2
- `src/correlation_engine/reporting/email_digest.py` — create: smtplib email sender
- `tests/test_reporting.py` — create: reporting unit tests

## Acceptance Criteria

- `generate_template_summary` always returns a non-empty string for any valid `Finding`
- LLM enhancement returns `None` (not raises) when Ollama is not running
- HTML report is a valid, self-contained HTML file that opens in a browser without errors
- HTML report file written to `data/reports/` with timestamped filename
- Email send returns `False` (not raises) when SMTP credentials missing or server unreachable
- All tests pass, including LLM tests that mock the Ollama connection

## Notes

- **Ollama setup**: user runs `ollama pull llama3.2` once; the `ollama` Python package (`pip install ollama`) is the client. Add `ollama>=0.1` to `pyproject.toml` as an optional dependency under `[project.optional-dependencies] llm = ["ollama>=0.1"]`
- The default model `llama3.2` is fast and capable for summarization. The model name should be configurable via `config/scan_config.yaml` (Phase 12) — field `llm_model: "llama3.2"`
- Ollama prompt design: keep context tight (<500 tokens input) to stay fast. Include: series A name, series B name, r value, lag, trigger types, and the template summary. Ask for "a concise 2-3 sentence analyst interpretation."
- For HTML: use a minimal CSS reset + card layout. The report should be readable as-is in any browser without internet access (no CDN fonts, no external scripts)
- Email: append domain/SMTP config instructions as commented-out example in `.env` file (in Phase 12)
- Jinja2 is already a dependency (comes with Streamlit), safe to import without adding to pyproject.toml — but add it explicitly for clarity
