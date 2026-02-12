# Test Data

Place small representative TradeStation XLSX trade list and MTM files here for ingest/compute tests.

- Keep files modest in size so test runs stay fast.
- Use the `tradeslist_<SYMBOL>_<INTERVAL>_<STRATEGY>.xlsx` naming pattern so metadata parsing remains deterministic.
- These files are exercised by `tests/test_ingest.py` and `tests/test_compute.py`.

For repo-wide guidance, see `AGENTS.md`. For code navigation, see `REPO_MAP.md`.
