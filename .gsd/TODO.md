# TODO.md — Pending Items

> Quick capture of ideas, tasks, and issues.
>
> Use `/add-todo` to add items, `/check-todos` to view.

## Priority Levels

| Level | Use For |
|-------|---------|
| `high` 🔴 | Blocking issues, urgent fixes |
| `medium` 🟡 | Normal priority (default) |
| `low` 🟢 | Nice-to-have, future ideas |

---

## Items

<!-- Active todos below -->

- [ ] Plan Phase 1 atomic tasks (uv setup, ingestion, features, pipeline) `high` — 2026-11-08
- [ ] Initialize `pyproject.toml` and generate `uv.lock` with core and dev dependencies `high` — 2026-11-08
- [ ] Build `src/data/ingestion.py` with `yfinance` download, caching in `data/raw/`, and `Data/*.csv` fallback `high` — 2026-11-08
- [ ] Implement log returns calculation and price trajectory reconstruction utility in `src/features/returns.py` `high` — 2026-11-08
- [ ] Implement technical indicator suite in `src/features/technical.py` `medium` — 2026-11-08
- [ ] Implement leak-free preprocessing and serialized scaling in `src/features/pipeline.py` `high` — 2026-11-08
- [ ] Implement PyTorch GRU with LayerNorm and MC Dropout in `src/models/gru.py` `medium` — 2026-11-08
- [ ] Implement NaivePersistenceBaseline in `src/models/baseline.py` `medium` — 2026-11-08
- [ ] Implement PyTorch trainer in `src/models/trainer.py` driven by `configs/config.yaml` `medium` — 2026-11-08
- [ ] Build Walk-Forward backtester with transaction costs in `src/evaluation/backtester.py` `medium` — 2026-11-08
- [ ] Build FastAPI serving backend in `src/api/` `medium` — 2026-11-08
- [ ] Build interactive Streamlit dashboard in `src/dashboard/` `medium` — 2026-11-08
- [ ] Build pytest suite executed via `uv run pytest` in `tests/` `medium` — 2026-11-08
- [ ] Create multi-stage `Dockerfile`, `docker-compose.yml`, and `Makefile` `medium` — 2026-11-08

---

*Last updated: 2026-11-08*
