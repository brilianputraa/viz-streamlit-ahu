# Option A: Single DB + Pgbouncer + Materialized Views

## Goal
Improve dashboard reliability and latency for a small-scale workload without adding a read replica. This design keeps one TimescaleDB instance, adds connection pooling, and introduces pre-aggregated views for common dashboard queries.

## Context
- Current stack: TimescaleDB + Airflow ETL + Streamlit dashboard.
- Pain points: DB connection spikes, long queries on `ahu_data.energy_readings`, and ETL/read contention.
- Constraints: Avoid high cost or heavy ops; keep changes minimal and reversible.

## Architecture Summary
- One TimescaleDB instance remains the system of record for writes and reads.
- Add a pgbouncer container with two pools:
  - `ahu_write`: smaller pool for Airflow ETL
  - `ahu_read`: larger pool for Streamlit
- Introduce materialized views for heavy dashboard queries.

```
Airflow ETL  --->  pgbouncer (ahu_write)  --->  TimescaleDB
Streamlit   --->  pgbouncer (ahu_read)   --->  TimescaleDB
                   |
                   +--> Materialized Views (daily/monthly rollups)
```

## Data Flow
1. Airflow writes into `ahu_data.energy_readings`.
2. Airflow runs `REFRESH MATERIALIZED VIEW` for dashboard summaries.
3. Streamlit reads summary data from materialized views.
4. Streamlit uses raw tables only for drill-down queries with narrow date ranges.

## Components
### 1) Pgbouncer
- Two database aliases pointing to the same TimescaleDB host.
- Pooling mode:
  - Streamlit: `transaction` pooling
  - Airflow: `transaction` or `session` (depending on ETL needs)
- Example settings:
  - `max_client_conn`: 200
  - `default_pool_size`: 20 (read), 5 (write)
  - `reserve_pool_size`: 5

### 2) Materialized Views
Recommended views:
- `mv_daily_ahu_costs`
  - Daily per-AHU costs: steam, cold water, electricity, total.
- `mv_monthly_ahu_costs`
  - Monthly rollups for trend charts.
- `mv_daily_ahu_kwh`
  - Daily kWh by coil (HCV/DH_HCV/CCV/PC_CCV).

Refresh strategy:
- Run after ETL completes.
- Use `CONCURRENTLY` if a unique index is available.
- If not, refresh during low-traffic windows.

### 3) Streamlit Query Routing
- Summary charts query materialized views.
- Drill-downs query raw tables within user-selected date ranges.
- Add lightweight caching (`st.cache_data`) for repeated view queries.

## Operational Notes
- Add DB-level monitoring for:
  - active connections
  - long-running queries
  - materialized view refresh time
- Use ETL to enforce indexed columns on date/time and AHU identifiers.
- Keep date filters mandatory on raw queries.

## Risks and Mitigations
- ETL overlap with view refresh: schedule refresh after ETL task completes.
- Large view refresh times: limit data scope or refresh only recent windows.
- Connection exhaustion: tune pgbouncer limits and idle timeouts.

## Rollout Steps (Minimal Disruption)
1. Deploy pgbouncer and point Streamlit to `ahu_read`.
2. Point Airflow to `ahu_write`.
3. Create materialized views (daily, monthly).
4. Add a refresh step to the ETL DAG.
5. Update Streamlit adapter to prefer views for summary charts.

## Success Criteria
- Streamlit dashboards stay responsive during ETL runs.
- No connection spikes or DB timeouts during peak usage.
- Summary charts consistently load under a few seconds.
