# WUR indicators

## Data processing workflow
00_make_buffers.py
│
├─> wrz.gpkg
└─> wrz_buffer.gpkg
    │
    ├─> 01_aggregate_over_buffers.py
    │   │
    │   └─> aggregated_series.parquet
    │       │
    │       ├─> 02_calculate_indicators.py
    │       │   │
    │       │   ├─> thresholds.csv (ignored)
    │       │   └─> indicator_series.parquet ───┐
    │       │                                   │
    │       └─> 03_calculate_standardised.py    |
    │           │                               │
    │           ├─> parameters.csv              │
    │           ├─> standardised_series.parquet ┤
    │           └─> si_plots/*.png              │
    │                                           │
    └─────────────────────────────────────────> 05_make_timeseries.py ────────────┐
                                                │   (also uses monthly_los files) |
                                                │                                 |
04_extract_monthly_restrictions.py ─────────> monthly_los_level[0-4]_melted.csv ──┘
                                                │
                                                └─> ts_with_levels.csv (FINAL OUTPUT)


Key data flows:

00: Creates buffered WRZ geometries
01: Aggregates W@H climate data (prbc, ep) over WRZ buffers
02: Calculates drought indicators from aggregated series
03: Standardizes variables (SPI, SPEI) from aggregated series
04: Extracts monthly restriction levels from LOS data (independent branch)
05: Combines indicators, standardized indices, and LOS levels into final timeseries

Main output: ts_with_levels.csv
