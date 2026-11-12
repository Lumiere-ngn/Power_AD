# Archive of the first RQ1 run (10 November 2026)

The first run of `analysis/rq1.py`, with every device in the data, including the `demo` (6) and `show` (1) units.
It was the only planned scoring of 1 August to 6 October. On the same day the user asked for the whole run to be
redone without the `demo` and `show` units; that second run is a deviation and is recorded in the deviation log of
`analysis/protocol/PROTOCOL.md`. Its results replace these in `analysis/protocol/RESULTS_RQ1.md`.

| File | Content |
|---|---|
| `RESULTS_RQ1.md` | Results text of the first run |
| `out/rq1_dev.json`, `out/rq1_dev.log`, `out/rq1_locked.json` | Development stage and locked settings |
| `out/rq1_test.json`, `out/rq1_test.log`, `out/rq1_test_predictions.parquet` | The test, as scored the first time |
| `rq1.py`, `rq_common.py` | The scripts as they were for this run |

Large files (`*.parquet`, `*.log`) are ignored by git; they exist only on this machine.
The Google Docs copy of these results is "RQ1 results (10 November 2026, v2)".
