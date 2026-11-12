# Archive of 2026-11-09

Results and code of the analysis as it stood before the redo of 9 November 2026 (anomaly detection and
causal relationships, with River). Nothing here is used by the current pipeline.

| Folder | Content |
|---|---|
| `out/` | Old outputs: streaming profile (`profile/`), `hourly.parquet`, `cross.json`, `spectral.json`, `seasonal.json`, old `anomaly_summary.json` and `anomalies.parquet`, `influence.json`, `report_data.json`, and the pilot research-question outputs (`rq1.json`, `rq2.json`, `rq*_examples.parquet`, `rq2_episodes.parquet`), with run logs. |
| `report/` | Old pages: `index.html` (main report), `influence.html`, `rq.html`, and the templates they were built from. |
| `code/` | Scripts that were removed (`profile_stream.py`, `cross_metric.py`, `spectral.py`, `seasonal.py`, `influence.py`, `influence_report.py`) the old versions of `anomaly.py` and `build_report.py`, and the pilot RQ1 script `rq1_nbm.py` (moved here on 10 November 2026, replaced by `analysis/rq1.py`). |
| `slides/` | `smart_sign_rq.pptx`. |
| `Smart-Sign Telemetry Profile.md` | Old text profile of the dataset. |

Notes
- The removed scripts and the old `anomaly.py` / `build_report.py` are also in git at commit `93eb148`.
- Large files (`*.parquet`, `profile/`, `*.log`) are ignored by git; they exist only on this machine.
- Published copies of the old pages (private artifacts): main report https://claude.ai/artifact/5quPowZC97hz9gRLVbguTA,
  research-question report https://claude.ai/artifact/JsGFMx2L2fnvfa6hWgWKXA. The influence report was a separate artifact.
- The pilot outputs of the research-question scripts are here (`out/rq1.json`, `out/rq2.json`, ...). `rq2_graph.py` and
  `rq_report.py` are still in `analysis/`; `rq_report.py` builds the pilot report and needs those two files in `analysis/out/`.
- `rq1_nbm.py` is not in git (it was never committed); this folder holds the only copy. Run it from `analysis/` if needed.
