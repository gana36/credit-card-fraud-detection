import os
import yaml
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def generate_report(cfg_path: str = "configs/monitoring.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    ref = cfg.get("reference_sample_path")
    cur = cfg.get("current_sample_path")
    out_dir = cfg.get("report_output_dir", "reports/")
    os.makedirs(out_dir, exist_ok=True)

    if not (os.path.exists(ref) and os.path.exists(cur)):
        print("Reference or current sample not found; skipping drift report.")
        return

    ref_df = pd.read_parquet(ref)
    cur_df = pd.read_parquet(cur)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    report.save_html(os.path.join(out_dir, "drift_report.html"))
    print({"status": "ok", "output": os.path.join(out_dir, "drift_report.html")})


if __name__ == "__main__":
    generate_report()
