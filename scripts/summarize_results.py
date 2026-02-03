"""
Summarize all available experiment results into a single report.
Pure Python (no pandas). Safe to run on cluster/login nodes.

Usage:
  python3 scripts/summarize_results.py
  python3 scripts/summarize_results.py --output-dir summary_outputs
"""
import argparse
import csv
import json
import os
from datetime import datetime
from glob import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _parse_float(val):
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    if "±" in s:
        s = s.split("±", 1)[0].strip()
    try:
        return float(s)
    except ValueError:
        return None


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _summarize_oodrb_csv(path):
    rows = _read_csv(path)
    if not rows:
        return None
    metrics = {}
    for key in ("auroc", "fpr95", "aupr_in", "aupr_out"):
        vals = [_parse_float(r.get(key)) for r in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            metrics[key] = sum(vals) / len(vals)
    return {
        "path": path,
        "rows": len(rows),
        "mean_metrics": metrics,
    }


def _summarize_jpeg_subset_csv(path):
    rows = _read_csv(path)
    return {
        "path": path,
        "rows": len(rows),
        "results": rows,
    }


def _latest_dir(pattern):
    candidates = sorted(glob(pattern))
    return candidates[-1] if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="summary_outputs")
    args = parser.parse_args()

    out_root = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(out_root, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"summary_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    report = {
        "timestamp": stamp,
        "phase3": {},
        "oodrb_corruption": {},
        "oodrb_natural": {},
        "jpeg_subsets": [],
        "total_benchmark": {},
        "aps_vs_nonaps": {},
        "idonly_thresholds": {},
    }

    # Phase3 (latest in archive)
    phase3_latest = _latest_dir(os.path.join(PROJECT_ROOT, "archive", "*", "phase3_output_*"))
    if phase3_latest:
        results_json = os.path.join(phase3_latest, "results.json")
        stats_json = os.path.join(phase3_latest, "statistical_summary.json")
        if os.path.isfile(results_json):
            report["phase3"]["results_json"] = results_json
            report["phase3"]["results"] = json.load(open(results_json))
        if os.path.isfile(stats_json):
            report["phase3"]["statistical_summary_json"] = stats_json
            report["phase3"]["statistical_summary"] = json.load(open(stats_json))

    # OODRobustBench results (corruption + natural)
    oodrb_dir = os.path.join(PROJECT_ROOT, "oodrb_results")
    if os.path.isdir(oodrb_dir):
        for path in sorted(glob(os.path.join(oodrb_dir, "oodrb_nac_*.csv"))):
            rows = _read_csv(path)
            if not rows:
                continue
            if rows[0].get("shift_type") == "natural":
                report["oodrb_natural"].setdefault("files", []).append(_summarize_oodrb_csv(path))
            else:
                report["oodrb_corruption"].setdefault("files", []).append(_summarize_oodrb_csv(path))

    # JPEG subset outputs
    jpeg_dir = os.path.join(PROJECT_ROOT, "total_benchmark_results")
    for path in sorted(glob(os.path.join(jpeg_dir, "jpeg_subset_*.csv"))):
        report["jpeg_subsets"].append(_summarize_jpeg_subset_csv(path))

    # Total benchmark (full)
    bench_csv = os.path.join(PROJECT_ROOT, "total_benchmark_results", "benchmark_results.csv")
    if os.path.isfile(bench_csv):
        report["total_benchmark"]["path"] = bench_csv
        report["total_benchmark"]["rows"] = len(_read_csv(bench_csv))

    # APS vs Non-APS (OpenOOD)
    aps_csv = os.path.join(PROJECT_ROOT, "ood_coverage", "analysis", "nac_aps_vs_nonaps.csv")
    if os.path.isfile(aps_csv):
        report["aps_vs_nonaps"]["path"] = aps_csv
        report["aps_vs_nonaps"]["rows"] = len(_read_csv(aps_csv))

    # ID-only thresholds (latest)
    idonly_latest = _latest_dir(os.path.join(PROJECT_ROOT, "ood_coverage", "analysis", "nac_idonly_thresholds_*"))
    if idonly_latest:
        thr_csv = os.path.join(idonly_latest, "threshold_table.csv")
        if os.path.isfile(thr_csv):
            report["idonly_thresholds"]["path"] = thr_csv
            report["idonly_thresholds"]["rows"] = len(_read_csv(thr_csv))

    # Write JSON
    out_json = os.path.join(out_dir, "summary.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    # Write Markdown summary
    out_md = os.path.join(out_dir, "summary.md")
    with open(out_md, "w") as f:
        f.write(f"# Results Summary ({stamp})\n\n")
        f.write("## Phase3\n")
        if report["phase3"]:
            f.write(f"- results.json: {report['phase3'].get('results_json','')}\n")
            if "results" in report["phase3"]:
                cfg = report["phase3"]["results"].get("config", {})
                metric = report["phase3"]["results"].get("metrics", {})
                f.write(f"- config: {cfg}\n")
                f.write(f"- metrics: {metric}\n")
        else:
            f.write("- (not found)\n")

        f.write("\n## OODRobustBench Corruption\n")
        for item in report["oodrb_corruption"].get("files", []):
            f.write(f"- {os.path.basename(item['path'])}: rows={item['rows']} mean={item['mean_metrics']}\n")

        f.write("\n## OODRobustBench Natural Shift\n")
        for item in report["oodrb_natural"].get("files", []):
            f.write(f"- {os.path.basename(item['path'])}: rows={item['rows']} mean={item['mean_metrics']}\n")

        f.write("\n## JPEG Subsets\n")
        if report["jpeg_subsets"]:
            for item in report["jpeg_subsets"]:
                f.write(f"- {os.path.basename(item['path'])}: rows={item['rows']}\n")
        else:
            f.write("- (no jpeg_subset_*.csv found)\n")

        f.write("\n## Total Benchmark\n")
        if report["total_benchmark"]:
            f.write(f"- {report['total_benchmark'].get('path','')} (rows={report['total_benchmark'].get('rows')})\n")
        else:
            f.write("- (benchmark_results.csv not found)\n")

        f.write("\n## APS vs Non-APS\n")
        if report["aps_vs_nonaps"]:
            f.write(f"- {report['aps_vs_nonaps'].get('path','')} (rows={report['aps_vs_nonaps'].get('rows')})\n")
        else:
            f.write("- (nac_aps_vs_nonaps.csv not found)\n")

        f.write("\n## ID-only Thresholds\n")
        if report["idonly_thresholds"]:
            f.write(f"- {report['idonly_thresholds'].get('path','')} (rows={report['idonly_thresholds'].get('rows')})\n")
        else:
            f.write("- (threshold_table.csv not found)\n")

    print(f"[DONE] Summary written to: {out_dir}")


if __name__ == "__main__":
    main()
