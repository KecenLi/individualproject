"""
Run JPEG-only benchmark with a subset of eps values.
Reuses run_total_benchmark.run_evaluation_cycle to stay consistent.
"""
import argparse
import csv
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from run_total_benchmark import run_evaluation_cycle
from src.loader import get_cifar10_loader


def _parse_eps_list(value: str):
    if not value:
        return []
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jpeg-norm", required=True, choices=["linf", "l2", "l1"])
    parser.add_argument("--jpeg-eps", required=True, help="Comma-separated eps list")
    parser.add_argument("--jpeg-iters", type=int, default=200)
    parser.add_argument("--model-name", default="Standard")
    parser.add_argument("--model-alias", default="ResNet18_Std")
    parser.add_argument("--layers", default="block1.layer.2,block2.layer.2,block3.layer.2")
    parser.add_argument("--limit-samples", type=int, default=10000)
    parser.add_argument("--profiling-samples", type=int, default=1000)
    parser.add_argument("--output-dir", default="total_benchmark_results")
    args = parser.parse_args()

    layers = [l.strip() for l in args.layers.split(",") if l.strip()]
    eps_list = _parse_eps_list(args.jpeg_eps)
    if not eps_list:
        raise ValueError("jpeg-eps list is empty.")

    test_loader = get_cifar10_loader(batch_size=32, n_examples=args.limit_samples)
    profiling_loader = get_cifar10_loader(batch_size=128, n_examples=args.profiling_samples)

    # JPEG-only perturbations (include Clean baseline for AUROC)
    perturbations = {"Clean": []}
    for eps in eps_list:
        key = f"JPEG_{args.jpeg_norm.upper()}_{eps}"
        perturbations[key] = [("jpeg", {"norm": args.jpeg_norm, "eps": eps, "n_iters": args.jpeg_iters})]

    results = run_evaluation_cycle(
        args.model_name,
        args.model_alias,
        layers,
        test_loader,
        profiling_loader,
        profiling_samples=args.profiling_samples,
        perturbations_override=perturbations,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"{args.jpeg_norm}_eps_{'-'.join(str(e) for e in eps_list)}"
    out_csv = os.path.join(args.output_dir, f"jpeg_subset_{tag}.csv")
    if not results:
        raise RuntimeError("No results were generated.")
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"[DONE] Saved: {out_csv}")


if __name__ == "__main__":
    main()
