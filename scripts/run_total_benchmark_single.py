"""
Run the total benchmark pipeline for a single model.
Reuses the existing run_evaluation_cycle from run_total_benchmark.py.
"""
import argparse
import os
import pandas as pd

from run_total_benchmark import run_evaluation_cycle
from src.loader import get_cifar10_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="RobustBench model name")
    parser.add_argument("--model-alias", required=True, help="Display name in outputs")
    parser.add_argument("--layers", required=True, help="Comma-separated layer names")
    parser.add_argument("--limit-samples", type=int, default=10000)
    parser.add_argument("--profiling-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="total_benchmark_results")
    args = parser.parse_args()

    layers = [l.strip() for l in args.layers.split(",") if l.strip()]

    test_loader = get_cifar10_loader(batch_size=32, n_examples=args.limit_samples)
    profiling_loader = get_cifar10_loader(batch_size=128, n_examples=args.profiling_samples)

    results = run_evaluation_cycle(
        args.model_name,
        args.model_alias,
        layers,
        test_loader,
        profiling_loader,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"benchmark_{args.model_alias}.csv")
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"[DONE] Saved: {out_csv}")


if __name__ == "__main__":
    main()
