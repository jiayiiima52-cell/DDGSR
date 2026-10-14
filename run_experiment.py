"""Run DDGSR with the shared five-seed protocol."""
import argparse
from pathlib import Path
import subprocess
import sys
from experiment_config import SEEDS, EVAL_KS

ROOT = Path(__file__).resolve().parent


def command(seed, data_root, output_dir, device=None):
    args = [sys.executable, str(ROOT / "train.py"),
        "--dataset", "tssgcf", "--data-root", str(data_root),
        "--full-train", "--metrics-only", "--variant", "ddgsr",
        "--no-contrastive", "--disagreement-only-gate",
        "--epochs", "22", "--dim", "256", "--layers", "2", "--neg-layers", "1",
        "--batch-size", "512", "--learning-rate", "0.0005", "--weight-decay", "0.00001",
        "--cl-weight", "0", "--temperature", "0.5", "--graph-ratio", "0.25",
        "--graph-weight", "0.4", "--update-every", "5", "--view-fusion-weight", "0.1",
        "--semantic-score-weight", "0.8", "--conflict-gate-beta", "2",
        "--disagreement-residual-gain", "2.5", "--top-k", "10",
        "--eval-ks", ",".join(map(str, EVAL_KS)), "--seed", str(seed),
        "--run-name", f"DDGSR_s{seed}", "--output-dir", str(output_dir)]
    if device:
        args += ["--device", device]
    return args


def main():
    parser = argparse.ArgumentParser(description="DDGSR experiment runner")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ddgsr")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for seed in args.seeds:
        cmd = command(seed, args.data_root.resolve(), args.output_dir.resolve(), args.device)
        if args.dry_run:
            print(subprocess.list2cmdline(cmd))
        else:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
