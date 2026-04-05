"""Convenience entrypoint: ``python -m liquid_lehome --mode train|eval``."""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "--mode":
        print("Usage: python -m liquid_lehome --mode train|eval [options]")
        print("  Train: python -m liquid_lehome --mode train --config configs/lehome/liquid_mdn_lehome.json")
        print("  Eval:  python -m liquid_lehome --mode eval --checkpoint path/to/best.pt")
        sys.exit(1)

    mode = sys.argv[2]
    # Remove --mode and its value so sub-parsers see the remaining args
    sys.argv = [sys.argv[0]] + sys.argv[3:]

    if mode == "train":
        from .train import main as train_main
        train_main()
    elif mode == "eval":
        from .eval import main as eval_main
        eval_main()
    else:
        print(f"Unknown mode: {mode}. Use 'train' or 'eval'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
