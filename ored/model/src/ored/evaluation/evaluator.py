from __future__ import annotations

import argparse
from typing import List, Optional

from ored.evaluation.lm_evaluator import evaluate_language_model
from ored.inference.predictor import DEFAULT_CHECKPOINT


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained Ored.ai checkpoint.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--samples", type=int, default=3,
                        help="how many text samples to score")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="sampling temperature for the samples")
    args = parser.parse_args(argv)

    evaluate_language_model(
        args.checkpoint,
        splits=args.splits,
        device=args.device,
        n_samples=args.samples,
        temperature=args.temperature,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
