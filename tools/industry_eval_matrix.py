#!/usr/bin/env python3
import sys, argparse
from tools.benchmark import run_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.device)
