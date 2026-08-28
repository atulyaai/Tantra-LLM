#!/usr/bin/env python3
import sys, argparse
from tools.dataset import generate_gold_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    generate_gold_datasets(force=args.force)
