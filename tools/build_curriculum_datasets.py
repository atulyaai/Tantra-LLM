#!/usr/bin/env python3
import sys, argparse
from tools.dataset import build_4track_curriculum

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_4track_curriculum(force=args.force)
