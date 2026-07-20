"""Create deterministic stratified folds from a path,label CSV manifest."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=591884092)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    if not {"path", "label"}.issubset(frame.columns):
        raise ValueError("manifest must contain path,label columns")
    args.output.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(frame.path, frame.label)):
        frame.iloc[train_idx].to_csv(args.output / f"train_fold{fold}.csv", index=False)
        frame.iloc[test_idx].to_csv(args.output / f"test_fold{fold}.csv", index=False)


if __name__ == "__main__":
    main()

