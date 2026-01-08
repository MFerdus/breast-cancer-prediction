\
"""
Train a breast-cancer classifier with PyCaret and export the best model.

This script is intentionally dataset-agnostic:
- It expects a CSV with a binary target column (default: Patient_Status).
- It runs PyCaret setup + compare_models + finalize_model + save_model.
- It writes the serialized model under ./models/

Example:
    python -m src.train --data data/BRCA\ Data.csv --target Patient_Status
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pycaret.classification import (
    setup,
    compare_models,
    finalize_model,
    save_model,
    pull,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to CSV dataset.")
    p.add_argument("--target", type=str, default="Patient_Status", help="Target column name.")
    p.add_argument("--session_id", type=int, default=42, help="Random seed.")
    p.add_argument("--fold", type=int, default=5, help="CV folds.")
    p.add_argument("--model_name", type=str, default="best_model", help="Base filename for saved model.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    if args.target not in df.columns:
        raise ValueError(
            f"Target column '{args.target}' not in dataset columns. "
            f"Available columns: {list(df.columns)[:30]}..."
        )

    # Basic setup (tune these to match your notebook if needed)
    setup(
        data=df,
        target=args.target,
        session_id=args.session_id,
        fold=args.fold,
        train_size=0.70,
        # Helpful defaults. You can disable or customize them as needed.
        remove_outliers=True,
        remove_multicollinearity=True,
        numeric_imputation="median",
        silent=True,
        verbose=False,
    )

    best = compare_models()
    _ = pull()  # comparison table (optional, useful in notebooks/logging)
    final = finalize_model(best)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True, parents=True)
    save_model(final, str(models_dir / args.model_name))

    print(f"✅ Saved model to: {models_dir / (args.model_name + '.pkl')}")


if __name__ == "__main__":
    main()
