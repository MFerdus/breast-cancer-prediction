\
"""
Batch prediction script.

Example:
    python -m src.predict --model models/best_model --input data/new_patients.csv --output reports/predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pycaret.classification import load_model, predict_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="Path base for PyCaret model (without .pkl) OR full path to .pkl")
    p.add_argument("--input", type=str, required=True, help="Input CSV for inference.")
    p.add_argument("--output", type=str, default="reports/predictions.csv", help="Where to write predictions.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    # PyCaret load_model expects a base name (without .pkl) OR full file path (works in recent versions)
    model = load_model(str(model_path).replace(".pkl", ""))

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)
    preds = predict_model(model, data=df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)
    print(f"✅ Wrote predictions to: {out_path}")


if __name__ == "__main__":
    main()
