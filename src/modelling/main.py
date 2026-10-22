"""Command-line entry point for model training."""

import argparse
from pathlib import Path

from src.modelling.training import train_and_save

DEFAULT_MODEL_PATH = Path("src/web_service/local_objects/abalone_model.pkl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Abalone ring-count model.")
    parser.add_argument("trainset_path", type=Path, help="CSV with the nine Abalone columns")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    result = train_and_save(args.trainset_path, args.model_path)
    print(f"Selected: {result['selected_model']}")
    print(f"Validation MAE: {result['selected_validation_mae']:.3f} rings")
    print(f"Held-out test MAE: {result['test']['mae']:.3f} rings")
    print(f"Saved full pipeline to {args.model_path}")


if __name__ == "__main__":
    main()
