import os
from pathlib import Path
from typing import Any
import yaml
import pandas as pd


class UseCaseNotFoundError(Exception):
    pass


class InvalidUseCaseError(Exception):
    pass


REQUIRED_FILES = ["config.yml", "model.pkl", "feature_order.txt", "example_data.csv"]


def get_use_case_path(use_case_name: str) -> Path:
    base_dir = Path(__file__).parent.parent / "data" / use_case_name
    return base_dir


def validate_use_case_folder(use_case_path: Path) -> None:
    missing_files = [f for f in REQUIRED_FILES if not (use_case_path / f).exists()]
    if missing_files:
        raise InvalidUseCaseError(
            f"Use case '{use_case_path.name}' is missing required files: {missing_files}"
        )


def load_config(use_case_path: Path) -> dict:
    config_file = use_case_path / "config.yml"
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def load_model(use_case_path: Path) -> Any:
    import pickle
    model_file = use_case_path / "model.pkl"
    with open(model_file, "rb") as f:
        return pickle.load(f)


def load_feature_order(use_case_path: Path) -> list:
    feature_file = use_case_path / "feature_order.txt"
    with open(feature_file, "r") as f:
        content = f.read()
        for line in content.split("\n"):
            if line.startswith("Variables:"):
                return eval(line.strip().split(": ")[1])
        raise InvalidUseCaseError(f"Could not find 'Variables:' in {feature_file}")


def load_example_data(use_case_path: Path) -> pd.DataFrame:
    example_file = use_case_path / "example_data.csv"
    return pd.read_csv(example_file)


def discover_use_cases() -> dict:
    base_dir = Path(__file__).parent.parent / "data"
    use_cases = {}

    if not base_dir.exists():
        return use_cases

    for item in base_dir.iterdir():
        if not item.is_dir():
            continue

        try:
            validate_use_case_folder(item)
            config = load_config(item)
            use_cases[item.name] = {
                "path": item,
                "config": config,
                "display_name": config.get("name", item.name),
                "description": config.get("description", ""),
            }
        except (InvalidUseCaseError, yaml.YAMLError) as e:
            continue

    return use_cases


def load_use_case(use_case_name: str) -> dict:
    use_case_path = get_use_case_path(use_case_name)

    if not use_case_path.exists():
        raise UseCaseNotFoundError(f"Use case '{use_case_name}' not found")

    validate_use_case_folder(use_case_path)

    config = load_config(use_case_path)
    model = load_model(use_case_path)
    feature_order = load_feature_order(use_case_path)
    example_data = load_example_data(use_case_path)

    return {
        "path": use_case_path,
        "config": config,
        "model": model,
        "feature_order": feature_order,
        "example_data": example_data,
    }