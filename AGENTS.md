# XAInyPredictor Agent Guide

## Running the App
```bash
xainypredictor                                    # Uses port 8001
shiny run src/XAInyPredictor/app.py --port 8001    # Alternative
```

## Development Commands
- **Install dev deps**: `pip install -e ".[dev]"`
- **Setup pre-commit**: `pre-commit install`
- **Run pre-commit**: `pre-commit run --all-files`
- **Build package**: `python3 -m build`
- **Generate docs**: `doxygen ./docs/Doxyfile`

## Testing
No tests exist. No test framework is properly configured.

## Type Checking
None configured (no mypy, pyright, etc.).

## Project Structure
```
src/XAInyPredictor/
  main.py              # Entry point (port 8001)
  app.py               # Main Shiny app
  modules/
    xai.py             # XAI analysis using KAN-based model (delta_xai)
    data_processing.py # Data cleaning, encoding, normalization
    model_registry.py  # Use case discovery/loading
  shinyapp/
    data_input.py      # Data Input tab
    data_exploration.py # Data Exploration tab
    prediction.py      # Prediction tab
  data/
    <use_case>/        # One folder per use case (rai, mock)
      config.yml       # Use case config (features, labels, encoding)
      model.pkl        # Trained KAN model formula
      feature_order.txt # Feature names in model order
      example_data.csv  # Sample data
```

## Use Case System
Each use case is a subfolder under `data/` with 4 required files. Add a new use case by creating a new subfolder with these files. The app's startup modal lets users select the initial use case, and the navbar dropdown allows runtime switching with a confirmation dialog. Model data is cached per use case to avoid redundant reloading on subsequent switches.

## Dependencies
- Runtime: requirements.txt (pandas, numpy, shiny, scikit-learn, sympy, etc.)
- Dev: pyproject.toml `[project.optional-dependencies]` - pre-commit, coverage, commitizen

## Linting
Pre-commit uses black, reorder-python-imports, hadolint, commitizen. Config in `.pre-commit-config.yaml`.

## Commits
Do not commit changes without explicit user approval.
