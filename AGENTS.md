# XAInyPredictor Agent Guide

## Running the App
```bash
xainypredictor                                    # Uses port 8001
shiny run src/XAInyPredictor/app.py --port 8001    # Alternative
```
Default port in main.py is 8001, not 8000.

## Development Commands

- **Install dev dependencies**: `pip install -e ".[dev]"`
- **Setup pre-commit hooks**: `pre-commit install`
- **Run pre-commit manually**: `pre-commit run --all-files`
- **Build package**: `python3 -m build`
- **Generate docs**: `doxygen ./docs/Doxyfile`

## Commits
Do NOT execute any commits.

## Testing
The existing test file (`test/tests.py`) is a broken template - it imports `mymodule.template` which doesn't exist. No real tests are implemented.

## Project Structure
```
src/XAInyPredictor/
  main.py              # Entry point
  app.py               # Main Shiny app, server, UI layout
  modules/
    xai.py             # XAI model analysis (delta_xai, etc.)
    data_processing.py # Data cleaning, encoding, normalization
    model_registry.py  # Use case discovery and loading
  shinyapp/
    data_input.py      # Data Input tab (file upload, manual entry)
    data_exploration.py  # Data Exploration tab
    prediction.py      # Prediction tab
  data/
    <use_case>/        # One folder per use case (e.g., rai, mock)
      config.yml       # Use case configuration (features, labels, encoding)
      model.pkl        # Trained model formula
      feature_order.txt # Feature names in model order
      example_data.csv  # Sample data for this use case
```

## Use Case System
Each use case is a subfolder under `data/` with 4 required files. Add a new use case by creating a new subfolder with these files. The app's startup modal lets users select the initial use case, and the navbar dropdown allows runtime switching with a confirmation dialog. Model data is cached per use case to avoid redundant reloading on subsequent switches.

## Dependencies
- Runtime: requirements.txt (pandas, numpy, shiny, scikit-learn, etc.)
- Dev: pyproject.toml `[project.optional-dependencies]` - pre-commit, coverage, commitizen

## Linting
Pre-commit uses black, reorder-python-imports, hadolint, commitizen. Config in `.pre-commit-config.yaml`.