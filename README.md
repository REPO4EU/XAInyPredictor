# XAInyPredictor

XAInyPredictor is a Shiny for Python research prototype for patient stratification. It lets users load or enter patient cohorts, inspect cohort context, generate model-based stratification outputs, compare patient profiles with a reference cohort, and export results for review.

The application currently supports multiple use cases through the `src/XAInyPredictor/data/<use_case>/` registry. Each use case defines its own input features, labels, model files, example data, and display text.

> Research prototype notice: outputs are intended for cohort-level clinical decision support research. Workflow utility and interpretation should be validated with clinical collaborators.

## Run The App

From the repository root:

```powershell
xainypredictor
```

The default URL is:

```text
http://127.0.0.1:8001
```

Alternative command:

```powershell
shiny run src\XAInyPredictor\app.py --port 8001
```

If port `8001` is already in use, choose another port:

```powershell
shiny run src\XAInyPredictor\app.py --port 8002
```

## Demo Flow

1. Start the app and choose a use case in the startup modal.
2. In **Data Input**, add patients manually, upload a file, or load the example cohort.
3. If uploading data, use **Download CSV template** to get example rows for the active use case, and **Download data dictionary** to inspect allowed values, ranges, and feature descriptions.
4. In **Cohort Context**, inspect feature distributions and compare a selected patient with the input or model reference population.
5. In **Patient Stratification**, review:
   - **Patient**: selected patient score, decision threshold, patient group, and profile comparison.
   - **Cohort**: cohort-level score distribution and group counts.
   - **Model**: model card and global feature importance.
   - **Reference Patients**: closest reference patients and similarity context.
6. Export the full ZIP report package, or download stratification results, closest patients, and cohort summary separately.
7. Use **Reset Cohort** when you want to clear the current cohort without switching use case.
8. Switch use cases from the top navigation dropdown when needed. The app asks for confirmation because switching clears the current cohort.

## CSV Upload Format

Uploaded CSV, TSV, or Excel files must contain the feature columns defined by the active use case. The safest option is to download the template from **Data Input** after selecting the use case.

General format:

```csv
ID,Feature 1,Feature 2,Feature 3
1,value,value,value
```

Notes:

- `ID` is optional. If it is missing, the app creates sequential patient IDs.
- Numeric fields must contain numeric values and respect the configured min/max range.
- Categorical fields must use the exact allowed values from the use case configuration.
- If a file has missing columns, invalid categories, or out-of-range values, the app shows a validation message before prediction.

## Use Cases

Use cases live under:

```text
src/XAInyPredictor/data/
```

Each use case folder requires:

```text
config.yml
model.pkl
feature_order.txt
example_data.csv
```

The `config.yml` file controls:

- feature names, labels, defaults, ranges, and allowed categories;
- target and patient group labels;
- tab titles and help text;
- encoding used to transform raw input into model-ready data.

## Development

Install development dependencies:

```powershell
pip install -e ".[dev]"
```

Run pre-commit:

```powershell
pre-commit run --all-files
```

Build the package:

```powershell
python -m build
```

Generate docs:

```powershell
doxygen .\docs\Doxyfile
```

## Testing Status

No automated test suite is currently configured. The existing `test/tests.py` file is a broken template leftover and should not be treated as a valid test entry point.
