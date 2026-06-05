# XAInyPredictor

XAInyPredictor is a Shiny for Python research prototype for interpretable stratification support. It lets users select a configured use case, build a working input set, confirm that set for analysis, inspect feature context, review model-based stratification outputs, compare profiles, and export results for review.

The app supports both patient-level cohorts and candidate-level sets. Labels such as patient, cohort, candidate, candidate set, and reference candidates are configured per use case in `config.yml`.

> Research prototype notice: outputs support exploratory clinical or translational research workflows. Workflow utility, model interpretation, and clinical relevance should be validated with appropriate domain collaborators before operational use.

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

## Supported Use Cases

Use cases live under `src/XAInyPredictor/data/`.

- `rai`: RAI-R Predictor for thyroid cancer patient stratification.
- `mock`: Diabetes Risk Predictor demo use case.
- `neoag`: Neoantigen Candidate Prioritizer for peptide-HLA candidate prioritization.

The startup modal selects the initial use case. The top navigation selector can switch use case at runtime; switching clears the current working set and shows a loading state while the selected model context is prepared.

## Workflow

1. Start the app and choose a use case.
2. In **Data Input**, review the selected use case summary.
3. Build the current working set using one or more sources:
   - **Manual Entry**
   - **Upload File**
   - **Example Cohort** or **Example Candidate Set**
4. Review the table at the top of **Data Input**. The table remains the working set even when switching between input-source tabs.
5. Use **Delete Selected** or **Reset** if needed.
6. Click **Confirm** to lock the current working set for downstream analysis.
7. Continue to **Cohort/Candidate Context** and **Patient/Candidate Stratification**.

Steps 2 and 3 are blocked until the current input set is confirmed. If the working set changes, it must be confirmed again so downstream plots, tables, and exports use the latest data.

## Data Input

Uploaded CSV, TSV, or Excel files must contain the feature columns defined by the active use case. The safest option is to download the CSV template and data dictionary from **Data Input** after selecting the use case.

General format:

```csv
ID,Feature 1,Feature 2,Feature 3
1,value,value,value
```

Notes:

- `ID` is optional. If it is missing, the app creates sequential IDs.
- Numeric fields accept comma or dot decimal separators in manual entry.
- Numeric fields must respect configured min/max ranges.
- Categorical fields must use the allowed values from the active use case configuration.
- Text fields can be used for identifiers such as peptide sequences or HLA alleles.
- If required columns are missing, categories are invalid, or values are out of range, the app shows validation feedback before the set can be confirmed.

## Analysis Views

The labels below adapt to each use case.

- **Cohort/Candidate Context**: feature distributions, selected record highlighting, reference population selection, and feature summary statistics.
- **Patient/Candidate Stratification**: selected record score, decision threshold, assigned class/group, stratification table, and profile comparison controls.
- **Cohort/Candidate Set**: set-level summary and score distribution.
- **Model**: model card, intended use, validation status, limitations, and global feature importance.
- **Reference Patients/Candidates**: closest reference-neighbor context when a reference set is configured.

For the neoantigen use case, closest reference candidates come from the model development reference dataset included with the package. The app uses that dataset internally for scaling, thresholding, and similarity calculations, but the reference view reports anonymized neighbor ranks, distances, scores, and classes rather than row-level reference features.

## Downloads

The stratification view includes a **Download outputs** menu:

- Report package
- Stratification results
- Closest reference records
- Cohort/candidate set summary

## Use Case Configuration

Each use case folder requires:

```text
config.yml
model.pkl
feature_order.txt
example_data.csv
```

Use cases may also include additional files such as `reference_data.csv`.

`config.yml` controls:

- model name, description, and metadata;
- feature definitions, labels, defaults, ranges, allowed values, and roles;
- target column and positive/negative classes;
- entity labels used across the UI;
- tab titles, help text, validation copy, and download labels;
- model-specific options such as false-negative-rate defaults and reference-set limits.

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

No automated test suite is currently configured.
