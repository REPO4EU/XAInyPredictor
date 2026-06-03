import pandas as pd
from shiny import Inputs, Outputs, Session, module, reactive, render, ui
from XAInyPredictor.modules.data_processing import clean_data


def build_form_fields(config: dict) -> tuple:
    """Build form fields dynamically from config.
    Returns (columns, labels_dict) where columns is a list of ui.div for layout,
    and labels_dict maps input_id to display_label for server-side access.
    """
    features = config.get("features", [])
    labels = config.get("labels", {})

    if not features:
        return None, {}

    columns = []
    labels_dict = {}
    current_column = []

    for idx, feature in enumerate(features):
        feat_name = feature.get("name", "")
        display_name = feature.get("display_name", feat_name)
        input_type = feature.get("input_type", "numeric")
        values = feature.get("values", [])
        display_values = feature.get("display_values", {})
        default = feature.get("default", "")
        help_text = feature.get("help_text", "")
        min_val = feature.get("min", 0)
        max_val = feature.get("max", 100)
        step = feature.get("step", 1)

        input_id = f"in_{feat_name.replace(' ', '_').replace('(', '').replace(')', '')}"
        labels_dict[input_id] = feat_name

        field_ui = None

        if input_type == "numeric":
            field_ui = ui.input_numeric(
                input_id,
                display_name,
                value=default if default else min_val,
                min=min_val,
                max=max_val,
                step=step
            )
        elif input_type == "select" and values:
            choices = {}
            for v in values:
                display = display_values.get(v, v)
                choices[v] = display
            field_ui = ui.input_select(input_id, display_name, choices=choices, selected=default)

        if help_text and field_ui:
            field_ui = ui.div(
                field_ui,
                ui.tooltip(
                    ui.span(ui.tags.i(class_="glyphicon glyphicon-question-sign"), style="cursor: pointer; color: #007bc2; margin-left: 5px;"),
                    help_text,
                    placement="right"
                ),
                style="display: flex; align-items: center;"
            )

        current_column.append(ui.div(field_ui, style="margin-bottom: 10px;"))

        if (idx + 1) % 4 == 0 or idx == len(features) - 1:
            if current_column:
                col_ui = ui.div(
                    *current_column,
                    style="padding: 10px;"
                )
                columns.append(col_ui)
                current_column = []

    return columns, labels_dict


@module.ui
def data_input_ui(config=None):
    if config is None:
        config = {}

    labels = config.get("labels", {})
    return ui.layout_sidebar(
        ui.sidebar(
            ui.h4(labels.get("data_source", "Data Source")),
            ui.input_radio_buttons(
                "input_method",
                labels.get("select_method", "Select Method:"),
                {"form": labels.get("manual_entry", "Manual Entry"), "file": labels.get("upload_file", "Upload File"), "example": labels.get("example_cohort", "Example Cohort")}
            ),
            ui.panel_conditional(
                "input.input_method == 'file'",
                ui.input_file(
                    "input_dataset_file",
                    "Upload File (TSV, CSV, Excel)",
                    accept=[".tsv", ".csv", ".xlsx"],
                    multiple=False,
                ),
                ui.download_button(
                    "download_input_template",
                    "Download CSV template",
                    class_="btn-default btn-sm input-template-download",
                ),
                ui.download_button(
                    "download_data_dictionary",
                    "Download data dictionary",
                    class_="btn-default btn-sm input-template-download",
                ),
                ui.p("Upload a CSV, TSV, or Excel file with the same columns as the template.", class_="upload-template-hint"),
            ),
        ),

        ui.card(
            ui.card_header("Selected Use Case"),
            ui.output_ui("use_case_summary"),
            class_="use-case-summary-card",
        ),
        ui.output_ui("input_validation_ui"),
        ui.output_ui("input_validation_success_ui"),
        ui.br(),

        ui.panel_conditional(
            "input.input_method == 'form'",
            ui.card(
                ui.card_header(ui.tags.b("+ " + (labels.get("form", {}).get("add_patient_button", "New Patient Entry")))),
                ui.output_ui("form_fields"),
                ui.card_footer(
                    ui.input_action_button("btn_add_form", labels.get("form", {}).get("add_patient_button", "Add Patient to Cohort"), class_="btn-primary", width="100%")
                )
            ),
            ui.br()
        ),

        ui.card(
            ui.card_header(ui.tags.b(labels.get("form", {}).get("cohort_title", "Current Patient Cohort"))),
            ui.output_data_frame("out_patient_table"),
            ui.card_footer(
                ui.div(
                    ui.panel_conditional(
                        "input.input_method == 'form'",
                        ui.input_action_button("btn_delete_selected", labels.get("form", {}).get("delete_selected_button", "Delete Selected"), class_="btn-danger btn-sm cohort-action-button")
                    ),
                    ui.input_action_button("btn_reset_cohort", "Reset Cohort", class_="btn-default btn-sm cohort-action-button"),
                    class_="cohort-actions",
                )
            ),
            full_screen=True
        )
    )


@module.server
def server(input: Inputs, output: Outputs, session: Session, model_data, config_init, config_reactive=None):
    labels_dict = reactive.Value({})
    feature_cols = reactive.Value([])
    form_df = reactive.Value(pd.DataFrame())
    id_counter = reactive.Value(1)
    validation_errors = reactive.Value([])
    validation_success = reactive.Value(None)

    @reactive.Effect
    def _rebuild_fields():
        current_cfg = config_reactive.get() if config_reactive else config_init
        features = current_cfg.get("features", []) if current_cfg else []

        new_labels = {}
        new_cols = []
        if features:
            new_cols = [f["name"] for f in features]
            for f in features:
                feat_name = f.get("name", "")
                input_id = f"in_{feat_name.replace(' ', '_').replace('(', '').replace(')', '')}"
                new_labels[input_id] = feat_name

        labels_dict.set(new_labels)
        feature_cols.set(new_cols)
        form_df.set(pd.DataFrame(columns=['ID'] + new_cols))
        id_counter.set(1)
        validation_errors.set([])
        validation_success.set(None)

        # Force the radio button back to form on use-case change
        ui.update_radio_buttons("input_method", selected="form")

    output_data = reactive.Value(None)
    output_is_custom = reactive.Value(False)

    def _ensure_id(df):
        if df is None: return None
        if 'ID' not in df.columns:
            df.insert(0, 'ID', range(1, len(df) + 1))
        else:
            cols = ['ID'] + [c for c in df.columns if c != 'ID']
            df = df[cols]
        return df

    def _format_values(values, max_items=5):
        clean_values = [str(v) for v in values if pd.notna(v)]
        preview = clean_values[:max_items]
        suffix = "" if len(clean_values) <= max_items else f" and {len(clean_values) - max_items} more"
        return ", ".join(preview) + suffix

    def _template_values(feature, n_rows=5):
        if "default" in feature:
            default = feature.get("default")
        else:
            default = ""

        if feature.get("input_type") == "select":
            values = feature.get("values", [])
            if not values:
                return [default] * n_rows
            start_idx = values.index(default) if default in values else 0
            return [values[(start_idx + i) % len(values)] for i in range(n_rows)]

        min_val = feature.get("min")
        max_val = feature.get("max")
        step = feature.get("step", 1)

        if min_val is None or max_val is None:
            return [default] * n_rows

        midpoint = (float(min_val) + float(max_val)) / 2
        numeric_values = [
            default if default != "" else midpoint,
            min_val,
            max_val,
            midpoint,
            min(max(float(default if default != "" else midpoint) + float(step), float(min_val)), float(max_val)),
        ]

        return [round(value, 2) if isinstance(value, float) and not value.is_integer() else int(value) if isinstance(value, float) else value for value in numeric_values[:n_rows]]

    def _template_dataframe(config):
        features = config.get("features", []) if config else []
        n_rows = 5
        rows = [{"ID": idx + 1} for idx in range(n_rows)]
        for feature in features:
            feat_name = feature.get("name")
            if feat_name:
                values = _template_values(feature, n_rows)
                for idx, row in enumerate(rows):
                    row[feat_name] = values[idx]
        return pd.DataFrame(rows)

    def _data_dictionary_dataframe(config):
        rows = []
        features = config.get("features", []) if config else []
        for feature in features:
            allowed_values = feature.get("values", [])
            rows.append(
                {
                    "Variable": feature.get("name", ""),
                    "Display Name": feature.get("display_name", feature.get("name", "")),
                    "Type": feature.get("type", ""),
                    "Input Type": feature.get("input_type", ""),
                    "Required": "YES",
                    "Allowed Values": ", ".join(map(str, allowed_values)),
                    "Min": feature.get("min", ""),
                    "Max": feature.get("max", ""),
                    "Default": feature.get("default", ""),
                    "Description": feature.get("help_text", ""),
                }
            )
        return pd.DataFrame(rows)

    def _validation_success_message(df, config, method):
        features = config.get("features", []) if config else []
        required_cols = [feature.get("name") for feature in features if feature.get("name")]
        matched_cols = [col for col in required_cols if col in df.columns]
        source = {
            "form": "manual cohort",
            "file": "uploaded file",
            "example": "example cohort",
        }.get(method, "cohort")
        return f"{len(df)} patient(s) loaded from {source}. {len(matched_cols)}/{len(required_cols)} required columns matched. No validation issues found."

    def _validate_input_data(df, config):
        if df is None or df.empty or not config:
            return []

        errors = []
        features = config.get("features", [])
        required_cols = [feature.get("name") for feature in features if feature.get("name")]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            errors.append(f"Missing required column(s): {', '.join(missing_cols)}.")

        for feature in features:
            feat_name = feature.get("name")
            if not feat_name or feat_name not in df.columns:
                continue

            display_name = feature.get("display_name", feat_name)
            input_type = feature.get("input_type", "numeric")
            series = df[feat_name]
            non_missing = series[series.notna()]

            if non_missing.empty:
                continue

            if input_type == "numeric":
                numeric_series = pd.to_numeric(non_missing, errors="coerce")
                invalid_values = non_missing[numeric_series.isna()].drop_duplicates().tolist()
                if invalid_values:
                    errors.append(f"{display_name} must be numeric. Invalid value(s): {_format_values(invalid_values)}.")
                    continue

                min_val = feature.get("min")
                max_val = feature.get("max")
                range_mask = pd.Series(False, index=numeric_series.index)
                if min_val is not None:
                    range_mask = range_mask | (numeric_series < min_val)
                if max_val is not None:
                    range_mask = range_mask | (numeric_series > max_val)

                if range_mask.any():
                    bad_values = non_missing[range_mask].drop_duplicates().tolist()
                    if min_val is not None and max_val is not None:
                        range_text = f"between {min_val} and {max_val}"
                    elif min_val is not None:
                        range_text = f"greater than or equal to {min_val}"
                    else:
                        range_text = f"less than or equal to {max_val}"
                    errors.append(f"{display_name} must be {range_text}. Out-of-range value(s): {_format_values(bad_values)}.")

            elif input_type == "select":
                allowed_values = feature.get("values", [])
                if not allowed_values:
                    continue

                invalid_values = sorted(set(non_missing.astype(str)) - set(map(str, allowed_values)))
                if invalid_values:
                    errors.append(
                        f"{display_name} has unexpected categor{('y' if len(invalid_values) == 1 else 'ies')}: "
                        f"{_format_values(invalid_values)}. Allowed: {', '.join(map(str, allowed_values))}."
                    )

        return errors

    @reactive.Effect
    @reactive.event(input.btn_add_form)
    def _add_patient():
        current_df = form_df.get()
        new_id = id_counter()
        current_labels = labels_dict.get()

        row_data = {'ID': new_id}
        for input_id, feat_name in current_labels.items():
            val = getattr(input, input_id, None)()
            if val is not None:
                if isinstance(val, str):
                    row_data[feat_name] = val
                else:
                    row_data[feat_name] = float(val)

        new_row = pd.DataFrame([row_data])
        updated_df = pd.concat([current_df, new_row], ignore_index=True)
        form_df.set(updated_df)
        id_counter.set(new_id + 1)
        ui.notification_show("Patient added.", type="message")

    @reactive.Effect
    @reactive.event(input.btn_delete_selected)
    def _delete_patient():
        selected_rows = input.out_patient_table_selected_rows()
        if not selected_rows:
            ui.notification_show("No rows selected.", type="warning")
            return

        current_df = form_df.get()
        updated_df = current_df.drop(index=list(selected_rows)).reset_index(drop=True)
        form_df.set(updated_df)
        ui.notification_show(f"Deleted {len(selected_rows)} patient(s).", type="message")

    @reactive.Effect
    @reactive.event(input.btn_reset_cohort)
    def _reset_cohort():
        current_cols = feature_cols.get()
        form_df.set(pd.DataFrame(columns=['ID'] + current_cols))
        id_counter.set(1)
        validation_errors.set([])
        validation_success.set(None)
        output_data.set(None)
        output_is_custom.set(False)
        ui.update_radio_buttons("input_method", selected="form")
        ui.notification_show("Cohort reset.", type="message")

    @reactive.Effect
    def _update_output_pipeline():
        method = str(input.input_method())
        raw_df = None
        is_custom = False
        current_cfg = config_reactive.get() if config_reactive else config_init

        md = model_data.get()
        example_raw_data = md.get("X_TEST_RAW")

        if method == "example":
            if example_raw_data is not None:
                raw_df = example_raw_data.copy()
            is_custom = False

        elif method == "form":
            raw_df = form_df.get()
            is_custom = True

        elif method == "file":
            file_info = input.input_dataset_file()
            is_custom = True
            if not file_info:
                output_data.set(None)
                output_is_custom.set(is_custom)
                validation_errors.set([])
                validation_success.set(None)
                return

            file_path = file_info[0]["datapath"]
            try:
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(".tsv"):
                    df = pd.read_csv(file_path, sep="\t")
                elif file_path.endswith(".xlsx"):
                    df = pd.read_excel(file_path)
                else:
                    raise ValueError("Unsupported file format!")
                raw_df = clean_data(df)
            except Exception as e:
                ui.notification_show(f"Error reading file: {e}", type="error")
                validation_errors.set([f"Could not read uploaded file: {e}"])
                validation_success.set(None)
                raw_df = None

        if raw_df is not None and not raw_df.empty:
            clean_df = _ensure_id(raw_df)
            errors = _validate_input_data(clean_df, current_cfg)
            validation_errors.set(errors)
            if errors:
                validation_success.set(None)
                output_data.set(None)
                output_is_custom.set(is_custom)
                ui.notification_show("Input validation failed. See details in Data Input.", type="error")
                return
            validation_success.set(_validation_success_message(clean_df, current_cfg, method))
            output_data.set(clean_df)
        else:
            validation_errors.set([])
            validation_success.set(None)
            output_data.set(None)

        output_is_custom.set(is_custom)

    @output
    @render.ui
    def input_validation_ui():
        errors = validation_errors.get()
        if not errors:
            return None

        return ui.div(
            ui.div("Input validation failed", class_="input-validation-title"),
            ui.tags.ul(*[ui.tags.li(error) for error in errors]),
            class_="input-validation-alert",
        )

    @output
    @render.ui
    def input_validation_success_ui():
        message = validation_success.get()
        if not message:
            return None

        return ui.div(
            ui.div("Input validation passed", class_="input-validation-title"),
            ui.p(message, class_="input-validation-success-copy"),
            class_="input-validation-success",
        )

    @render.download(filename="input_template.csv")
    def download_input_template():
        current_cfg = config_reactive.get() if config_reactive else config_init
        yield _template_dataframe(current_cfg or {}).to_csv(index=False)

    @render.download(filename="data_dictionary.csv")
    def download_data_dictionary():
        current_cfg = config_reactive.get() if config_reactive else config_init
        yield _data_dictionary_dataframe(current_cfg or {}).to_csv(index=False)

    @output
    @render.ui
    def use_case_summary():
        current_cfg = config_reactive.get() if config_reactive else config_init
        if not current_cfg:
            return ui.p("No use case selected.")

        features = current_cfg.get("features", [])
        labels = current_cfg.get("labels", {})
        positive_label = labels.get("positive_class_label", current_cfg.get("positive_class", "Positive"))
        negative_label = labels.get("negative_class_label", current_cfg.get("negative_class", "Negative"))

        return ui.div(
            ui.div(
                ui.div("Model objective", class_="use-case-summary-label"),
                ui.div(current_cfg.get("description", "Patient stratification model"), class_="use-case-summary-value"),
                class_="use-case-summary-item use-case-summary-wide",
            ),
            ui.div(
                ui.div("Patient groups", class_="use-case-summary-label"),
                ui.div(f"{negative_label} / {positive_label}", class_="use-case-summary-value"),
                class_="use-case-summary-item",
            ),
            ui.div(
                ui.div("Input variables", class_="use-case-summary-label"),
                ui.div(str(len(features)), class_="use-case-summary-value"),
                class_="use-case-summary-item",
            ),
            class_="use-case-summary",
        )

    @output
    @render.ui
    def form_fields():
        current_cfg = config_reactive.get() if config_reactive else config_init
        form_columns, _ = build_form_fields(current_cfg or {})
        if form_columns:
            return ui.layout_columns(
                *form_columns, col_widths=tuple([4] * len(form_columns))
            )
        return ui.p("No features configured.")

    @output
    @render.data_frame
    def out_patient_table():
        display_df = output_data.get()

        if display_df is None or display_df.empty:
            return render.DataGrid(
                pd.DataFrame({"Message": ["No data uploaded."]}),
                width="100%",
                selection_mode="none"
            )

        display_df = display_df.copy()
        display_df.columns = [c.replace('_', ' ') for c in display_df.columns]
        display_df = _ensure_id(display_df)

        return render.DataGrid(
            display_df,
            width="100%",
            selection_mode="rows"
        )

    return {"data": output_data, "is_custom": output_is_custom}
